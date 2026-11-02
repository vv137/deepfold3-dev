"""Fused sigmoid-gated dual GEMM Triton kernel for DeepFold3.

Replaces the triangle multiply pre-einsum pattern:

    proj = proj_linear(z0)              # GEMM 1: [M, C] → [M, 2C]
    gate = sigmoid(gate_linear(z0))     # GEMM 2: [M, C] → [M, 2C] + sigmoid
    out  = proj * gate                  # elementwise

with a single kernel that computes both GEMMs in one pass:

    out = sigmoid(x @ W1.T + b1) * (x @ W2.T + b2) [* mask[:, None]]

Key design insights (adapted from NVIDIA cuEquivariance patterns):

1. SHARED INPUT TILES — Both GEMMs read the same input x. In a tiled GEMM,
   we load the x tile once from global memory and reuse it for two dot
   products (gate and proj), halving the x bandwidth relative to two
   separate GEMMs.

2. TWO_INPUTS MODE — Optionally use separate x1, x2 for the two GEMMs:
   sigmoid(x1 @ W1.T) * (x2 @ W2.T). Useful when gate and proj inputs
   differ (e.g., after separate LayerNorm branches).

3. BIAS SUPPORT — Optional bias vectors b1, b2 added after GEMM
   accumulation before the fused epilogue.

4. MASK FUSION — Optional per-row mask applied in epilogue:
   output *= mask[:, None]. This fuses the AdaLN-zero gate into the kernel.

5. TF32 PRECISION — Configurable precision via precision_dot helper.

6. FUSED EPILOGUE — The sigmoid + elementwise multiply happens in
   registers immediately after the K-accumulation loop, before writing to
   global memory. This eliminates the intermediate gate tensor entirely.

7. OPTIONAL TRANSPOSE_OUT — Write the output in [N, M] layout so
   downstream einsum contractions see contiguous memory along the
   contracted dimension.

Falls back to separate proj + sigmoid(gate) + multiply when Triton is
unavailable or input is not on CUDA.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    from deepfold3.model.triton import precision_dot
    from deepfold3.model.triton.cache import persistent_autotune

    # ── Autotune config space ────────────────────────────────────────────
    # Covers the DeepFold3 triangle-mul shapes: K=128, N=256 (2×C_z).
    # num_stages 1-4: 1 for Ampere (48KB shared), 2-3 for Ada/Blackwell, 4 for Hopper.
    _AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_SIZE_M": 8}, num_stages=ns, num_warps=nw)
        for bm in (64, 128)
        for bn in (32, 64, 128)
        for bk in (32, 64)
        for ns in (1, 2, 3, 4)
        for nw in (4, 8)
    ]

    @persistent_autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _fused_dual_gemm_fwd_kernel(
        # ── Pointers ──
        X_ptr,  # [M, K] input for GEMM 1 (and GEMM 2 when TWO_INPUTS=False)
        X2_ptr,  # [M, K] input for GEMM 2 (only used when TWO_INPUTS=True)
        W1_ptr,  # [N, K] gate weights
        W2_ptr,  # [N, K] proj weights
        B1_ptr,  # [N] gate bias (only used when HAS_B1=True)
        B2_ptr,  # [N] proj bias (only used when HAS_B2=True)
        MASK_ptr,  # [M] per-row mask (only used when APPLY_MASK=True)
        OUT_ptr,  # [M, N] output (or [N, M] if TRANSPOSE_OUT)
        # ── Dimensions ──
        M,
        N,
        K,
        # ── Strides ──
        stride_xm,
        stride_xk,
        stride_x2m,
        stride_x2k,
        stride_w1n,
        stride_w1k,
        stride_w2n,
        stride_w2k,
        stride_om,
        stride_on,
        # ── Compile-time constants ──
        TWO_INPUTS: tl.constexpr,
        HAS_B1: tl.constexpr,
        HAS_B2: tl.constexpr,
        APPLY_MASK: tl.constexpr,
        TRANSPOSE_OUT: tl.constexpr,
        PRECISION: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """Fused dual GEMM: out = sigmoid(x @ W1.T + b1) * (x @ W2.T + b2) [* mask].

        Grid: (ceil(M/BLOCK_M) * ceil(N/BLOCK_N),)
        Each program computes one [BLOCK_M, BLOCK_N] output tile.
        """
        pid = tl.program_id(0).to(tl.int64)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)

        # Grouped ordering for L2 cache locality
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        if TWO_INPUTS:
            x2_ptrs = X2_ptr + offs_m[:, None] * stride_x2m + offs_k[None, :] * stride_x2k
        w1_ptrs = W1_ptr + offs_n[None, :] * stride_w1n + offs_k[:, None] * stride_w1k
        w2_ptrs = W2_ptr + offs_n[None, :] * stride_w2n + offs_k[:, None] * stride_w2k

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_proj = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        n_mask = offs_n[None, :] < N

        for _k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - _k * BLOCK_K
            k_mask = offs_k < k_remaining
            m_mask = offs_m[:, None] < M

            x = tl.load(x_ptrs, mask=m_mask & k_mask[None, :], other=0.0)
            w1 = tl.load(w1_ptrs, mask=k_mask[:, None] & n_mask, other=0.0)
            w2 = tl.load(w2_ptrs, mask=k_mask[:, None] & n_mask, other=0.0)

            if TWO_INPUTS:
                x2 = tl.load(x2_ptrs, mask=m_mask & k_mask[None, :], other=0.0)
                acc_gate = precision_dot(x, w1, acc_gate, PRECISION)
                acc_proj = precision_dot(x2, w2, acc_proj, PRECISION)
                x2_ptrs += BLOCK_K * stride_x2k
            else:
                acc_gate = precision_dot(x, w1, acc_gate, PRECISION)
                acc_proj = precision_dot(x, w2, acc_proj, PRECISION)

            x_ptrs += BLOCK_K * stride_xk
            w1_ptrs += BLOCK_K * stride_w1k
            w2_ptrs += BLOCK_K * stride_w2k

        if HAS_B1:
            b1 = tl.load(B1_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc_gate += b1[None, :]
        if HAS_B2:
            b2 = tl.load(B2_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc_proj += b2[None, :]

        sig = tl.sigmoid(acc_gate)
        out = sig * acc_proj

        if APPLY_MASK:
            mask_vals = tl.load(MASK_ptr + offs_m, mask=offs_m < M, other=0.0)
            out = out * mask_vals[:, None]

        out = out.to(X_ptr.type.element_ty)

        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if TRANSPOSE_OUT:
            out_ptrs = OUT_ptr + offs_n[None, :] * stride_om + offs_m[:, None] * stride_on
            tl.store(out_ptrs, out, mask=out_mask)
        else:
            out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
            tl.store(out_ptrs, out, mask=out_mask)

    @persistent_autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K"], reset_to_zero=["GRAD_MASK_ptr"])
    @triton.jit
    def _fused_dual_gemm_bwd_pregemm_kernel(
        # ── Pointers ──
        GRAD_OUT_ptr,  # [M, N] upstream gradient
        X_ptr,  # [M, K] saved input (GEMM 1)
        X2_ptr,  # [M, K] saved input (GEMM 2, only when TWO_INPUTS)
        W1_ptr,  # [N, K] gate weights
        W2_ptr,  # [N, K] proj weights
        B1_ptr,  # [N] gate bias (only when HAS_B1)
        B2_ptr,  # [N] proj bias (only when HAS_B2)
        MASK_ptr,  # [M] per-row mask (only when APPLY_MASK)
        GRAD_XW1_ptr,  # [M, N] output: d_loss/d(gate_logits)
        GRAD_XW2_ptr,  # [M, N] output: d_loss/d(proj_output)
        GRAD_MASK_ptr,  # [M] output: d_loss/d(mask) (only when APPLY_MASK)
        # ── Dimensions ──
        M,
        N,
        K,
        # ── Strides ──
        stride_gm,
        stride_gn,
        stride_xm,
        stride_xk,
        stride_x2m,
        stride_x2k,
        stride_w1n,
        stride_w1k,
        stride_w2n,
        stride_w2k,
        # ── Compile-time constants ──
        TWO_INPUTS: tl.constexpr,
        HAS_B1: tl.constexpr,
        HAS_B2: tl.constexpr,
        APPLY_MASK: tl.constexpr,
        PRECISION: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """Backward pre-GEMM: recompute sigmoid, derive local gradient factors."""
        pid = tl.program_id(0).to(tl.int64)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        if TWO_INPUTS:
            x2_ptrs = X2_ptr + offs_m[:, None] * stride_x2m + offs_k[None, :] * stride_x2k
        w1_ptrs = W1_ptr + offs_n[None, :] * stride_w1n + offs_k[:, None] * stride_w1k
        w2_ptrs = W2_ptr + offs_n[None, :] * stride_w2n + offs_k[:, None] * stride_w2k
        n_mask = offs_n[None, :] < N

        # Recompute both forward accumulations
        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_proj = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for _k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - _k * BLOCK_K
            k_mask = offs_k < k_remaining
            m_mask = offs_m[:, None] < M

            x = tl.load(x_ptrs, mask=m_mask & k_mask[None, :], other=0.0)
            w1 = tl.load(w1_ptrs, mask=k_mask[:, None] & n_mask, other=0.0)
            w2 = tl.load(w2_ptrs, mask=k_mask[:, None] & n_mask, other=0.0)

            if TWO_INPUTS:
                x2 = tl.load(x2_ptrs, mask=m_mask & k_mask[None, :], other=0.0)
                acc_gate = precision_dot(x, w1, acc_gate, PRECISION)
                acc_proj = precision_dot(x2, w2, acc_proj, PRECISION)
                x2_ptrs += BLOCK_K * stride_x2k
            else:
                acc_gate = precision_dot(x, w1, acc_gate, PRECISION)
                acc_proj = precision_dot(x, w2, acc_proj, PRECISION)

            x_ptrs += BLOCK_K * stride_xk
            w1_ptrs += BLOCK_K * stride_w1k
            w2_ptrs += BLOCK_K * stride_w2k

        if HAS_B1:
            b1 = tl.load(B1_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc_gate += b1[None, :]
        if HAS_B2:
            b2 = tl.load(B2_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc_proj += b2[None, :]

        # Load upstream gradient
        grad_out = tl.load(
            GRAD_OUT_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)

        sig = tl.sigmoid(acc_gate)

        if APPLY_MASK:
            mask_vals = tl.load(MASK_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
            # grad_mask: sum over N of (grad_out * sig * proj) per row
            grad_mask_partial = tl.sum(grad_out * sig * acc_proj, axis=1)
            tl.atomic_add(GRAD_MASK_ptr + offs_m, grad_mask_partial, mask=offs_m < M)
            grad_out_eff = grad_out * mask_vals[:, None]
        else:
            grad_out_eff = grad_out

        grad_xw2 = (grad_out_eff * sig).to(X_ptr.type.element_ty)
        grad_xw1 = (grad_out_eff * acc_proj * sig * (1.0 - sig)).to(X_ptr.type.element_ty)

        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(GRAD_XW1_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn, grad_xw1, mask=out_mask)
        tl.store(GRAD_XW2_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn, grad_xw2, mask=out_mask)


class FusedGatedDualGEMMFunction(torch.autograd.Function):
    """Autograd wrapper for the fused dual GEMM kernel.

    Forward fuses two GEMMs + sigmoid + multiply into one kernel.
    Backward uses the cuEQ split pattern:
      1. Pre-GEMM kernel: recompute sigmoid, derive grad_xw1/grad_xw2  (fused)
      2. Standard matmul: grad_x = grad_xw1 @ W1 + grad_xw2 @ W2      (cuBLAS)
      3. Standard matmul: grad_W1 = grad_xw1.T @ x, grad_W2 = ...      (cuBLAS)
      4. Bias grads: grad_b1 = grad_xw1.sum(0), grad_b2 = ...
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, x, w_gate, w_proj, x2, b_gate, b_proj, mask, precision):
        x = x.contiguous()
        w_gate = w_gate.contiguous()
        w_proj = w_proj.contiguous()
        M, K = x.shape
        N = w_gate.shape[0]

        two_inputs = x2 is not None
        has_b1 = b_gate is not None
        has_b2 = b_proj is not None
        apply_mask = mask is not None

        if two_inputs:
            x2 = x2.contiguous()
        if has_b1:
            b_gate = b_gate.contiguous()
        if has_b2:
            b_proj = b_proj.contiguous()
        if apply_mask:
            mask = mask.contiguous()

        out = torch.empty((M, N), dtype=x.dtype, device=x.device)

        # Dummy pointers for optional arguments
        x2_ptr = x2 if two_inputs else x
        b1_ptr = b_gate if has_b1 else x
        b2_ptr = b_proj if has_b2 else x
        mask_ptr = mask if apply_mask else x

        def grid(META):
            return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

        _fused_dual_gemm_fwd_kernel[grid](
            x,
            x2_ptr,
            w_gate,
            w_proj,
            b1_ptr,
            b2_ptr,
            mask_ptr,
            out,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            x2.stride(0) if two_inputs else x.stride(0),
            x2.stride(1) if two_inputs else x.stride(1),
            w_gate.stride(0),
            w_gate.stride(1),
            w_proj.stride(0),
            w_proj.stride(1),
            out.stride(0),
            out.stride(1),
            TWO_INPUTS=two_inputs,
            HAS_B1=has_b1,
            HAS_B2=has_b2,
            APPLY_MASK=apply_mask,
            TRANSPOSE_OUT=False,
            PRECISION=precision,
        )

        # Only save tensors needed in backward: b_gate/b_proj are not read there
        # (bias grads are derived from grad_xw1/grad_xw2 sums, not from saved biases).
        ctx.save_for_backward(x, w_gate, w_proj, x2, mask)
        ctx.two_inputs = two_inputs
        ctx.has_b1 = has_b1
        ctx.has_b2 = has_b2
        ctx.b_gate = b_gate  # stored on ctx (not save_for_backward) for kernel recompute
        ctx.b_proj = b_proj
        ctx.apply_mask = apply_mask
        ctx.precision = precision
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        x, w_gate, w_proj, x2, mask = ctx.saved_tensors
        two_inputs = ctx.two_inputs
        b_gate = ctx.b_gate
        b_proj = ctx.b_proj
        has_b1 = ctx.has_b1
        has_b2 = ctx.has_b2
        apply_mask = ctx.apply_mask
        precision = ctx.precision

        M, K = x.shape
        N = w_gate.shape[0]
        grad_out = grad_out.contiguous()

        grad_xw1 = torch.empty((M, N), dtype=x.dtype, device=x.device)
        grad_xw2 = torch.empty((M, N), dtype=x.dtype, device=x.device)

        grad_mask_buf = torch.zeros(M, dtype=torch.float32, device=x.device) if apply_mask else x

        x2_ptr = x2 if two_inputs else x
        b1_ptr = b_gate if has_b1 else x
        b2_ptr = b_proj if has_b2 else x
        mask_ptr = mask if apply_mask else x

        def grid(META):
            return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

        _fused_dual_gemm_bwd_pregemm_kernel[grid](
            grad_out,
            x,
            x2_ptr,
            w_gate,
            w_proj,
            b1_ptr,
            b2_ptr,
            mask_ptr,
            grad_xw1,
            grad_xw2,
            grad_mask_buf,
            M,
            N,
            K,
            grad_out.stride(0),
            grad_out.stride(1),
            x.stride(0),
            x.stride(1),
            x2.stride(0) if two_inputs else x.stride(0),
            x2.stride(1) if two_inputs else x.stride(1),
            w_gate.stride(0),
            w_gate.stride(1),
            w_proj.stride(0),
            w_proj.stride(1),
            TWO_INPUTS=two_inputs,
            HAS_B1=has_b1,
            HAS_B2=has_b2,
            APPLY_MASK=apply_mask,
            PRECISION=precision,
        )

        # cuBLAS GEMMs for weight and input gradients
        if two_inputs:
            grad_x = grad_xw1 @ w_gate
            grad_x2 = grad_xw2 @ w_proj
            grad_w_proj = grad_xw2.T @ x2
        else:
            grad_x = grad_xw1 @ w_gate + grad_xw2 @ w_proj
            grad_x2 = None
            grad_w_proj = grad_xw2.T @ x

        grad_w_gate = grad_xw1.T @ x
        grad_b_gate = grad_xw1.sum(dim=0) if has_b1 else None
        grad_b_proj = grad_xw2.sum(dim=0) if has_b2 else None
        grad_mask_out = grad_mask_buf.to(mask.dtype) if apply_mask else None

        return grad_x, grad_w_gate, grad_w_proj, grad_x2, grad_b_gate, grad_b_proj, grad_mask_out, None


def fused_gated_dual_gemm(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_proj: torch.Tensor,
    x2: torch.Tensor | None = None,
    b_gate: torch.Tensor | None = None,
    b_proj: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    precision: int = 0,
) -> torch.Tensor:
    """Compute sigmoid(x @ W_gate.T + b1) * (x2 @ W_proj.T + b2) [* mask] in a single kernel.

    Args:
        x: Input [*, K] — flattened to [M, K] internally.
        w_gate: Gate weight [N, K].
        w_proj: Projection weight [N, K].
        x2: Optional separate input for proj GEMM [*, K]. When None, x is
            shared for both GEMMs.
        b_gate: Optional gate bias [N].
        b_proj: Optional proj bias [N].
        mask: Optional per-row mask [*] — broadcast as mask[:, None] in epilogue.
        precision: TF32 precision mode (0=DEFAULT, 1=TF32, 2=TF32x3, 3=IEEE).

    Returns:
        Output [*, N] — same leading dims as x, last dim N.
    """
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1])
    x2_flat = x2.reshape(-1, x2.shape[-1]) if x2 is not None else None
    mask_flat = mask.reshape(-1) if mask is not None else None
    out = FusedGatedDualGEMMFunction.apply(x_flat, w_gate, w_proj, x2_flat, b_gate, b_proj, mask_flat, precision)
    return out.view(*shape[:-1], w_gate.shape[0])


class FusedGatedDualGEMM(nn.Module):
    """Fused sigmoid-gated dual GEMM kernel — STATELESS.

    Drop-in replacement for the triangle-multiply pre-einsum pattern:
        proj = proj_linear(x)
        gate = sigmoid(gate_linear(x))
        out = proj * gate

    Uses a single Triton kernel that shares the input tile load across
    both GEMMs and fuses the sigmoid + multiply into the epilogue.

    Holds no parameters; callers pass w_gate and w_proj (and optional biases)
    at call time. Weights must be shaped [out_features, in_features] in
    standard nn.Linear convention.

    Args:
        in_features: Input dimension (K = c_z in triangle multiply).
        out_features: Output dimension (N = 2 * c_z for proj/gate).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(
        self,
        x: torch.Tensor,
        w_gate: torch.Tensor,
        w_proj: torch.Tensor,
        b_gate: torch.Tensor | None = None,
        b_proj: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [*, in_features] input.
            w_gate: [out_features, in_features] gate weight (standard nn.Linear shape).
            w_proj: [out_features, in_features] projection weight.
            b_gate: [out_features] optional gate bias.
            b_proj: [out_features] optional projection bias.
            mask: optional per-row mask broadcast as mask[..., None] in epilogue.
        """
        if not HAS_TRITON or not x.is_cuda:
            gate = x @ w_gate.T
            proj = x @ w_proj.T
            if b_gate is not None:
                gate = gate + b_gate
            if b_proj is not None:
                proj = proj + b_proj
            out = torch.sigmoid(gate) * proj
            if mask is not None:
                out = out * mask.unsqueeze(-1)
            return out
        return fused_gated_dual_gemm(x, w_gate, w_proj, b_gate=b_gate, b_proj=b_proj, mask=mask)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}"
