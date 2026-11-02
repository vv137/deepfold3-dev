"""Fused AdaLN + Linear Triton kernel for DeepFold3.

Fuses the AdaLN (adaptive layer normalization) + Linear pattern used in the
diffusion transformer FFN:

    xn = F.layer_norm(x, [C])         # LN (no affine)
    xn = scale * xn + bias            # external conditioning
    out = linear(xn)                  # GEMM (no bias)

Eliminates materializing the intermediate `xn` tensor.

The forward pass reuses `_first_forward_kernel` from `layernorm_linear.py`
for mean/rstd computation, then applies a new `_ada_second_forward_kernel`
that folds in per-element scale/bias from conditioning.

Falls back to `F.linear(scale * F.layer_norm(x, [K]) + bias, W.T)` when
Triton configs are unavailable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


from deepfold3.model.triton import calculate_settings as _calculate_settings

if HAS_TRITON:
    from deepfold3.model.triton.cache import persistent_autotune
    from deepfold3.model.triton.layernorm_linear import (
        _GEMM_CONFIGS,
        _first_forward_kernel,
        _get_dtype_flag,
        _second_backward_kernel,
    )

    @persistent_autotune(configs=_GEMM_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _ada_second_forward_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        scale_ptr,
        bias_ptr,
        M,
        N,
        K,
        Mean_ptr,
        RSTD_ptr,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_sm,
        stride_sk,
        Mean_row_stride,
        RSTD_row_stride,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        DTYPE: tl.constexpr,
    ):
        """Fused AdaLN-normalized GEMM: c = (scale * ((x - mean) * rstd) + bias) @ b."""
        pid = tl.program_id(axis=0).to(tl.int64)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        scale_ptrs = scale_ptr + (offs_m[:, None] * stride_sm + offs_k[None, :] * stride_sk)
        bias_ptrs = bias_ptr + (offs_m[:, None] * stride_sm + offs_k[None, :] * stride_sk)

        mean = tl.load(Mean_ptr + offs_m * Mean_row_stride, mask=offs_m < M, other=0.0)
        rstd = tl.load(RSTD_ptr + offs_m * RSTD_row_stride, mask=offs_m < M, other=0.0)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_mask = offs_k < K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_mask[None, :]), other=0.0)
            b = tl.load(b_ptrs, mask=(k_mask[:, None]) & (offs_n[None, :] < N), other=0.0)
            s = tl.load(scale_ptrs, mask=(offs_m[:, None] < M) & (k_mask[None, :]), other=0.0)
            bi = tl.load(bias_ptrs, mask=(offs_m[:, None] < M) & (k_mask[None, :]), other=0.0)

            # AdaLN: scale * ((x - mean) * rstd) + bias
            a = s * ((a - mean[:, None]) * rstd[:, None]) + bi

            accumulator = tl.dot(a, b, accumulator)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            scale_ptrs += BLOCK_SIZE_K * stride_sk
            bias_ptrs += BLOCK_SIZE_K * stride_sk

        c = accumulator.to(DTYPE)
        c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    @persistent_autotune(configs=_GEMM_CONFIGS, key=["M", "K", "N"], reset_to_zero=["dB", "c1", "c2"])
    @triton.jit
    def _ada_first_backward_kernel(
        dOUT_ptr,
        B,
        dX,
        dB,
        X,
        Mean,
        RSTD,
        SCALE,
        BIAS,
        dSCALE,
        dBIAS_ada,
        c1,
        c2,
        M,
        K,
        N,
        stride_am,
        stride_an,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_ck,
        stride_sm,
        stride_sk,
        stride_c1,
        stride_c2,
        stride_mean,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """Backward pass 1: dY = dOUT @ B^T, accumulate dB, dSCALE, dBIAS, c1, c2.

        Unlike the standard LN-Linear backward, scale/bias are per-element [M, K]
        (from conditioning), not shared [K] (from LN affine params).
        """
        pid = tl.program_id(axis=0).to(tl.int64)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_in_group = GROUP_SIZE_M * num_pid_k
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_k = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        offs_n = tl.arange(0, BLOCK_SIZE_N)
        dOUT_ptrs = dOUT_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
        B_T_ptrs = B + (offs_k[None, :] * stride_bk + offs_n[:, None] * stride_bn)

        x = tl.load(
            X + (offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        mean = tl.load(Mean + (offs_m * stride_mean), mask=offs_m < M, other=0.0)
        rstd = tl.load(RSTD + (offs_m * stride_mean), mask=offs_m < M, other=0.0)
        x_hat = (x - mean[:, None]) * rstd[:, None]

        scale = tl.load(
            SCALE + (offs_m[:, None] * stride_sm + offs_k[None, :] * stride_sk),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        bias = tl.load(
            BIAS + (offs_m[:, None] * stride_sm + offs_k[None, :] * stride_sk),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        # Transpose of the adaln-transformed x for dB accumulation
        y_T = tl.trans((x_hat * scale) + bias)

        dy = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

        for _n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
            dOUT = tl.load(dOUT_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)

            db = tl.dot(y_T, dOUT)
            tl.atomic_add(
                dB + offs_k[:, None] * N + offs_n[None, :],
                db.to(dB.type.element_ty),
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                sem="relaxed",
            )

            b_T = tl.load(B_T_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
            dy = tl.dot(dOUT, b_T, dy)

            dOUT_ptrs += BLOCK_SIZE_N * stride_an
            B_T_ptrs += BLOCK_SIZE_N * stride_bn
            offs_n += BLOCK_SIZE_N

        # dy is d_loss/d_adaln_output = d_loss/d(scale * x_hat + bias)
        # d_scale = dy * x_hat (per-element)
        # d_bias_ada = dy (per-element)
        ds_vals = dy * x_hat
        db_vals = dy

        tl.store(
            dSCALE + (offs_m[:, None] * stride_sm + offs_k[None, :] * stride_sk),
            ds_vals.to(dSCALE.type.element_ty),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        )
        tl.store(
            dBIAS_ada + (offs_m[:, None] * stride_sm + offs_k[None, :] * stride_sk),
            db_vals.to(dBIAS_ada.type.element_ty),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        )

        # wdy = dy * scale for LN backward
        wdy = dy * scale

        c1_vals = tl.sum(x_hat * wdy, axis=1) / K
        c2_vals = tl.sum(wdy, axis=1) / K

        tl.atomic_add(c1 + offs_m * stride_c1, c1_vals.to(c1.type.element_ty), mask=offs_m < M, sem="relaxed")
        tl.atomic_add(c2 + offs_m * stride_c2, c2_vals.to(c2.type.element_ty), mask=offs_m < M, sem="relaxed")

        tl.store(
            dX + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck,
            wdy.to(dX.type.element_ty),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        )


def ada_layernorm_linear_forward(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused AdaLN + Linear forward.

    Block sizes are picked by ``persistent_autotune`` based on (M, N, K).
    """
    shape = x.shape
    K = shape[-1]
    x = x.reshape(-1, K).contiguous()
    scale = scale.reshape(-1, K).contiguous()
    bias = bias.reshape(-1, K).contiguous()
    M = x.shape[0]
    N = linear_weight.shape[1]

    c = torch.empty((M, N), dtype=x.dtype, device=x.device)
    Mean = torch.empty((M,), dtype=x.dtype, device=x.device)
    RSTD = torch.empty((M,), dtype=x.dtype, device=x.device)

    # Pass 1: compute per-row mean and rstd
    BLOCK_SIZE, num_warps = _calculate_settings(K)
    _first_forward_kernel[(M,)](
        x,
        x.stride(0),
        Mean,
        Mean.stride(0),
        RSTD,
        RSTD.stride(0),
        K,
        1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    # Pass 2: fused AdaLN + GEMM
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    _ada_second_forward_kernel[grid](
        x,
        linear_weight,
        c,
        scale,
        bias,
        M,
        N,
        K,
        Mean,
        RSTD,
        x.stride(0),
        x.stride(1),
        linear_weight.stride(0),
        linear_weight.stride(1),
        c.stride(0),
        c.stride(1),
        scale.stride(0),
        scale.stride(1),
        Mean.stride(0),
        RSTD.stride(0),
        DTYPE=_get_dtype_flag(x.dtype),
    )

    c = c.view(shape[:-1] + (N,))
    return c, x, linear_weight, scale, bias, Mean, RSTD


def ada_layernorm_linear_backward(
    dOUT: torch.Tensor,
    X: torch.Tensor,
    B: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    Mean: torch.Tensor,
    RSTD: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused AdaLN + Linear backward.

    Returns:
        (dX, dB, dScale, dBias)
    """
    dOUT_shape = dOUT.shape
    dOUT = dOUT.view(-1, dOUT_shape[-1]).contiguous()
    M, K = X.shape
    N = dOUT.shape[-1]

    dX = torch.empty((M, K), dtype=X.dtype, device=X.device)
    dB = torch.zeros((K, N), dtype=torch.float32, device=X.device)

    dScale = torch.empty((M, K), dtype=X.dtype, device=X.device)
    dBias_ada = torch.empty((M, K), dtype=X.dtype, device=X.device)

    c1 = torch.zeros((M,), dtype=torch.float32, device=X.device)
    c2 = torch.zeros((M,), dtype=torch.float32, device=X.device)

    _ada_first_backward_kernel[
        lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]),)
    ](
        dOUT,
        B,
        dX,
        dB,
        X,
        Mean,
        RSTD,
        scale,
        bias,
        dScale,
        dBias_ada,
        c1,
        c2,
        M,
        K,
        N,
        dOUT.stride(0),
        dOUT.stride(1),
        B.stride(0),
        B.stride(1),
        dX.stride(0),
        dX.stride(1),
        scale.stride(0),
        scale.stride(1),
        c1.stride(0),
        c2.stride(0),
        Mean.stride(0),
    )

    # Pass 2: finish dX using second backward kernel
    BLOCK_SIZE_2, num_warps_2 = _calculate_settings(K)
    _second_backward_kernel[(M,)](
        dX,
        X,
        c1,
        c2,
        Mean,
        RSTD,
        M,
        K,
        N,
        X.stride(0),
        Mean.stride(0),
        BLOCK_SIZE=BLOCK_SIZE_2,
        num_warps=num_warps_2,
    )

    dX = dX.view(dOUT_shape[:-1] + (K,))
    dB = dB.to(X.dtype)
    dScale = dScale.view(dOUT_shape[:-1] + (K,))
    dBias_ada = dBias_ada.view(dOUT_shape[:-1] + (K,))

    return dX, dB, dScale, dBias_ada


class AdaLayerNormLinearFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(
        ctx,
        X,
        linear_weight,
        scale,
        bias,
    ):
        X = X.contiguous()
        linear_weight = linear_weight.contiguous()
        scale = scale.contiguous()
        bias = bias.contiguous()
        orig_shape = X.shape
        c, X_flat, linear_weight, scale_flat, bias_flat, Mean, RSTD = ada_layernorm_linear_forward(
            X, linear_weight, scale, bias
        )
        # Save tensors in their original shape so non-reentrant gradient
        # checkpointing sees matching metadata between original and recomputed
        # forward. The flat buffers are contiguous; .view() shares storage and
        # we re-flatten in backward for free.
        ctx.save_for_backward(
            X_flat.view(orig_shape),
            linear_weight,
            scale_flat.view(orig_shape),
            bias_flat.view(orig_shape),
            Mean,
            RSTD,
        )
        return c

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dOUT):
        X_orig, linear_weight, scale_orig, bias_orig, Mean, RSTD = ctx.saved_tensors
        K = X_orig.shape[-1]
        X_flat = X_orig.reshape(-1, K).contiguous()
        scale_flat = scale_orig.reshape(-1, K).contiguous()
        bias_flat = bias_orig.reshape(-1, K).contiguous()
        dOUT = dOUT.contiguous()
        dX, dB, dScale, dBias = ada_layernorm_linear_backward(
            dOUT, X_flat, linear_weight, scale_flat, bias_flat, Mean, RSTD
        )
        return dX, dB, dScale, dBias


class FusedAdaLayerNormLinear(nn.Module):
    """Fused AdaLN + Linear kernel — STATELESS.

    Replaces the pattern:
        xn = F.layer_norm(x, [K])
        xn = scale * xn + bias
        out = F.linear(xn, W.T)

    where scale/bias come from external conditioning (per-element, not learnable).

    Holds no parameters; callers pass the linear weight (stored as [K, N] for
    the Triton kernel) at call time.

    Args:
        in_features: Input/normalized dimension (K).
        out_features: Output dimension (N).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
        linear_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., K].
            scale: Conditioning scale [..., K].
            bias: Conditioning bias [..., K].
            linear_weight: [in_features, out_features] transposed linear
                weight (for standard nn.Linear weight [out, in], pass `.T`).

        Returns:
            Output tensor [..., N].
        """
        if not HAS_TRITON or not x.is_cuda:
            xn = F.layer_norm(x, (self.in_features,))
            xn = scale * xn + bias
            return F.linear(xn, linear_weight.T)
        return AdaLayerNormLinearFunction.apply(x, linear_weight, scale, bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}"
