"""Inference-only MSA transition: LayerNorm + SwiGLU + output projection + residual in one Triton kernel.

The stock block materialises ``[S*N, 2*hidden]`` and ``[S*N, hidden]`` tensors, which at
MSA scale are several times the size of the MSA itself. This kernel walks ``[BM, C]`` row
tiles and keeps the hidden activations in registers, ``BH`` hidden columns at a time, so no
hidden-sized tensor exists. With ``inplace=True`` the update is written over ``x`` and the
call allocates nothing proportional to ``x``; tiles are disjoint and each one reads its rows
before storing them, so ``OUT`` may alias ``X``.

Forward only: there is no backward, and calling the kernel with grad enabled raises
``RuntimeError`` rather than returning a tensor autograd cannot see through.

The tile is fixed (see ``_tile``), never autotuned: autotune launches every candidate on
live data, which would apply an in-place residual once per launch.

Numerics are the stock statement's, not its bits. LayerNorm statistics are fp32. Under bf16
the normalised input is rounded once to bf16 and ``silu(a) * b`` once to bf16; projections
accumulate in fp32 in this kernel's own order. In fp32, GEMMs use tf32x3 (IEEE under
``torch.set_float32_matmul_precision("highest")``): plain TF32 truncates its inputs and was
measured 1000x further from fp64 for a 3x saving this kernel does not need.

The kernel design follows the shared transition provider's fused rows (opt_core
``kernels/transition/af3/af3_fused.py``, AF3-PyTorch ``af3t_glu_proj.py``) and Boltz-2
``msa2/trans2.py``, Apache-2.0, Copyright 2026 Anthropic, PBC; this is a new implementation
adapted to DeepFold3.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from deepfold3.model import kernel_census
from deepfold3.model.inference_guard import inference_only
from deepfold3.model.triton import Precision

logger = logging.getLogger(__name__)

_SUPPORTED_CHANNELS = (64, 128)
_BH = 16  # hidden columns per GEMM; H must be a multiple

# (C, precision) -> launch tile, the fastest fit measured on RTX 6000 Ada (101 KB shared
# memory) at S=16000, N=566. Larger tiles exceed that limit under tf32x3 at C=128, and spill
# badly under IEEE, whose GEMMs lower to scalar FMA.
_TILES: dict[tuple[int, int], dict[str, int]] = {
    (64, Precision.DEFAULT): {"BM": 128, "num_warps": 8, "num_stages": 2},
    (64, Precision.TF32x3): {"BM": 64, "num_warps": 4, "num_stages": 1},
    (64, Precision.IEEE): {"BM": 64, "num_warps": 1, "num_stages": 1},
    (128, Precision.DEFAULT): {"BM": 128, "num_warps": 8, "num_stages": 2},
    (128, Precision.TF32x3): {"BM": 16, "num_warps": 2, "num_stages": 1},
    (128, Precision.IEEE): {"BM": 16, "num_warps": 1, "num_stages": 1},
}


def _tile(channels: int, precision: int) -> dict[str, int]:
    return _TILES[(channels, precision)]


_warned: set[str] = set()


if HAS_TRITON:
    from deepfold3.model.triton import precision_dot

    @triton.jit
    def _msa_transition_kernel(
        X,  # [M, C] bf16 | fp32
        OUT,  # [M, C] may alias X
        NORM_W,  # [C] fp32
        NORM_B,  # [C] fp32
        W_GATE,  # [2H, C] compute dtype; rows 0..H are silu(a), rows H..2H are b
        W_OUT,  # [C, H] compute dtype
        M,
        eps,
        scale,
        C: tl.constexpr,
        H: tl.constexpr,
        BM: tl.constexpr,
        BH: tl.constexpr,
        RESIDUAL: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
        """OUT = x + scale * update when RESIDUAL, else OUT = update."""
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        row_mask = (rows < M)[:, None]
        offs = rows.to(tl.int64)[:, None] * C + tl.arange(0, C)[None, :]
        cols = tl.arange(0, C)

        x = tl.load(X + offs, mask=row_mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / C
        centered = x - mean[:, None]
        var = tl.sum(centered * centered, axis=1) / C
        rstd = 1.0 / tl.sqrt(var + eps)
        norm_w = tl.load(NORM_W + cols)
        norm_b = tl.load(NORM_B + cols)
        y = (centered * rstd[:, None] * norm_w[None, :] + norm_b[None, :]).to(W_GATE.dtype.element_ty)

        acc = tl.zeros((BM, C), dtype=tl.float32)
        for h0 in range(0, H, BH):
            hid = h0 + tl.arange(0, BH)
            gate_a = tl.load(W_GATE + hid[None, :] * C + cols[:, None])  # [C, BH]
            gate_b = tl.load(W_GATE + (H + hid)[None, :] * C + cols[:, None])
            a = precision_dot(y, gate_a, tl.zeros((BM, BH), dtype=tl.float32), PRECISION)
            b = precision_dot(y, gate_b, tl.zeros((BM, BH), dtype=tl.float32), PRECISION)
            h = (a * tl.sigmoid(a) * b).to(W_OUT.dtype.element_ty)
            proj = tl.load(W_OUT + cols[None, :] * H + hid[:, None])  # [BH, C]
            acc = precision_dot(h, proj, acc, PRECISION)

        update = acc.to(W_OUT.dtype.element_ty)
        if RESIDUAL:
            value = x + (update.to(tl.float32) * scale).to(W_OUT.dtype.element_ty).to(tl.float32)
            tl.store(OUT + offs, value.to(OUT.dtype.element_ty), mask=row_mask)
        else:
            tl.store(OUT + offs, update.to(OUT.dtype.element_ty), mask=row_mask)


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    """The dtype the stock statement computes in: the autocast dtype if active, else x's."""
    if torch.is_autocast_enabled(x.device.type):
        return torch.get_autocast_dtype(x.device.type)
    return x.dtype


def _precision(dtype: torch.dtype) -> int:
    """GEMM precision for a compute dtype; the fp32 modes do not apply to bf16 operands."""
    if dtype != torch.float32:
        return Precision.DEFAULT
    return Precision.IEEE if torch.get_float32_matmul_precision() == "highest" else Precision.TF32x3


@inference_only("fused_msa_transition")
def fused_msa_transition(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    gate_weight: torch.Tensor,
    out_weight: torch.Tensor,
    *,
    eps: float = 1e-5,
    residual_scale: float | None = None,
    inplace: bool = False,
) -> torch.Tensor:
    """Run LN + SwiGLU + projection over ``x`` in one kernel.

    Args:
        x: ``[..., C]`` bf16 or fp32 CUDA tensor.
        norm_weight: ``[C]`` LayerNorm scale.
        norm_bias: ``[C]`` LayerNorm offset.
        gate_weight: ``[2H, C]`` linear1 weight.
        out_weight: ``[C, H]`` linear2 weight.
        eps: LayerNorm epsilon.
        residual_scale: If set, return ``x + residual_scale * update`` rather than the update.
        inplace: Write the result over ``x``. Requires ``residual_scale`` and contiguous ``x``.

    Returns:
        ``[..., C]`` tensor; ``x`` itself when ``inplace``.

    Raises:
        InferenceOnlyKernelError: If grad is enabled. This kernel has no backward.
        ValueError: If the inputs are outside what the kernel supports.
    """
    if not HAS_TRITON:
        raise ValueError("fused_msa_transition requires triton")
    if inplace and residual_scale is None:
        raise ValueError("inplace=True requires residual_scale")
    if inplace and not x.is_contiguous():
        raise ValueError("inplace=True requires a contiguous tensor")

    C = x.shape[-1]
    H = out_weight.shape[1]
    if C not in _SUPPORTED_CHANNELS:
        raise ValueError(f"unsupported channel width {C}; supported: {_SUPPORTED_CHANNELS}")
    if gate_weight.shape != (2 * H, C) or out_weight.shape != (C, H) or H % _BH:
        raise ValueError(f"bad weight shapes {tuple(gate_weight.shape)}, {tuple(out_weight.shape)} for C={C}")

    dtype = _compute_dtype(x)
    gate = gate_weight.detach().to(dtype).contiguous()
    proj = out_weight.detach().to(dtype).contiguous()
    norm_w = norm_weight.detach().float().contiguous()
    norm_b = norm_bias.detach().float().contiguous()

    flat = x.view(-1, C) if x.is_contiguous() else x.contiguous().view(-1, C)
    M = flat.shape[0]
    residual = residual_scale is not None
    if inplace:
        out = flat
    else:
        out = torch.empty((M, C), dtype=torch.promote_types(x.dtype, dtype) if residual else dtype, device=x.device)
    if M > 0:
        precision = _precision(dtype)
        tile = _tile(C, precision)
        _msa_transition_kernel[(triton.cdiv(M, tile["BM"]),)](
            flat,
            out,
            norm_w,
            norm_b,
            gate,
            proj,
            M,
            float(eps),
            float(residual_scale) if residual else 1.0,
            C=C,
            H=H,
            BM=tile["BM"],
            BH=_BH,
            RESIDUAL=residual,
            PRECISION=int(precision),
            num_warps=tile["num_warps"],
            num_stages=tile["num_stages"],
        )
    return x if inplace else out.view(x.shape)


def msa_transition_unsupported(block: nn.Module, x: torch.Tensor) -> str | None:
    """Return why the kernel cannot serve ``block(x)`` in place, or None if it can."""
    if not HAS_TRITON:
        return "triton is not installed"
    if not x.is_cuda:
        return f"input is on {x.device.type}"
    C = x.shape[-1]
    if C not in _SUPPORTED_CHANNELS:
        return f"channel width {C} is not supported (supported: {_SUPPORTED_CHANNELS})"
    norm = block.norm
    if norm.weight is None or norm.bias is None or getattr(norm, "output_scale", 1.0) != 1.0:
        return "LayerNorm is not a plain affine LayerNorm"
    H = block.linear2.weight.shape[1]
    if block.linear1.weight.shape != (2 * H, C) or H % _BH:
        return f"hidden width {H} does not fit the launch tile"
    dtype = _compute_dtype(x)
    if x.dtype not in (torch.bfloat16, torch.float32) or dtype not in (torch.bfloat16, torch.float32):
        return f"unsupported dtype (input {x.dtype}, compute {dtype})"
    if torch.promote_types(x.dtype, dtype) != x.dtype:
        return f"compute dtype {dtype} is wider than input dtype {x.dtype}"
    if not torch.is_autocast_enabled(x.device.type) and block.linear1.weight.dtype != x.dtype:
        return f"weight dtype {block.linear1.weight.dtype} differs from input dtype {x.dtype} without autocast"
    return None


@inference_only("fused_msa_transition")
def msa_transition_residual_(block: nn.Module, x: torch.Tensor, scale: float) -> torch.Tensor | None:
    """Apply ``x + scale * block(x)`` over ``x`` with the fused kernel.

    The caller must own ``x``: nothing else may hold a reference to its storage.

    Args:
        block: TransitionBlock-shaped module (``norm``, ``linear1``, ``linear2``).
        x: ``[..., C]`` MSA activations; overwritten.
        scale: Residual output scale.

    Returns:
        The updated tensor, or None when the kernel cannot serve this call. The reason is
        logged once per distinct reason, counted in the kernel census, and the caller takes
        the stock path. Errors from the kernel itself, out-of-memory included, propagate.

    Raises:
        InferenceOnlyKernelError: If grad is enabled. The flag must be off for training.
    """
    reason = msa_transition_unsupported(block, x)
    if reason is not None:
        kernel_census.record("fused_msa_transition", f"refused:{reason}")
        if reason not in _warned:
            _warned.add(reason)
            logger.warning("use_fused_msa_transition_inference is on but using the stock path: %s", reason)
        return None
    return fused_msa_transition(
        x.contiguous(),
        block.norm.weight,
        block.norm.bias,
        block.linear1.weight,
        block.linear2.weight,
        eps=block.norm.eps,
        residual_scale=scale,
        inplace=True,
    )
