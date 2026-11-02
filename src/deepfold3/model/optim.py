"""Optimized kernel backends for DeepFold3 model modules.

Provides factory functions that return either a baseline PyTorch module or an
optimized Triton replacement based on OptimConfig flags. All replacements
are API-compatible drop-ins.
"""

from __future__ import annotations

import importlib
from functools import lru_cache

import torch.nn as nn

from deepfold3.model import kernel_census
from deepfold3.model.config import OptimConfig

# Dims where the fused transition is used. The gate is by channel count, not by
# role, so it serves the C=128 MSA and template blocks and misses C=256 pair blocks.
_PAIR_TRANSITION_CHANNELS = (64, 128)
# Single/MSA channel dims where fused transition is slow (M=N, small → PyTorch wins).
_SINGLE_TRANSITION_CHANNELS = (384,)


@lru_cache(maxsize=16)
def _try_import(module_path: str, attr: str):
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None


def is_available(backend: str) -> bool:
    """Check if an optimization backend is importable."""
    checks = {
        "triton_transition": ("deepfold3.model.triton.fused_transition", "FusedTransitionBlock"),
        "triton_evo_attention": ("deepfold3.model.triton.evo_attention", "fused_evo_attention"),
        "triton_layernorm_linear": ("deepfold3.model.triton.layernorm_linear", "FusedLayerNormLinear"),
        "triton_ada_transition": ("deepfold3.model.triton.fused_ada_transition", "FusedAdaTransitionBlock"),
        "triton_triangle_mul_gate": ("deepfold3.model.triton.triangle_mul_gate", "FusedTriangleMulGate"),
        "torch_compile": ("torch", "compile"),
    }
    if backend not in checks:
        return False
    mod_path, attr = checks[backend]
    return _try_import(mod_path, attr) is not None


def make_transition_block(
    channels: int,
    num_intermediate_factor: int = 4,
    config: OptimConfig | None = None,
    linear_out_init_gain: float | None = None,
) -> nn.Module:
    """Create a TransitionBlock, optionally using fused Triton kernels.

    Uses separate flags for pair vs single transitions because they operate
    at vastly different M sizes:
    - Pair transitions: M = N² (up to 590k) — Triton wins at 1.2-1.3x
    - Single transitions: M = N (384-768) — PyTorch wins (Triton overhead dominates)

    ``linear_out_init_gain`` overrides the AF3 zeros init on the SwiGLU output
    projection (``linear2`` / ``linear_out``) with a scaled row-orthogonal
    init. Fused transition kernels carry the projection internally and have
    no hook for this; combining them is rejected with ``ValueError``.
    """
    block = None
    use_fused = False
    if config is not None:
        if channels not in _PAIR_TRANSITION_CHANNELS + _SINGLE_TRANSITION_CHANNELS:
            for lever in ("use_fused_transition_pair", "use_tiled_linear"):
                if getattr(config, lever):
                    kernel_census.inactive(f"{lever}[C={channels}]", f"gated to C in {_PAIR_TRANSITION_CHANNELS}")
        if config.use_fused_transition_pair and channels in _PAIR_TRANSITION_CHANNELS:
            use_fused = True
        elif config.use_fused_transition_single and channels in _SINGLE_TRANSITION_CHANNELS:
            use_fused = True
        if use_fused:
            if linear_out_init_gain is not None:
                raise ValueError(
                    "linear_out_init_gain requires use_fused_transition_pair=False and "
                    "use_fused_transition_single=False; the fused kernel carries the output "
                    "projection internally and would bypass the override."
                )
            cls = _try_import("deepfold3.model.triton.fused_transition", "FusedTransitionBlock")
            if cls is not None:
                block = cls(channels, num_intermediate_factor=num_intermediate_factor)
    if block is None:
        from deepfold3.model.primitives import TransitionBlock

        block = TransitionBlock(
            channels,
            num_intermediate_factor=num_intermediate_factor,
            optim=config,
            linear_out_init_gain=linear_out_init_gain,
        )
    # Tiled recomputation: only wrap non-fused blocks. FusedTransitionBlock
    # contains a fused LayerNorm+Linear — tiling would compute LN statistics
    # per chunk instead of globally, corrupting the normalization.
    if config is not None and config.use_tiled_linear and channels in _PAIR_TRANSITION_CHANNELS and not use_fused:
        from deepfold3.model.tiled_linear import TiledLinear

        block = TiledLinear(block, num_shards=config.tiled_linear_shards, tile_inference=config.tiled_linear_inference)
    return block


def make_ada_transition_block(
    channels: int, cond_dim: int, num_intermediate_factor: int = 2, config: OptimConfig | None = None
) -> nn.Module:
    """Create an AdaTransitionBlock, optionally using fused Triton kernels."""
    if config and config.use_fused_ada_transition:
        cls = _try_import("deepfold3.model.triton.fused_ada_transition", "FusedAdaTransitionBlock")
        if cls is not None:
            return cls(channels, cond_dim, num_intermediate_factor=num_intermediate_factor)
    from deepfold3.model.primitives import AdaTransitionBlock

    return AdaTransitionBlock(channels, cond_dim, num_intermediate_factor=num_intermediate_factor, optim=config)


def print_availability():
    """Print which optimization backends are available."""
    backends = [
        ("triton_transition", "Triton FusedTransition"),
        ("triton_evo_attention", "Triton FusedEvoAttention"),
        ("triton_layernorm_linear", "Triton FusedLayerNormLinear"),
        ("triton_ada_transition", "Triton FusedAdaTransition"),
        ("triton_triangle_mul_gate", "Triton FusedTriangleMulGate"),
        ("torch_compile", "torch.compile"),
    ]
    print("Optimization backends:")
    for key, name in backends:
        status = "available" if is_available(key) else "not found"
        print(f"  {name}: {status}")
