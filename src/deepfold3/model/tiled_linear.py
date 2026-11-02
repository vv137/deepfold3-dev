"""Tiled linear computation for memory-efficient pair transitions.

Chunks input along a spatial dimension and processes each chunk independently.
During training, each chunk's forward is wrapped in
``torch.utils.checkpoint.checkpoint(use_reentrant=False)`` so the chunk's
activations are released between chunks and recomputed during backward.
During inference, chunking reduces peak memory for very long sequences.

All operations in TransitionBlock (LayerNorm, Linear, SwiGLU) act on the last
dimension independently per spatial position, so chunking along dim=-2 produces
bitwise identical results to the full forward pass.

Implementation note: an earlier version used a custom ``torch.autograd.Function``
that ran each chunk under ``torch.no_grad()`` in forward and ``torch.enable_grad()``
in backward. That pattern is incompatible with an *outer* non-reentrant
gradient checkpoint (the saved-tensor hooks see different module-level state
between the original forward and the recompute, producing
``CheckpointError: Recomputed values have different metadata``). The current
implementation uses ``torch.utils.checkpoint.checkpoint`` per chunk, which
composes correctly with outer non-reentrant checkpointing.
"""

from __future__ import annotations

import math
from functools import partial as _partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from deepfold3.model import kernel_census

torch_checkpoint = _partial(_torch_checkpoint, use_reentrant=False, determinism_check="none")


def _auto_shards(x: torch.Tensor) -> int:
    """Compute number of shards from input shape.

    Heuristic: ceil(spatial_dim / channel_dim). For pair [N, N, 128] with
    N=384 this gives ceil(384/128) = 3 shards.
    """
    return max(1, math.ceil(x.shape[-2] / x.shape[-1]))


def _tiled_forward(x: torch.Tensor, module: nn.Module, num_shards: int) -> torch.Tensor:
    """Tiled forward of ``module(chunk)`` along ``dim=-2``."""
    chunks = x.chunk(num_shards, dim=-2)
    if torch.is_grad_enabled() and x.requires_grad:
        outputs = [torch_checkpoint(module, chunk) for chunk in chunks]
    else:
        outputs = [module(chunk) for chunk in chunks]
    return torch.cat(outputs, dim=-2)


def _tiled_ada_forward(x: torch.Tensor, cond: torch.Tensor, module: nn.Module, num_shards: int) -> torch.Tensor:
    """Tiled forward of ``module(chunk_x, chunk_cond)`` along ``dim=-2``."""
    chunks_x = x.chunk(num_shards, dim=-2)
    chunks_c = cond.chunk(num_shards, dim=-2)
    if torch.is_grad_enabled() and (x.requires_grad or cond.requires_grad):
        outputs = [torch_checkpoint(module, xc, cc) for xc, cc in zip(chunks_x, chunks_c, strict=True)]
    else:
        outputs = [module(xc, cc) for xc, cc in zip(chunks_x, chunks_c, strict=True)]
    return torch.cat(outputs, dim=-2)


def tiled_opm_contract(a: torch.Tensor, b: torch.Tensor, contract_fn, num_shards: int) -> torch.Tensor:
    """2D-tiled outer product contraction.

    Chunks both ``a`` (dim=2, rows) and ``b`` (dim=2, cols) so the intermediate
    is ``[B, chunk_i, chunk_j, c_h, c_h]`` — constant in N. Each (i, j) chunk's
    contract is wrapped in a non-reentrant checkpoint when grad is enabled.
    """
    chunks_a = a.chunk(num_shards, dim=2)
    chunks_b = b.chunk(num_shards, dim=2)
    use_ckpt = torch.is_grad_enabled() and (a.requires_grad or b.requires_grad)

    row_outputs = []
    for a_chunk in chunks_a:
        col_outputs = []
        for b_chunk in chunks_b:
            if use_ckpt:
                out = torch_checkpoint(contract_fn, a_chunk, b_chunk)
            else:
                out = contract_fn(a_chunk, b_chunk)
            col_outputs.append(out)
        row_outputs.append(torch.cat(col_outputs, dim=2))  # cat along j
    return torch.cat(row_outputs, dim=1)  # cat along i


class _TransparentWrapper(nn.Module):
    """Base class for wrappers with transparent state dict keys.

    Re-registers inner module's parameters, buffers, and submodules directly
    on the wrapper so that state_dict keys match the unwrapped module exactly.
    The inner module and wrapper share the same parameter objects.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        # Re-register inner's direct attributes on self for transparent state dict
        for name, param in inner._parameters.items():
            self.register_parameter(name, param)
        for name, buf in inner._buffers.items():
            self.register_buffer(name, buf)
        for name, mod in inner._modules.items():
            self.add_module(name, mod)
        # Keep a reference to inner for calling its forward method
        object.__setattr__(self, "_inner_ref", inner)

    @property
    def inner(self) -> nn.Module:
        return self._inner_ref


class TiledLinear(_TransparentWrapper):
    """Transparent wrapper that tiles computation along dim=-2.

    During training (requires_grad): uses custom autograd function that
    recomputes each chunk's forward during backward, avoiding storage of
    intermediate activations.

    During inference: optionally chunks forward pass to reduce peak memory
    for very long sequences (controlled by tile_inference flag).

    State dict is transparent — keys match the inner module exactly (no
    '_inner.' prefix) for full checkpoint compatibility.

    Args:
        inner: The module to wrap (TransitionBlock, FusedTransitionBlock, etc.)
        num_shards: Number of chunks along dim=-2. 0 = auto heuristic.
        tile_inference: Also tile during inference (for memory-constrained cases).
    """

    def __init__(self, inner: nn.Module, num_shards: int = 0, tile_inference: bool = False):
        super().__init__(inner)
        self._num_shards = num_shards
        self._tile_inference = tile_inference

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shards = self._num_shards or _auto_shards(x)
        if shards <= 1:
            return self.inner(x)

        if self.training and x.requires_grad:
            kernel_census.record("TiledLinear", "tiled_grad")
            return _tiled_forward(x, self.inner, shards)

        if self._tile_inference or self.training:
            kernel_census.record("TiledLinear", "tiled")
            chunks = x.chunk(shards, dim=-2)
            return torch.cat([self.inner(chunk) for chunk in chunks], dim=-2)

        return self.inner(x)

    def __repr__(self) -> str:
        return f"TiledLinear(shards={self._num_shards or 'auto'}, tile_inference={self._tile_inference}, inner={self.inner})"


class TiledAdaLinear(_TransparentWrapper):
    """Transparent wrapper for modules taking (x, cond) — e.g. AdaTransitionBlock.

    Same as TiledLinear but handles the conditioning input. Both x and cond
    are chunked along dim=-2.
    """

    def __init__(self, inner: nn.Module, num_shards: int = 0, tile_inference: bool = False):
        super().__init__(inner)
        self._num_shards = num_shards
        self._tile_inference = tile_inference

    @torch.compiler.disable
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shards = self._num_shards or _auto_shards(x)
        if shards <= 1:
            return self.inner(x, cond)

        if self.training and x.requires_grad:
            return _tiled_ada_forward(x, cond, self.inner, shards)

        if self._tile_inference or self.training:
            chunks_x = x.chunk(shards, dim=-2)
            chunks_c = cond.chunk(shards, dim=-2)
            return torch.cat([self.inner(xc, cc) for xc, cc in zip(chunks_x, chunks_c, strict=True)], dim=-2)

        return self.inner(x, cond)

    def __repr__(self) -> str:
        return f"TiledAdaLinear(shards={self._num_shards or 'auto'}, tile_inference={self._tile_inference}, inner={self.inner})"
