"""Diffusion head for DeepFold3.

Implements the denoising diffusion model for atom position generation.

Haiku scope: diffuser~/diffusion_head~/

Key submodules:
    conditioning~/  → see _conditioning() in AF3
    transformer~/   → DiffusionTransformer (24 layers, AdaLN-zero)
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from contextlib import nullcontext
from functools import partial as _partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from deepfold3.model.atom_cross_attn import (
    AtomDecoder,
    AtomEncoder,
    AtomEncoderStatic,
    broadcast_for_samples,
    expand_for_samples_shared,
)
from deepfold3.model.config import DiffusionConfig, OptimConfig
from deepfold3.model.optim import make_transition_block
from deepfold3.model.precision import upcast_fp32 as _upcast_fp32
from deepfold3.model.primitives import (
    TRUNCATED_NORMAL_STDDEV_FACTOR,
    LayerNorm,
    LinearNoBias,
    af3_linear_init_,
    layer_norm,
)
from deepfold3.model.restraints import Restraint, active_restraints, restraint_gradient, rigid_displacement

torch_checkpoint = _partial(_torch_checkpoint, use_reentrant=False, determinism_check="none")

# Fixed seed for the non-trainable Fourier noise-embedding basis (w ~ N(0,1),
# b ~ U[0,1)). NOT needed for DDP rank agreement — DDP broadcasts buffers from
# rank 0 at construction and the basis never changes — but it keeps the basis
# reproducible across separate runs (so A/B arms share one fixed basis instead of
# a per-launch random confound) and deterministic on single-GPU / non-DDP. The
# buffers are persisted in checkpoints and overwritten by AF3-converted weights on
# load. See DiffusionConditioningBlock.__init__.
_FOURIER_INIT_SEED = 314159


@dataclasses.dataclass
class DiffusionStatic:
    """All noise-independent state for one (molecule, trunk) pair.

    Lets the diffusion sampling/training loops compute conditioning + atom
    encoder cache once and reuse them across the (potentially many) diffusion
    samples and timesteps that share the same molecule.
    """

    encoder_static: AtomEncoderStatic
    pair_cond: torch.Tensor  # [B?, N, N, c_pair_cond]
    single_cond_pre_noise: torch.Tensor  # [B?, N, c_single_cond]
    ref_pos: torch.Tensor  # [B?, N, M, 3] — for input scaling
    ref_mask: torch.Tensor  # [B?, N, M] — for input/output masking
    seq_mask: torch.Tensor  # [B?, N]
    # Set by ``DiffusionHead.hoist_static``: per super-block token pair logits
    # with the padding mask folded in. ``single_cond_pre_noise`` then stays [B, ...].
    pair_logits: tuple[torch.Tensor, ...] | None = None

    def expand_for_samples(self, num_samples: int) -> DiffusionStatic:
        if num_samples == 1:
            return self

        def exp(t: torch.Tensor) -> torch.Tensor:
            return t.repeat_interleave(num_samples, dim=0).contiguous()

        return DiffusionStatic(
            encoder_static=self.encoder_static.expand_for_samples(num_samples),
            pair_cond=exp(self.pair_cond),
            single_cond_pre_noise=exp(self.single_cond_pre_noise),
            ref_pos=exp(self.ref_pos),
            ref_mask=exp(self.ref_mask),
            seq_mask=exp(self.seq_mask),
        )


def noise_schedule(
    t: torch.Tensor,
    sigma_data: float = 16.0,
    smin: float = 0.0004,
    smax: float = 160.0,
    p: float = 7.0,
) -> torch.Tensor:
    """AF3 noise schedule: sigma = sigma_data * (smax^(1/p) + t*(smin^(1/p)-smax^(1/p)))^p"""
    return sigma_data * (smax ** (1 / p) + t * (smin ** (1 / p) - smax ** (1 / p))) ** p


def _random_augmentation_from_vectors(
    positions: torch.Tensor,  # [N, M, 3]
    mask: torch.Tensor,  # [N, M] bool or float
    v0: torch.Tensor,  # [3]
    v1: torch.Tensor,  # [3]
    translation: torch.Tensor,  # [3]
) -> torch.Tensor:
    """Apply random rigid augmentation using pre-sampled random vectors.

    Matches AF3 diffusion_head.random_augmentation() exactly. Gram-Schmidt
    and rotation apply use fp32 (AF3 `jax.lax.Precision.HIGHEST` in
    diffusion_head.py:38,72) — orthogonality preservation is numerically
    critical and breaks under bf16.
    """
    out_dtype = positions.dtype
    # Upcast for rotation / centering math (matches AF3 HIGHEST).
    positions_f = positions.float()
    mask_f = mask.float().unsqueeze(-1)  # [N, M, 1]
    total = mask_f.sum().clamp(min=1e-6)
    center = (positions_f * mask_f).sum(dim=(0, 1), keepdim=True) / total
    pos = positions_f - center

    # Random rotation via Gram-Schmidt (fp32).
    v0_f = v0.float()
    v1_f = v1.float()
    e0 = v0_f / v0_f.norm().clamp(min=1e-10)
    v1_orth = v1_f - e0 * torch.dot(v1_f, e0)
    e1 = v1_orth / v1_orth.norm().clamp(min=1e-10)
    e2 = torch.linalg.cross(e0, e1)
    rot = torch.stack([e0, e1, e2], dim=0)

    pos = pos @ rot + translation.float()
    return (pos * mask_f).to(out_dtype)


def _random_augmentation(
    positions: torch.Tensor,
    mask: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Apply random rigid augmentation (backward-compatible single-sample interface)."""
    device = positions.device
    v0 = torch.randn(3, device=device, generator=generator)
    v1 = torch.randn(3, device=device, generator=generator)
    translation = torch.randn(3, device=device, generator=generator)
    return _random_augmentation_from_vectors(positions, mask, v0, v1, translation)


def build_diffusion_clean_mask(
    asym_id: torch.Tensor,  # [B, N] int
    seq_mask: torch.Tensor,  # [B, N] bool
    num_samples: int,
    *,
    prob: float,
    mode: str = "chain",
    min_frac: float = 0.25,
    max_frac: float = 0.75,
) -> torch.Tensor | None:
    """``[B, S, N]`` bool — tokens handed to the denoiser as clean ground truth.

    Masked / partial diffusion (motif scaffolding). Each of the ``S`` diffusion
    samples independently fires with probability ``prob``; a sample that does
    not fire gets an all-False row and is noised normally, so ``prob`` is the
    fraction of the diffusion batch that becomes a conditional-placement task.

    ``mode="chain"`` keeps a random non-empty PROPER subset of chains clean —
    always ≥1 clean and ≥1 noised, so there is both something to place and
    something to place it against. A single-chain crop has no such subset and
    falls back to ``"span"``.

    Returns ``None`` when nothing fired, so callers can skip the work entirely.
    """
    if prob <= 0.0 or num_samples <= 0:
        return None
    B, N = asym_id.shape
    dev = asym_id.device
    fire = torch.rand(B, num_samples, device=dev) < prob
    if not bool(fire.any()):
        return None

    mask = torch.zeros(B, num_samples, N, dtype=torch.bool, device=dev)
    lo, hi = float(min_frac), float(max_frac)
    for b in range(B):
        valid_b = seq_mask[b].bool()
        n_valid = int(valid_b.sum())
        if n_valid < 4:
            continue
        chains = asym_id[b][valid_b].unique()
        valid_idx = valid_b.nonzero(as_tuple=True)[0]
        for s in range(num_samples):
            if not bool(fire[b, s]):
                continue
            frac = lo + float(torch.rand((), device=dev)) * (hi - lo)
            if mode == "chain" and chains.numel() >= 2:
                perm = chains[torch.randperm(chains.numel(), device=dev)]
                k = max(1, min(int(chains.numel()) - 1, int(round(frac * float(chains.numel())))))
                keep = perm[:k]
                mask[b, s] = (asym_id[b].unsqueeze(-1) == keep).any(-1) & valid_b
            else:  # "span", and the single-chain fallback for "chain"
                ln = max(1, min(n_valid - 1, int(round(frac * n_valid))))
                start = int(torch.randint(0, n_valid - ln + 1, (), device=dev))
                mask[b, s, valid_idx[start : start + ln]] = True
    return mask


@dataclasses.dataclass(frozen=True)
class GuidanceTerm:
    """One term of the classifier-free guidance sum in :meth:`DiffusionHead.sample`.

    The guided denoiser is ``D_c + Σ_k (w_k − 1)·(D_c − D_u,k)``, so one term is
    the usual ``D_u + w·(D_c − D_u)``, ``w = 1`` adds nothing, and each term
    costs one more denoiser pass per step inside its σ̂ range.

    Attributes:
        scale: The term's guidance weight ``w_k``.
        pair_keep: ``[B?, N, N]`` bool, False where this term's unconditional
            branch zeroes the trunk pair; ``None`` is the ``cond_drop`` drop.
        sigma_range: ``(lo, hi)``: apply where lo ≤ σ̂ ≤ hi, ``None`` everywhere.
    """

    scale: float
    pair_keep: torch.Tensor | None = None
    sigma_range: tuple[float, float] | None = None


def interface_pair_keep(asym_id: torch.Tensor) -> torch.Tensor:
    """``[B?, N, N]`` bool: True on intra-chain pairs, False on every inter-chain pair."""
    return asym_id.unsqueeze(-1) == asym_id.unsqueeze(-2)


def draw_interface_pair_keep(
    asym_id: torch.Tensor,  # [B, N]
    seq_mask: torch.Tensor,  # [B, N]
    generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    """Random inter-chain pair-block drop for training (``cond_drop_interface_frac``).

    Per batch element: r ~ U(0, 1), each unordered pair of chains present in
    ``seq_mask`` dropped with probability r (both blocks), at least one pair
    always dropped. Returns ``[B, N, N]`` bool keep (True = conditioned), with
    intra-chain and padding entries always True, or ``None`` when any element
    has fewer than two chains — there is no interface to drop.
    """
    B, N = asym_id.shape
    keep = torch.ones(B, N, N, dtype=torch.bool, device=asym_id.device)
    for b in range(B):
        valid = seq_mask[b].bool()
        chains = asym_id[b][valid].unique()  # sorted
        C = chains.numel()
        if C < 2:
            return None
        r = torch.rand((), generator=generator, device=asym_id.device)
        drop = torch.rand(C, C, generator=generator, device=asym_id.device) < r
        drop = torch.triu(drop, diagonal=1)
        if not bool(drop.any()):
            iu = torch.triu_indices(C, C, offset=1, device=asym_id.device)
            k = int(torch.randint(iu.shape[1], (), generator=generator, device=asym_id.device))
            drop[iu[0, k], iu[1, k]] = True
        drop = drop | drop.T
        idx = torch.searchsorted(chains, asym_id[b]).clamp_(max=C - 1)
        keep_b = ~drop[idx.unsqueeze(-1), idx.unsqueeze(-2)]
        keep[b] = keep_b | ~(valid.unsqueeze(-1) & valid.unsqueeze(-2))
    return keep


def cond_drop_stats(
    asym_id: torch.Tensor | None,  # [B, N]
    seq_mask: torch.Tensor,  # [B, N]
    *,
    use_conditioning: bool,
    pair_keep: torch.Tensor | None,  # [B, N, N]; set iff an interface drop fired
) -> dict[str, torch.Tensor]:
    """Per-forward conditioning-drop indicators for logging (0-d float tensors).

    ``none`` / ``full`` / ``interface`` are one-hot, so their window means are
    the realised drop rates. ``multichain`` is whether every batch element has
    ≥ 2 chains — the condition an interface drop needs — so its mean is the
    share of forwards eligible for one. ``interface_pair_frac`` (fraction of
    inter-chain token pairs zeroed) is present only on an interface drop.
    """
    dev = seq_mask.device
    interface = pair_keep is not None
    full = not use_conditioning
    out = {
        "none": torch.tensor(float(not full and not interface), device=dev),
        "full": torch.tensor(float(full), device=dev),
        "interface": torch.tensor(float(interface), device=dev),
    }
    if asym_id is not None:
        valid = seq_mask.bool()
        multi = all(asym_id[b][valid[b]].unique().numel() >= 2 for b in range(asym_id.shape[0]))
        out["multichain"] = torch.tensor(float(multi), device=dev)
        if interface:
            pv = valid.unsqueeze(-1) & valid.unsqueeze(-2)
            inter = pv & ~interface_pair_keep(asym_id)
            out["interface_pair_frac"] = (~pair_keep & inter).sum() / inter.sum().clamp(min=1)
    return out


def mask_trunk_pair(embeddings: dict[str, torch.Tensor], pair_keep: torch.Tensor) -> dict[str, torch.Tensor]:
    """Copy of ``embeddings`` with the trunk pair zeroed where ``pair_keep`` is False.

    Same effect as ``compute_static(use_conditioning=False)`` restricted to
    those entries: ``pair_cond`` is computed per (i, j), so a dropped entry
    gets exactly its fully-dropped value and a kept one its conditioned value.
    The caller's dict (shared with the confidence / distogram heads) is not
    modified.
    """
    pair = embeddings["pair"]
    return {**embeddings, "pair": pair * pair_keep.unsqueeze(-1).to(pair.dtype)}


def _augment_in_place(
    positions: torch.Tensor,  # [B?, S, N, M, 3]
    mask: torch.Tensor,  # [B?, N, M] bool
    generator: torch.Generator | None,
) -> None:
    """Apply random augmentation to each sample in-place.

    Pre-samples all random vectors in a single kernel call to reduce
    the number of small CUDA kernel launches from 3*B*S to 3.
    """
    assert positions.dim() in (4, 5), f"Expected 4D or 5D positions, got {positions.dim()}D"
    device = positions.device
    S = positions.shape[-4]

    if positions.dim() == 5:  # [B, S, N, M, 3]
        B = positions.shape[0]
        # Pre-sample all random vectors: 3 vectors per (b, s) pair
        v0_all = torch.randn(B, S, 3, device=device, generator=generator)
        v1_all = torch.randn(B, S, 3, device=device, generator=generator)
        trans_all = torch.randn(B, S, 3, device=device, generator=generator)
        for b in range(B):
            for s in range(S):
                positions[b, s] = _random_augmentation_from_vectors(
                    positions[b, s], mask[b], v0_all[b, s], v1_all[b, s], trans_all[b, s]
                )
    else:  # [S, N, M, 3]
        v0_all = torch.randn(S, 3, device=device, generator=generator)
        v1_all = torch.randn(S, 3, device=device, generator=generator)
        trans_all = torch.randn(S, 3, device=device, generator=generator)
        for s in range(S):
            positions[s] = _random_augmentation_from_vectors(positions[s], mask, v0_all[s], v1_all[s], trans_all[s])


class DiffusionTransformerBlock(nn.Module):
    """Single AdaLN-zero transformer block for the diffusion model.

    Haiku: self_attention + transition_block functions called repeatedly in a
    layer_stack under transformer~/

    The block uses:
    - AdaLN self-attention (conditioned on single_cond)
    - AdaLN transition block

    Attribute names (converter uses the module name, e.g. 'transformer'):
        self-attention parts (name=module_name, e.g. 'transformer'):
            {name}layer_norm              → attn_norm (no elementwise affine)
            {name}single_cond_layer_norm  → attn_cond_norm_w
            {name}single_cond_scale       → attn_cond_scale
            {name}single_cond_bias        → attn_cond_bias
            {name}q_projection            → to_q (w/ bias)
            {name}k_projection            → to_k (no bias)
            {name}v_projection            → to_v (no bias)
            {name}gating_query            → to_g
            {name}transition2             → out_proj
            {name}adaptive_zero_cond      → zero_cond (bias=-2)
        transition parts (name=module_name):
            {name}ffw_layer_norm          → ffn_norm (no elementwise affine)
            {name}ffw_single_cond_*       → ffn_cond_*
            {name}ffw_transition1         → ffn_linear1
            {name}ffw_transition2         → ffn_linear2
            {name}ffw_adaptive_zero_cond  → ffn_zero_cond (bias=-2)
    """

    def __init__(
        self,
        c: int,
        c_single_cond: int,
        num_heads: int = 16,
        key_dim: int | None = None,
        value_dim: int | None = None,
        num_intermediate_factor: int = 2,
        optim: OptimConfig | None = None,
        cond_bias_init_beta: float | None = None,
    ):
        super().__init__()
        # AF3: key_dim/value_dim default to c (num_channels) when None
        key_dim = key_dim if key_dim is not None else c
        value_dim = value_dim if value_dim is not None else c
        assert key_dim % num_heads == 0
        assert value_dim % num_heads == 0
        self.num_heads = num_heads
        self.key_dim_per_head = key_dim // num_heads
        self.val_dim_per_head = value_dim // num_heads
        self.scale = self.key_dim_per_head**-0.5

        num_int = c * num_intermediate_factor

        # Self-attention with AdaLN-zero
        # Conditioning inputs are c_single_cond-dimensional, outputs are c-dimensional
        self.attn_norm = LayerNorm(c, elementwise_affine=False)
        self.attn_cond_norm_w = nn.Parameter(torch.ones(c_single_cond))
        # AF3: single_cond_scale and single_cond_bias use initializer='zeros'
        self.attn_cond_scale = nn.Linear(c_single_cond, c)
        af3_linear_init_(self.attn_cond_scale.weight, "zeros")
        nn.init.zeros_(self.attn_cond_scale.bias)
        self.attn_cond_bias = LinearNoBias(c_single_cond, c, initializer="zeros")
        self.to_q = nn.Linear(c, key_dim)  # q_projection has bias
        af3_linear_init_(self.to_q.weight, "linear")
        nn.init.zeros_(self.to_q.bias)
        self.to_k = LinearNoBias(c, key_dim)
        self.to_v = LinearNoBias(c, value_dim)
        # AF3: gating_query uses initializer='zeros' (no bias)
        self.to_g = LinearNoBias(c, value_dim, initializer="zeros")
        # AF3: transition2 uses default init (with single_cond), adaptive_zero_cond gates it
        self.out_proj = LinearNoBias(value_dim, c)
        self.zero_cond = nn.Linear(c_single_cond, c)

        # FFN with AdaLN-zero
        self.ffn_norm = LayerNorm(c, elementwise_affine=False)
        self.ffn_cond_norm_w = nn.Parameter(torch.ones(c_single_cond))
        # AF3: single_cond_scale and single_cond_bias use initializer='zeros'
        self.ffn_cond_scale = nn.Linear(c_single_cond, c)
        af3_linear_init_(self.ffn_cond_scale.weight, "zeros")
        nn.init.zeros_(self.ffn_cond_scale.bias)
        self.ffn_cond_bias = LinearNoBias(c_single_cond, c, initializer="zeros")
        # AF3: ffw_transition1 uses initializer='relu'
        self.ffn_linear1 = LinearNoBias(c, num_int * 2, initializer="relu")
        self.ffn_linear2 = LinearNoBias(num_int, c)  # ffw_transition2 no bias
        self.ffn_zero_cond = nn.Linear(c_single_cond, c)

        # Initialize zero_cond gates
        nn.init.zeros_(self.zero_cond.weight)
        nn.init.constant_(self.zero_cond.bias, -2.0)
        nn.init.zeros_(self.ffn_zero_cond.weight)
        nn.init.constant_(self.ffn_zero_cond.bias, -2.0)

        # AdaLN ``cond_bias`` projections: AF3 ships with zeros init (so the
        # AdaLN bias starts at 0). ``cond_bias_init_beta`` overrides this with
        # a small trunc-normal init: ``W_b ~ trunc_normal(0, (β/√fan_in)²)``.
        # For unit-variance ``LN(single_cond)`` input, output stddev ≈ β.
        # Recommended starting range: β ∈ [1e-3, 3e-3]. ``None`` keeps zeros.
        # Applied to both the attention-side (``attn_cond_bias``) and the
        # FFN-side (``ffn_cond_bias``) projections of every block.
        if cond_bias_init_beta is not None:
            self._init_cond_bias_small_normal(self.attn_cond_bias.weight, cond_bias_init_beta)
            self._init_cond_bias_small_normal(self.ffn_cond_bias.weight, cond_bias_init_beta)

        # Fused FFN kernels (optional). Stateless — read parent's ffn_linear1
        # weight at call time (single source of truth; no sync).
        self._fused_ada_ln_linear = None
        self._fused_swiglu = None
        if optim is not None and optim.use_fused_ada_transition:
            try:
                from deepfold3.model.triton.ada_layernorm_linear import FusedAdaLayerNormLinear
                from deepfold3.model.triton.swiglu import FusedSwiGLU

                self._fused_ada_ln_linear = FusedAdaLayerNormLinear(c, num_int * 2)
                self._fused_swiglu = FusedSwiGLU()
            except ImportError:
                pass

        if optim is not None and optim.use_torch_compile:
            self._compute = torch.compile(self._compute)

    @staticmethod
    def _init_cond_bias_small_normal(weight: torch.Tensor, beta: float) -> None:
        """``W ~ trunc_normal(0, (β/√fan_in)²)`` — output stddev ≈ β for unit input."""
        if beta <= 0:
            raise ValueError(f"cond_bias_init_beta must be > 0, got {beta}")
        fan_in = weight.shape[1]
        target_std = float(beta) / math.sqrt(fan_in)
        std = target_std / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0, b=2.0)

    def _compute(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,  # None: already folded into ``pair_logits``
        single_cond: torch.Tensor,
        pair_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        Cs = single_cond.shape[-1]
        H = self.num_heads
        Dk = self.key_dim_per_head
        Dv = self.val_dim_per_head

        # AdaLN-zero self-attention
        xn = layer_norm(x, [C])
        cn = layer_norm(single_cond, [Cs]) * self.attn_cond_norm_w
        scale = torch.sigmoid(self.attn_cond_scale(cn))
        bias = self.attn_cond_bias(cn)
        xn = scale * xn + bias

        q = self.to_q(xn).view(B, N, H, Dk).permute(0, 2, 1, 3).float()  # [B, H, N, Dk]
        k = self.to_k(xn).view(B, N, H, Dk).permute(0, 2, 1, 3).float()
        v = self.to_v(xn).view(B, N, H, Dv).permute(0, 2, 1, 3).to(x.dtype)

        logits = torch.matmul(q * self.scale, k.transpose(-2, -1))  # [B, H, N, N]
        if pair_logits is not None:
            logits = logits + pair_logits.float()
        if mask is not None:
            logits = logits + (1e9 * (mask.float() - 1.0)).view(B, 1, 1, N)
        w = F.softmax(logits, dim=-1).to(x.dtype)

        out = torch.matmul(w, v)  # [B, H, N, Dv]
        out = out.permute(0, 2, 1, 3).reshape(B, N, H * Dv)

        g = torch.sigmoid(self.to_g(xn))
        out = out * g
        out = self.out_proj(out)
        gate = torch.sigmoid(self.zero_cond(single_cond))
        x = x + gate * out

        # AdaLN-zero FFN
        cn = layer_norm(single_cond, [Cs]) * self.ffn_cond_norm_w
        scale = torch.sigmoid(self.ffn_cond_scale(cn))
        bias = self.ffn_cond_bias(cn)

        if self._fused_ada_ln_linear is not None and x.is_cuda:
            # Stateless fused AdaLN+Linear reads parent's ffn_linear1 weight.
            h = self._fused_ada_ln_linear(x, scale, bias, self.ffn_linear1.weight.T)
            ff = self._fused_swiglu(h)
        else:
            xn = layer_norm(x, [C])
            xn = scale * xn + bias
            a, b = self.ffn_linear1(xn).chunk(2, dim=-1)
            ff = F.silu(a) * b

        ff_out = self.ffn_linear2(ff)
        gate = torch.sigmoid(self.ffn_zero_cond(single_cond))
        return x + gate * ff_out

    def forward(
        self,
        x: torch.Tensor,  # [B?, N, c]
        mask: torch.Tensor,  # [B?, N]
        single_cond: torch.Tensor,  # [B?, N, c_single_cond]
        pair_logits: torch.Tensor | None,  # [B?, H, N, N]
    ) -> torch.Tensor:
        has_batch = x.dim() == 3
        if not has_batch:
            x = x.unsqueeze(0)
            mask = mask.unsqueeze(0)
            single_cond = single_cond.unsqueeze(0)
            if pair_logits is not None:
                pair_logits = pair_logits.unsqueeze(0)

        x = self._compute(x, mask, single_cond, pair_logits)

        if not has_batch:
            x = x.squeeze(0)
        return x


class DiffusionTransformer(nn.Module):
    """Stack of DiffusionTransformerBlocks with pair-biased logits.

    Haiku: Transformer with num_blocks=24, super_block_size=4.

    Pair logits are recomputed once per super-block using a per-super-block
    projection (AF3 layer_stack gives each super-block its own weights).

    Attribute names:
        pair_norm_w            → bare Parameter (scale for pair LN, no offset)
        pair_logits_proj       → ModuleList[num_super] of Linear(c_pair, sbs*H)
        layers                 → ModuleList of DiffusionTransformerBlock
        output_norm_w          → bare Parameter (scale for output LN, no offset)
    """

    def __init__(
        self,
        c: int,
        c_single_cond: int,
        num_blocks: int = 24,
        num_heads: int = 16,
        key_dim: int | None = None,
        value_dim: int | None = None,
        num_intermediate_factor: int = 2,
        super_block_size: int = 4,
        c_pair: int = 128,
        optim: OptimConfig | None = None,
        cond_bias_init_beta: float | None = None,
    ):
        super().__init__()
        assert num_blocks % super_block_size == 0
        self.num_blocks = num_blocks
        self.super_block_size = super_block_size
        num_super = num_blocks // super_block_size

        # Scale-only LayerNorm for pair conditioning (create_offset=False in AF3)
        self.pair_norm_w = nn.Parameter(torch.ones(c_pair))
        # Per-super-block pair→logit projection (each super-block has own weights)
        self.pair_logits_proj = nn.ModuleList(
            [LinearNoBias(c_pair, super_block_size * num_heads) for _ in range(num_super)]
        )

        self.layers = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    c,
                    c_single_cond,
                    num_heads,
                    key_dim,
                    value_dim,
                    num_intermediate_factor,
                    optim=optim,
                    cond_bias_init_beta=cond_bias_init_beta,
                )
                for _ in range(num_blocks)
            ]
        )

        # Scale-only output norm (create_offset=False in AF3)
        self.output_norm_w = nn.Parameter(torch.ones(c))

    def hoist_pair_logits(
        self,
        pair_cond: torch.Tensor,  # [B, N, N, c_pair]
        mask: torch.Tensor,  # [B, N]  float
    ) -> tuple[torch.Tensor, ...]:
        """Per super-block pair logits ``[B, sbs*H, N, N]``, padding mask folded in."""
        pair_norm = layer_norm(pair_cond, [pair_cond.shape[-1]]) * self.pair_norm_w
        bias = 1e9 * (mask.float() - 1.0)[:, None, None, :]
        return tuple(proj(pair_norm).movedim(-1, -3).float() + bias for proj in self.pair_logits_proj)

    def forward_hoisted(
        self,
        act: torch.Tensor,  # [B, N, c]
        single_cond: torch.Tensor,  # [1 or B, N, c]
        pair_logits: tuple[torch.Tensor, ...],  # from :meth:`hoist_pair_logits`
    ) -> torch.Tensor:
        """:meth:`forward` on hoisted logits; ``single_cond`` may broadcast over ``act``."""
        H = self.layers[0].num_heads
        sbs = self.super_block_size
        if self.layers[0]._fused_ada_ln_linear is not None:  # the kernel does not broadcast
            single_cond = single_cond.expand(act.shape[0], -1, -1)
        for super_idx, pair_logits_all in enumerate(pair_logits):
            for block_in_super in range(sbs):
                start = block_in_super * H
                pl = pair_logits_all[..., start : start + H, :, :]
                act = self.layers[super_idx * sbs + block_in_super]._compute(act, None, single_cond, pl)
        return layer_norm(act, [act.shape[-1]]) * self.output_norm_w

    def forward(
        self,
        act: torch.Tensor,  # [B?, N, c]
        mask: torch.Tensor,  # [B?, N]  float
        single_cond: torch.Tensor,  # [B?, N, c]
        pair_cond: torch.Tensor,  # [B?, N, N, c_pair]
    ) -> torch.Tensor:
        H = self.layers[0].num_heads
        sbs = self.super_block_size
        num_super = self.num_blocks // sbs

        # Compute pair norm once (pair_cond is constant across super-blocks)
        pair_norm = layer_norm(pair_cond, [pair_cond.shape[-1]]) * self.pair_norm_w

        for super_idx in range(num_super):
            # Per-super-block pair logit projection
            pair_logits_all = self.pair_logits_proj[super_idx](pair_norm)  # [..., N, N, sbs*H]
            # Move last dim (sbs*H) before the two N dims: [..., sbs*H, N, N]
            pair_logits_all = pair_logits_all.movedim(-1, -3)

            for block_in_super in range(sbs):
                i = super_idx * sbs + block_in_super
                # Slice along the sbs*H dim (now at dim -3)
                start = block_in_super * H
                end = start + H
                pl = pair_logits_all[..., start:end, :, :]  # [..., H, N, N]
                act = self.layers[i](act, mask, single_cond, pl)

        act = layer_norm(act, [act.shape[-1]]) * self.output_norm_w
        return act


class DiffusionConditioningBlock(nn.Module):
    """Projects trunk embeddings to diffusion conditioning vectors.

    Haiku: DiffusionHead._conditioning()

    Attribute names (under diffusion_head~/):
        pair_cond_initial_norm       → pair_cond_norm   (no offset)
        pair_cond_initial_projection → pair_cond_proj
        pair_transition_0~/ffw_*     → pair_trans0.*
        pair_transition_1~/ffw_*     → pair_trans1.*
        single_cond_initial_norm     → single_cond_norm  (no offset)
        single_cond_initial_projection → single_cond_proj
        noise_embedding_initial_norm → noise_emb_norm   (no offset)
        noise_embedding_initial_projection → noise_emb_proj
        single_transition_0~/ffw_*   → single_trans0.*
        single_transition_1~/ffw_*   → single_trans1.*
    """

    def __init__(
        self,
        c_pair_in: int,
        c_single_in: int,
        rel_feat_dim: int,
        c_pair_out: int,
        c_single_out: int,
        target_feat_dim: int = 447,
        use_trunk_single: bool = True,
        sigma_data: float = 16.0,
        fourier_dim: int = 256,
        num_intermediate_factor: int = 2,
        optim: OptimConfig | None = None,
    ):
        super().__init__()
        self.sigma_data = sigma_data
        # Pair conditioning: concat(pair_emb, rel_feat) → linear
        # Concat-then-project conditioning with plain Lecun init over the full
        # concatenated fan-in (OF3/AF3 stock). The previous per-concat-slice
        # ``chunked_linear_init_`` was removed.
        pair_in = c_pair_in + rel_feat_dim
        self.pair_cond_norm_w = nn.Parameter(torch.ones(pair_in))
        self.pair_cond_proj = LinearNoBias(pair_in, c_pair_out)
        self.pair_trans0 = make_transition_block(
            c_pair_out, num_intermediate_factor=num_intermediate_factor, config=optim
        )
        self.pair_trans1 = make_transition_block(
            c_pair_out, num_intermediate_factor=num_intermediate_factor, config=optim
        )

        # Single conditioning: concat(single_embedding, target_feat) → linear.
        # ``use_trunk_single=False`` drops the trunk half and conditions on
        # ``target_feat`` alone — correct exactly when the trunk stops updating
        # the single track, at which point ``single_embedding`` is a fixed
        # pointwise function of ``target_feat`` and the concatenation is a
        # duplicate that also perturbs the shared LayerNorm's statistics. See
        # ``DiffusionConfig.conditioning_use_trunk_single``.
        self.use_trunk_single = bool(use_trunk_single)
        single_in = (c_single_in if self.use_trunk_single else 0) + target_feat_dim  # 384 + 447 = 831
        self.single_cond_norm_w = nn.Parameter(torch.ones(single_in))
        self.single_cond_proj = LinearNoBias(single_in, c_single_out)

        # Noise embedding conditioning — fixed (non-trainable) random Fourier
        # features. These were previously zero-initialised and only ever filled by
        # loading AF3-converted weights, so *from-scratch* training left them at 0
        # → cos(2π·(x·0+0)) = 1 for every σ → LN(const)=0 → the whole noise
        # embedding contributed nothing and the diffusion conditioning carried NO
        # explicit σ signal (the net could only see σ via the EDM input scaling,
        # which saturates to unit noise at high σ). Sample them here instead.
        # AF3's pseudocode says w,b ~ N(0,I) but its reference weights are
        # w ~ N(0,1) and b ~ Uniform[0,1) — b is a phase inside cos(2π·) (standard
        # Rahimi–Recht random Fourier features). A private fixed-seed generator
        # keeps the basis reproducible across runs (DDP already syncs buffers from
        # rank 0 at construction, so this is not for rank agreement);
        # load_state_dict overwrites these when initialising from AF3-converted
        # weights or resuming.
        _fourier_gen = torch.Generator().manual_seed(_FOURIER_INIT_SEED)
        self.register_buffer("fourier_weight", torch.randn(fourier_dim, generator=_fourier_gen))
        self.register_buffer("fourier_bias", torch.rand(fourier_dim, generator=_fourier_gen))
        self.noise_emb_norm_w = nn.Parameter(torch.ones(fourier_dim))
        self.noise_emb_proj = LinearNoBias(fourier_dim, c_single_out)

        self.single_trans0 = make_transition_block(
            c_single_out, num_intermediate_factor=num_intermediate_factor, config=optim
        )
        self.single_trans1 = make_transition_block(
            c_single_out, num_intermediate_factor=num_intermediate_factor, config=optim
        )

    def _fourier_noise_embedding(self, sigma: torch.Tensor) -> torch.Tensor:
        """Fixed Fourier embedding of noise level using registered buffers.

        sigma: scalar or batch of noise levels.
        Returns: [..., fourier_dim]

        Matches AF3 noise_level_embeddings.py:138-139 — weights/biases are
        hardcoded fp32 constants in the reference. We upcast here to keep
        the embedding numerically stable under bf16 autocast; downstream
        LayerNorm/Linear in ``add_noise`` cast back to the ambient dtype.
        """
        sigma_f = sigma.float()
        x = (1 / 4) * torch.log(sigma_f)
        w = self.fourier_weight.float()
        b = self.fourier_bias.float()
        emb = x.unsqueeze(-1) * w + b
        return torch.cos(2 * math.pi * emb)

    def compute_static(
        self,
        single_emb: torch.Tensor,  # [B?, N, c_single_in]
        pair_emb: torch.Tensor,  # [B?, N, N, c_pair_in]
        target_feat: torch.Tensor,  # [B?, N, c_single_in]
        rel_feat: torch.Tensor,  # [B?, N, N, rel_feat_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Noise-independent half: returns ``(single_cond_pre_noise, pair_cond)``.

        ``pair_cond`` is final and constant across noise samples; the noise
        embedding is added to ``single_cond_pre_noise`` in :meth:`add_noise`.
        """
        # Pair cond
        pair_in = torch.cat([pair_emb, rel_feat], dim=-1)
        pair_norm = layer_norm(pair_in, [pair_in.shape[-1]]) * self.pair_cond_norm_w
        pair_cond = self.pair_cond_proj(pair_norm)
        pair_cond = pair_cond + self.pair_trans0(pair_cond)
        pair_cond = pair_cond + self.pair_trans1(pair_cond)

        # Single cond (without noise)
        single_in = torch.cat([single_emb, target_feat], dim=-1) if self.use_trunk_single else target_feat
        single_norm = layer_norm(single_in, [single_in.shape[-1]]) * self.single_cond_norm_w
        single_cond_pre_noise = self.single_cond_proj(single_norm)
        return single_cond_pre_noise, pair_cond

    def add_noise(
        self,
        single_cond_pre_noise: torch.Tensor,  # [B?, N, c_single_out]
        noise_level: torch.Tensor,  # scalar or [B]
    ) -> torch.Tensor:
        """Add the Fourier noise embedding and the two single transitions."""
        noise_emb = self._fourier_noise_embedding(noise_level / self.sigma_data)
        while noise_emb.dim() < single_cond_pre_noise.dim():
            noise_emb = noise_emb.unsqueeze(-2)
        noise_expanded = noise_emb.expand(*single_cond_pre_noise.shape[:-1], -1)
        noise_norm = layer_norm(noise_expanded, [noise_emb.shape[-1]]) * self.noise_emb_norm_w
        single_cond = single_cond_pre_noise + self.noise_emb_proj(noise_norm)
        single_cond = single_cond + self.single_trans0(single_cond)
        single_cond = single_cond + self.single_trans1(single_cond)
        return single_cond

    def forward(
        self,
        single_emb: torch.Tensor,  # [N, c_single_in]
        pair_emb: torch.Tensor,  # [N, N, c_pair_in]
        target_feat: torch.Tensor,  # [N, c_single_in]
        rel_feat: torch.Tensor,  # [N, N, rel_feat_dim]
        noise_level: torch.Tensor,  # scalar
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single_cond_pre_noise, pair_cond = self.compute_static(single_emb, pair_emb, target_feat, rel_feat)
        single_cond = self.add_noise(single_cond_pre_noise, noise_level)
        return single_cond, pair_cond


class DiffusionHead(nn.Module):
    """Full diffusion head: conditioning + atom encoder + transformer + atom decoder.

    Haiku scope: diffuser~/diffusion_head~/
    """

    def __init__(
        self,
        cfg: DiffusionConfig,
        c_single: int = 384,
        c_pair: int = 128,
        rel_feat_dim: int = 139,
        optim: OptimConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        c_diff = cfg.atom_cross_att.per_token_channels  # 768

        # Conditioning
        self.conditioning = DiffusionConditioningBlock(
            c_pair_in=c_pair,
            c_single_in=c_single,
            rel_feat_dim=rel_feat_dim,
            c_pair_out=cfg.conditioning_pair_channel,
            c_single_out=cfg.conditioning_seq_channel,
            use_trunk_single=cfg.conditioning_use_trunk_single,
            sigma_data=cfg.sigma_data,
            fourier_dim=cfg.fourier_dim,
            num_intermediate_factor=cfg.conditioning_num_intermediate_factor,
            optim=optim,
        )

        # Atom encoder (per-atom conditioning → token activations)
        self.atom_encoder = AtomEncoder(
            per_atom_channels=cfg.atom_cross_att.per_atom_channels,
            per_token_channels=cfg.atom_cross_att.per_token_channels,
            per_atom_pair_channels=cfg.atom_cross_att.per_atom_pair_channels,
            atom_transformer_num_blocks=cfg.atom_cross_att.atom_transformer.num_blocks,
            Q=cfg.atom_cross_att.queries_subset_size,
            K=cfg.atom_cross_att.keys_subset_size,
            num_element_classes=cfg.atom_cross_att.num_element_classes,
            num_atom_name_chars=cfg.atom_cross_att.num_atom_name_chars,
            atom_name_length=cfg.atom_cross_att.atom_name_length,
            ref_element_zero_index=cfg.atom_cross_att.ref_element_zero_index,
            # compute_static() feeds trunk_single=embeddings["single"] and
            # trunk_pair=pair_cond (the DiffusionConditioning output), so these
            # must track the conditioning widths, not the AF3 defaults.
            c_trunk_single=c_single,
            c_trunk_pair=cfg.conditioning_pair_channel,
        )

        # Project trunk single to diffusion act (scale-only LN + no-bias projection)
        # AF3: single_cond_embedding_projection uses final_init (zeros)
        self.single_cond_emb_norm_w = nn.Parameter(torch.ones(cfg.conditioning_seq_channel))
        self.single_cond_embedding_projection = LinearNoBias(cfg.conditioning_seq_channel, c_diff, initializer="zeros")

        # Main diffusion transformer
        self.transformer = DiffusionTransformer(
            c=c_diff,
            c_single_cond=cfg.conditioning_seq_channel,
            num_blocks=cfg.transformer.num_blocks,
            num_heads=cfg.transformer.num_head,
            key_dim=cfg.transformer.key_dim,
            value_dim=cfg.transformer.value_dim,
            num_intermediate_factor=cfg.transformer.num_intermediate_factor,
            super_block_size=cfg.transformer.super_block_size,
            c_pair=cfg.conditioning_pair_channel,
            optim=optim,
            cond_bias_init_beta=cfg.transformer.cond_bias_init_beta,
        )

        # Atom decoder
        self.atom_decoder = AtomDecoder(
            per_atom_channels=cfg.atom_cross_att.per_atom_channels,
            per_token_channels=c_diff,
            per_atom_pair_channels=cfg.atom_cross_att.per_atom_pair_channels,
            atom_transformer_num_blocks=cfg.atom_cross_att.atom_transformer.num_blocks,
        )

    def compute_static(
        self,
        embeddings: dict[str, torch.Tensor],
        ref_pos: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_element: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        seq_mask: torch.Tensor,
        rel_feat: torch.Tensor,
        ref_space_uid: torch.Tensor | None = None,  # [B?, N, M] int64
        use_conditioning: bool = True,
    ) -> DiffusionStatic:
        """Pre-compute everything that does NOT depend on noisy positions /
        noise level: conditioning ``pair_cond``, single-cond pre-noise, and the
        atom encoder static state. Reuse via :meth:`denoise` ``static=`` kwarg.

        ``use_conditioning=False`` is the OpenFold-3 trunk-conditioning drop
        (``use_conditioning_prob``, `diffusion_conditioning.py:241-249`): the
        trunk pair — and with ``cfg.cond_drop == "both"`` the trunk single — is
        zeroed on the way into the conditioning block while ``target_feat`` /
        ``rel_feat`` survive. Matching OF3 and AF3
        Algorithm 20, the **un-zeroed** trunk single is still handed to the atom
        encoder below (`diffusion_module.py:198-204`), so the drop is partial.
        """
        cond_single, cond_pair = embeddings["single"], embeddings["pair"]
        if not use_conditioning:
            cond_pair = cond_pair * 0
            if self.cfg.cond_drop == "both":
                cond_single = cond_single * 0
        single_cond_pre_noise, pair_cond = self.conditioning.compute_static(
            single_emb=cond_single,
            pair_emb=cond_pair,
            target_feat=embeddings["target_feat"],
            rel_feat=rel_feat,
        )
        encoder_static = self.atom_encoder.compute_static(
            ref_pos=ref_pos,
            ref_mask=ref_mask,
            ref_element=ref_element,
            ref_charge=ref_charge,
            ref_atom_name_chars=ref_atom_name_chars,
            ref_space_uid=ref_space_uid,
            trunk_single=embeddings["single"],
            trunk_pair=pair_cond,
        )
        return DiffusionStatic(
            encoder_static=encoder_static,
            pair_cond=pair_cond,
            single_cond_pre_noise=single_cond_pre_noise,
            ref_pos=ref_pos,
            ref_mask=ref_mask,
            seq_mask=seq_mask,
        )

    def hoist_static(self, static: DiffusionStatic, num_samples: int) -> DiffusionStatic:
        """``static`` for a sampler of ``num_samples``, with the work the plain
        expansion repeats per step moved here.

        Replaces :meth:`DiffusionStatic.expand_for_samples` in the sampler
        (``cfg.hoist_sampler``): drops atom windows that hold no atoms, computes
        the pair logits and atom AdaLN terms once, and keeps one copy of the
        state rather than ``num_samples``. The result is read-only and assumes
        one noise level across the batch, which the sampler guarantees.

        The arithmetic is unchanged but GEMM shapes are not, and under TF32
        (``matmul_precision("high")``) cuBLAS results depend on shape: changing
        only the sample count already moves the plain denoiser by ~0.1 Å. Expect
        that, not bit equality; see docs/optimizations.md.
        """
        enc = static.encoder_static.trim_empty_subsets()
        hoist_args = (enc.queries_single, enc.keys_single, enc.keys_mask, enc.atom_pair)
        encoder_hoist = self.atom_encoder.atom_transformer.hoist(*hoist_args)
        decoder_hoist = self.atom_decoder.atom_transformer.hoist(*hoist_args)
        pair_logits = self.transformer.hoist_pair_logits(static.pair_cond, static.seq_mask)
        shared = enc.share_for_samples(num_samples)
        return DiffusionStatic(
            encoder_static=dataclasses.replace(
                shared,
                encoder_hoist=tuple(h.broadcast_for_samples(num_samples) for h in encoder_hoist),
                decoder_hoist=tuple(h.broadcast_for_samples(num_samples) for h in decoder_hoist),
            ),
            pair_cond=static.pair_cond,
            single_cond_pre_noise=static.single_cond_pre_noise,
            ref_pos=static.ref_pos,
            ref_mask=expand_for_samples_shared(static.ref_mask, num_samples),
            seq_mask=static.seq_mask,
            pair_logits=tuple(broadcast_for_samples(t, num_samples) for t in pair_logits),
        )

    def denoise(
        self,
        noisy_pos: torch.Tensor,  # [B?, N, M, 3]
        noise_level: torch.Tensor,  # scalar or [B]
        embeddings: dict[str, torch.Tensor] | None = None,
        ref_pos: torch.Tensor | None = None,  # [B?, N, M, 3]
        ref_mask: torch.Tensor | None = None,  # [B?, N, M]
        ref_element: torch.Tensor | None = None,  # [B?, N, M] int64
        ref_charge: torch.Tensor | None = None,  # [B?, N, M]
        ref_atom_name_chars: torch.Tensor | None = None,  # [B?, N, M, 4] int64
        ref_space_uid: torch.Tensor | None = None,  # [B?, N, M] int64
        seq_mask: torch.Tensor | None = None,  # [B?, N]
        rel_feat: torch.Tensor | None = None,  # [B?, N, N, rel_feat_dim]
        static: DiffusionStatic | None = None,
        use_conditioning: bool = True,
    ) -> torch.Tensor:
        """Run one denoising step. Returns denoised positions [B?, N, M, 3].

        When ``static`` is provided, the noise-independent inputs (``embeddings``,
        ``ref_*``, ``seq_mask``, ``rel_feat``) are ignored — the cached state is
        reused. Otherwise the static is rebuilt inline from those inputs.
        """
        sd = self.cfg.sigma_data

        if static is None:
            static = self.compute_static(
                embeddings=embeddings,
                ref_pos=ref_pos,
                ref_mask=ref_mask,
                ref_element=ref_element,
                ref_charge=ref_charge,
                ref_atom_name_chars=ref_atom_name_chars,
                ref_space_uid=ref_space_uid,
                seq_mask=seq_mask,
                rel_feat=rel_feat,
                use_conditioning=use_conditioning,
            )

        ref_mask_eff = static.ref_mask
        seq_mask_eff = static.seq_mask
        pair_cond = static.pair_cond

        # Reshape noise_level for broadcasting with position tensors
        nl = noise_level
        while nl.dim() < noisy_pos.dim():
            nl = nl.unsqueeze(-1)

        # Scale inputs
        act_pos = noisy_pos * ref_mask_eff.unsqueeze(-1)
        act_pos = act_pos / (nl**2 + sd**2).sqrt()

        # Add noise embedding to single conditioning
        if static.pair_logits is None:
            single_cond = self.conditioning.add_noise(static.single_cond_pre_noise, noise_level)
        else:
            # One noise level per step, so each molecule needs one row.
            S = noisy_pos.shape[0] // static.single_cond_pre_noise.shape[0]
            single_cond = self.conditioning.add_noise(static.single_cond_pre_noise, noise_level[::S])
            single_cond = broadcast_for_samples(single_cond, S)

        # Atom encoder (uses cached static)
        enc = self.atom_encoder(noisy_atom_pos=act_pos, static=static.encoder_static)
        act = enc.token_act.float()

        # Add trunk single conditioning to token activations
        sc_norm = layer_norm(single_cond, [single_cond.shape[-1]]) * self.single_cond_emb_norm_w
        act = act + self.single_cond_embedding_projection(sc_norm)

        # Main transformer (already batch-aware)
        if static.pair_logits is None:
            act = self.transformer(act, seq_mask_eff, single_cond, pair_cond)
        else:
            act = self.transformer.forward_hoisted(act, single_cond, static.pair_logits)

        # Atom decoder
        pos_update = self.atom_decoder(act, enc)

        # Combine skip + update (AF3 EDM-style)
        skip_scaling = sd**2 / (nl**2 + sd**2)
        out_scaling = nl * sd / (nl**2 + sd**2).sqrt()
        result = skip_scaling * noisy_pos + out_scaling * pos_update
        return result * ref_mask_eff.unsqueeze(-1)

    def _denoise_samples(
        self,
        noisy_pos: torch.Tensor,  # [B?, S, N, M, 3]
        noise_level: torch.Tensor,  # scalar or [B?, S]
        embeddings: dict[str, torch.Tensor] | None = None,
        ref_pos: torch.Tensor | None = None,
        ref_mask: torch.Tensor | None = None,
        ref_element: torch.Tensor | None = None,
        ref_charge: torch.Tensor | None = None,
        ref_atom_name_chars: torch.Tensor | None = None,
        ref_space_uid: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
        rel_feat: torch.Tensor | None = None,
        static: DiffusionStatic | None = None,
        static_bs: DiffusionStatic | None = None,
        use_conditioning: bool = True,
    ) -> torch.Tensor:
        """Denoise S diffusion samples in a SINGLE batched ``denoise`` call.

        Collapses ``[B?, S, N, M, 3]`` into a single ``B*S`` batch dim and
        invokes ``denoise`` once. The diffusion transformer runs with ``B*S``
        parallelism in one shot rather than as ``S`` sequential calls.

        ``static`` (at the original ``[B, ...]`` layout) is expanded to
        ``[B*S, ...]`` here. ``static_bs`` is the pre-expanded form — if the
        caller already invoked ``static.expand_for_samples(S)`` once outside
        a loop (the inference / mini-rollout pattern), passing it via this
        kwarg avoids re-materialising the expanded copy on every step.

        Wrapped in ``torch.utils.checkpoint(use_reentrant=False)`` so the
        entire batched forward is recomputed once during backward, freeing
        intermediates between forward and backward.
        """
        # Determine input layout: [S, N, M, 3] (unbatched) or [B, S, N, M, 3]
        has_batch = noisy_pos.dim() == 5
        if has_batch:
            B, S = noisy_pos.shape[0], noisy_pos.shape[1]
            shape_tail = noisy_pos.shape[2:]  # (N, M, 3)
        else:
            B, S = 1, noisy_pos.shape[0]
            shape_tail = noisy_pos.shape[1:]
            noisy_pos = noisy_pos.unsqueeze(0)  # [1, S, N, M, 3]

        BS = B * S

        # Collapse position tensor: [B, S, N, M, 3] → [BS, N, M, 3]
        noisy_pos_bs = noisy_pos.reshape(BS, *shape_tail).contiguous()

        # Noise level: scalar / [S] / [B] / [B, S] → [BS]
        if noise_level.dim() == 0:
            nl_bs = noise_level.expand(BS)
        elif noise_level.dim() == 1:
            if noise_level.shape[0] == S and not has_batch:
                nl_bs = noise_level.repeat(B)
            elif noise_level.shape[0] == B and has_batch:
                nl_bs = noise_level.repeat_interleave(S)
            else:
                nl_bs = noise_level.reshape(-1)
        elif noise_level.dim() == 2:
            nl_bs = noise_level.reshape(BS)
        else:
            nl_bs = noise_level.reshape(BS)

        # Resolve the static cache for this batched call. Three modes:
        #   1. Caller passed ``static_bs`` already at [B*S, ...] → use directly.
        #   2. Caller passed ``static`` at [B, ...] → expand here.
        #   3. Neither → build from inputs at [1, ...] then expand.
        if static_bs is None:
            if static is None:
                # Promote unbatched ref_* to [1, ...] so the encoder/conditioning
                # always see a batched layout.
                def _b(t: torch.Tensor | None) -> torch.Tensor | None:
                    if t is None:
                        return None
                    return t if has_batch else t.unsqueeze(0)

                static = self.compute_static(
                    embeddings={k: _b(v) for k, v in embeddings.items()},
                    ref_pos=_b(ref_pos),
                    ref_mask=_b(ref_mask),
                    ref_element=_b(ref_element),
                    ref_charge=_b(ref_charge),
                    ref_atom_name_chars=_b(ref_atom_name_chars),
                    ref_space_uid=_b(ref_space_uid),
                    seq_mask=_b(seq_mask),
                    rel_feat=_b(rel_feat),
                    use_conditioning=use_conditioning,
                )
            static_bs = static.expand_for_samples(S)

        use_ckpt = torch.is_grad_enabled()
        if use_ckpt:
            result_bs = torch_checkpoint(
                self.denoise,
                noisy_pos_bs,
                nl_bs,
                None,  # embeddings — unused when static is provided
                None,  # ref_pos
                None,  # ref_mask
                None,  # ref_element
                None,  # ref_charge
                None,  # ref_atom_name_chars
                None,  # ref_space_uid
                None,  # seq_mask
                None,  # rel_feat
                static_bs,
            )
        else:
            result_bs = self.denoise(
                noisy_pos=noisy_pos_bs,
                noise_level=nl_bs,
                static=static_bs,
            )

        # Reshape back: [BS, N, M, 3] → [B, S, N, M, 3] → caller layout
        result = result_bs.reshape(B, S, *result_bs.shape[1:])
        if not has_batch:
            result = result.squeeze(0)
        return result

    @torch.no_grad()
    def sample(
        self,
        embeddings: dict[str, torch.Tensor],
        ref_pos: torch.Tensor,  # [B?, N, M, 3]
        ref_mask: torch.Tensor,  # [B?, N, M]
        ref_element: torch.Tensor,  # [B?, N, M]
        ref_charge: torch.Tensor,  # [B?, N, M]
        ref_atom_name_chars: torch.Tensor,  # [B?, N, M, 4]
        seq_mask: torch.Tensor,  # [B?, N]
        rel_feat: torch.Tensor,  # [B?, N, N, rel_feat_dim]
        generator: torch.Generator | None = None,
        ref_space_uid: torch.Tensor | None = None,  # [B?, N, M] int64
        guidance: bool = True,
        guidance_pair_keep: torch.Tensor | None = None,  # [B?, N, N] bool
        guidance_terms: Sequence[GuidanceTerm] | None = None,
        restraints: Sequence[Restraint] = (),
        restraint_weight: float = 0.0,
        restraint_rigid: bool = True,
        restraint_max_step: float = 2.0,
        restraint_after_step: bool = False,
        restraint_after_sigma: float = 16.0,
    ) -> dict[str, torch.Tensor]:
        """Draw samples from the diffusion model (Euler-Maruyama sampler).

        Supports both unbatched and batched inputs. For unbatched inputs,
        returns 'atom_positions' [S, N, M, 3] and 'mask' [S, N, M].
        For batched inputs, returns [B, S, N, M, 3] and [B, S, N, M].

        Classifier-free guidance follows ``cfg.guidance_*``; ``guidance=False``
        turns it off for this call (the training mini-rollout). The
        unconditional branch zeroes the trunk pair where
        ``guidance_pair_keep`` is False — any subset of chain pairs — and
        falls back to the ``cond_drop`` drop when it is ``None``.
        ``cfg.guidance_drop == "interface"`` requires it (see
        :func:`interface_pair_keep`). ``guidance_terms``, when given, replaces
        all of that with a linear combination of terms (:class:`GuidanceTerm`).
        """
        cfg = self.cfg
        terms: tuple[GuidanceTerm, ...] = ()
        if not guidance:
            pass
        elif guidance_terms is not None:
            terms = tuple(t for t in guidance_terms if t.scale != 1.0)
        elif cfg.guidance_scale != 1.0:
            if cfg.guidance_drop == "interface" and guidance_pair_keep is None:
                raise ValueError("guidance_drop='interface' needs guidance_pair_keep (interface_pair_keep(asym_id))")
            terms = (GuidanceTerm(cfg.guidance_scale, guidance_pair_keep, cfg.guidance_sigma_range),)
        if cfg.skip_amp_sample:
            # Protenix-style ``skip_amp.sample_diffusion``: upcast float
            # inputs and run the full sampler with autocast disabled.
            embeddings = _upcast_fp32(embeddings)
            ref_pos = _upcast_fp32(ref_pos)
            ref_charge = _upcast_fp32(ref_charge)
            seq_mask = _upcast_fp32(seq_mask)
            rel_feat = _upcast_fp32(rel_feat)
            amp_ctx = torch.amp.autocast(device_type=ref_mask.device.type, enabled=False)
        else:
            amp_ctx = nullcontext()
        with amp_ctx:
            return self._sample_impl(
                embeddings,
                ref_pos,
                ref_mask,
                ref_element,
                ref_charge,
                ref_atom_name_chars,
                seq_mask,
                rel_feat,
                generator,
                ref_space_uid=ref_space_uid,
                guidance_terms=terms,
                restraints=tuple(restraints),
                restraint_weight=restraint_weight,
                restraint_rigid=restraint_rigid,
                restraint_max_step=restraint_max_step,
                restraint_after_step=restraint_after_step,
                restraint_after_sigma=restraint_after_sigma,
            )

    def _sample_impl(
        self,
        embeddings: dict[str, torch.Tensor],
        ref_pos: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_element: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        seq_mask: torch.Tensor,
        rel_feat: torch.Tensor,
        generator: torch.Generator | None,
        ref_space_uid: torch.Tensor | None = None,  # [B?, N, M] int64
        guidance_terms: tuple[GuidanceTerm, ...] = (),
        restraints: tuple[Restraint, ...] = (),
        restraint_weight: float = 0.0,
        restraint_rigid: bool = True,
        restraint_max_step: float = 2.0,
        restraint_after_step: bool = False,
        restraint_after_sigma: float = 16.0,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        N, M = ref_mask.shape[-2:]
        device = ref_mask.device
        batch_shape = ref_mask.shape[:-2]  # () or (B,)

        noise_levels = noise_schedule(
            torch.linspace(0, 1, cfg.steps + 1, device=device),
            sigma_data=cfg.sigma_data,
            smin=cfg.smin,
            smax=cfg.smax,
            p=cfg.noise_p,
        )

        # Pre-compute per-step gamma to avoid GPU→CPU sync inside the loop
        gammas = torch.where(noise_levels[1:] > cfg.gamma_min, cfg.gamma_0, 0.0)  # [steps]

        # Initialize from pure noise: [B?, S, N, M, 3]
        positions = torch.randn(*batch_shape, cfg.num_samples, N, M, 3, device=device, generator=generator)
        positions = positions * noise_levels[0]
        aug_mask = ref_mask.bool()

        # Compute the noise-independent static cache once and reuse it for all
        # ``steps`` × ``num_samples`` denoise calls.  Promote unbatched inputs
        # to a leading B=1 so the encoder/conditioning always see batched.
        has_batch = ref_pos.dim() == 4

        def _ensure_b(t: torch.Tensor | None) -> torch.Tensor | None:
            if t is None:
                return None
            return t if has_batch else t.unsqueeze(0)

        static_inputs = dict(
            embeddings={k: _ensure_b(v) for k, v in embeddings.items()},
            ref_pos=_ensure_b(ref_pos),
            ref_mask=_ensure_b(ref_mask),
            ref_element=_ensure_b(ref_element),
            ref_charge=_ensure_b(ref_charge),
            ref_atom_name_chars=_ensure_b(ref_atom_name_chars),
            ref_space_uid=_ensure_b(ref_space_uid),
            seq_mask=_ensure_b(seq_mask),
            rel_feat=_ensure_b(rel_feat),
        )
        static = self.compute_static(**static_inputs)

        # Pre-expand to [B*num_samples, ...] once. ``_denoise_samples`` will
        # use this directly for all 200 steps instead of materialising a
        # fresh expanded copy each step.
        def _for_samples(st: DiffusionStatic) -> DiffusionStatic:
            return (
                self.hoist_static(st, cfg.num_samples) if cfg.hoist_sampler else st.expand_for_samples(cfg.num_samples)
            )

        static_bs = _for_samples(static)

        # Classifier-free guidance: one static per unconditional branch, built
        # only for active terms so the unguided path costs nothing.
        uncond: list[tuple[float, object, float, float]] = []
        for term in guidance_terms:
            if term.pair_keep is not None:
                u_inputs = {
                    **static_inputs,
                    "embeddings": mask_trunk_pair(static_inputs["embeddings"], _ensure_b(term.pair_keep)),
                }
                static_u = self.compute_static(**u_inputs)
            else:
                static_u = self.compute_static(**static_inputs, use_conditioning=False)
            lo, hi = term.sigma_range or (float("-inf"), float("inf"))
            uncond.append((term.scale, _for_samples(static_u), lo, hi))
        # Host-side sigma per step: noise_levels is tiny, one sync here. Both
        # the guidance interval test and the restraint schedule read it.
        sig_hat_host_all = (noise_levels[:-1] * (1 + gammas)).tolist()
        sig_hat_host = sig_hat_host_all

        for step in range(cfg.steps):
            sigma_prev = noise_levels[step]
            sigma_next = noise_levels[step + 1]

            # Random augmentation per sample — required for AF3 equivalence.
            _augment_in_place(positions, aug_mask, generator)

            # Stochastic gamma step (pre-computed, no GPU sync).
            gamma = gammas[step]
            sigma_hat = sigma_prev * (1 + gamma)

            # Add noise
            noise_scale = cfg.noise_scale * (sigma_hat**2 - sigma_prev**2).sqrt()
            positions_hat = positions + noise_scale * torch.randn_like(positions, generator=generator)

            den_c = self._denoise_samples(positions_hat, sigma_hat, static_bs=static_bs)
            denoised = den_c
            for w, static_u_bs, lo, hi in uncond:
                if lo <= sig_hat_host[step] <= hi:
                    # Same positions_hat → all predictions share the augmented frame.
                    den_u = self._denoise_samples(positions_hat, sigma_hat, static_bs=static_u_bs)
                    denoised = denoised + (w - 1.0) * (den_c - den_u)
            # Steering the prediction and pushing the coordinates are not
            # alternatives, they are two halves of the schedule. Pushing holds
            # the placement hardest (7uvs fnat 0.545 against 0.424) but leaves
            # the surroundings rough (iRMSD 7.4 against 6.3, ICS 0.07 against
            # 0.16), because the model gets less room to repair. So when
            # --restraint-after-step is on, the push runs while sigma_hat is
            # above the gate and this branch takes over below it: hold the
            # chain where it belongs early, then negotiate the details.
            steer = not restraint_after_step or sig_hat_host_all[step] < restraint_after_sigma
            if restraints and restraint_weight and steer:
                # Steer the predicted clean structure, not the noisy positions:
                # D is what the Euler step aims at, so pulling it into the
                # restraint moves the trajectory without adding noise of its
                # own. A gradient step of weight/2 on the quadratic penalty
                # removes that fraction of each violation, so the restraint is
                # approached geometrically over the steps rather than snapped.
                # The soft-min sharpens as the noise falls: early on the pull
                # is on the group, late on the closest atoms.
                active = active_restraints(restraints, sig_hat_host_all[step])
                if active:
                    sharp = 0.25 + 1.75 * (1.0 - min(sig_hat_host_all[step] / cfg.sigma_data, 1.0))
                    if restraint_rigid:
                        # Move each restrained chain as a body: a subset pull
                        # stretches the chain where the selection ends.
                        denoised = denoised + rigid_displacement(
                            denoised,
                            ref_mask.bool(),
                            active,
                            weight=restraint_weight,
                            sharpness=sharp,
                            max_step=restraint_max_step,
                        )
                    else:
                        grad = restraint_gradient(denoised, ref_mask.bool(), active, sharpness=sharp)
                        move = -0.5 * restraint_weight * grad
                        norm = move.norm(dim=-1, keepdim=True)
                        denoised = denoised + move * (restraint_max_step / norm.clamp(min=restraint_max_step))
            d_t = sigma_next - sigma_hat
            positions = positions_hat + cfg.step_scale * d_t * (positions_hat - denoised) / sigma_hat

            if restraints and restraint_weight and restraint_after_step:
                # Push the trajectory itself rather than the prediction it aims
                # at. Steering ``denoised`` passes through the Euler
                # coefficient (step_scale*|d_t|/sigma_hat, ~0.7 over most of the
                # schedule) and the next denoiser call is free to put the chain
                # back; this is not negotiable with the model. It is therefore
                # also unsupervised: the restraint knows nothing about clashes,
                # so a rigid push can drive two chains into each other. Hence
                # the gate — below ``restraint_after_sigma`` the loop stops
                # pushing and the remaining steps are pure denoising, which
                # repairs what the push broke and leaves the final structure a
                # model output rather than a shoved one.
                # Never on the last step: whatever the gate says, the final
                # update is then a denoiser output, so the structure the caller
                # gets is one the model signed off on. At that sigma_hat the
                # model polishes rather than relocates, so holding the chain
                # until then costs nothing and the placement survives.
                last = step == cfg.steps - 1
                if sig_hat_host_all[step] >= restraint_after_sigma and not last:
                    active = active_restraints(restraints, sig_hat_host_all[step])
                    if active:
                        sharp = 0.25 + 1.75 * (1.0 - min(sig_hat_host_all[step] / cfg.sigma_data, 1.0))
                        if restraint_rigid:
                            positions = positions + rigid_displacement(
                                positions,
                                ref_mask.bool(),
                                active,
                                weight=restraint_weight,
                                sharpness=sharp,
                                max_step=restraint_max_step,
                            )
                        else:
                            grad = restraint_gradient(positions, ref_mask.bool(), active, sharpness=sharp)
                            move = -0.5 * restraint_weight * grad
                            norm = move.norm(dim=-1, keepdim=True)
                            positions = positions + move * (restraint_max_step / norm.clamp(min=restraint_max_step))

        mask_expanded = ref_mask.unsqueeze(-3).expand(*batch_shape, cfg.num_samples, N, M)
        return {
            "atom_positions": positions,
            "mask": mask_expanded,
        }

    def sample_training(
        self,
        gt_positions: torch.Tensor,  # [B?, N, M, 3]
        gt_mask: torch.Tensor,  # [B?, N, M]
        embeddings: dict[str, torch.Tensor],
        ref_pos: torch.Tensor,  # [B?, N, M, 3]
        ref_mask: torch.Tensor,  # [B?, N, M]
        ref_element: torch.Tensor,  # [B?, N, M]
        ref_charge: torch.Tensor,  # [B?, N, M]
        ref_atom_name_chars: torch.Tensor,  # [B?, N, M, 4]
        seq_mask: torch.Tensor,  # [B?, N]
        rel_feat: torch.Tensor,  # [B?, N, N, rel_feat_dim]
        num_samples: int = 1,
        static: DiffusionStatic | None = None,
        ref_space_uid: torch.Tensor | None = None,  # [B?, N, M] int64
        clean_mask: torch.Tensor | None = None,  # [B?, S, N] bool
        use_conditioning: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward: augment GT, add noise, denoise once (AF3 §3.7.1).

        ``clean_mask`` marks tokens whose (augmented) ground-truth coordinates
        are substituted back into the noisy input — masked / partial diffusion,
        see ``DiffusionConfig.diffusion_mask_prob``. ``use_conditioning=False``
        is OF3's trunk-conditioning drop.

        Returns:
            gt_augmented: Augmented ground truth [B?, S, N, M, 3].
            denoised: Denoised predictions [B?, S, N, M, 3].
            noise_levels: Sampled noise levels [B?, S].
        """
        cfg = self.cfg
        if cfg.skip_amp_sample_training:
            # Protenix-style ``skip_amp.sample_diffusion_training``: upcast
            # float inputs and run the training rollout with autocast off.
            gt_positions = _upcast_fp32(gt_positions)
            embeddings = _upcast_fp32(embeddings)
            ref_pos = _upcast_fp32(ref_pos)
            ref_charge = _upcast_fp32(ref_charge)
            seq_mask = _upcast_fp32(seq_mask)
            rel_feat = _upcast_fp32(rel_feat)
            amp_ctx = torch.amp.autocast(device_type=gt_positions.device.type, enabled=False)
        else:
            amp_ctx = nullcontext()
        with amp_ctx:
            return self._sample_training_impl(
                gt_positions,
                gt_mask,
                embeddings,
                ref_pos,
                ref_mask,
                ref_element,
                ref_charge,
                ref_atom_name_chars,
                seq_mask,
                rel_feat,
                num_samples,
                static,
                ref_space_uid=ref_space_uid,
                clean_mask=clean_mask,
                use_conditioning=use_conditioning,
            )

    def _sample_training_impl(
        self,
        gt_positions: torch.Tensor,
        gt_mask: torch.Tensor,
        embeddings: dict[str, torch.Tensor],
        ref_pos: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_element: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        seq_mask: torch.Tensor,
        rel_feat: torch.Tensor,
        num_samples: int,
        static: DiffusionStatic | None,
        ref_space_uid: torch.Tensor | None = None,  # [B?, N, M] int64
        clean_mask: torch.Tensor | None = None,  # [B?, S, N] bool
        use_conditioning: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        batch_shape = gt_positions.shape[:-3]  # () or (B,)
        device = gt_positions.device

        # Augment ground truth: expand to S copies, augment each independently
        gt_aug = gt_positions.unsqueeze(-4).expand(*batch_shape, num_samples, *gt_positions.shape[-3:]).clone()
        _augment_in_place(gt_aug, gt_mask.bool(), generator=None)

        # Sample noise levels. Base: log-normal ln σ = ln σ_data + p_mean + p_std·z.
        # Optional (sigma_mix_alpha > 0): with prob α draw from a high-σ component
        # instead — either a log-uniform over [σ_data, σ_data·smax] (simplest
        # ablation) or a high-σ log-normal bump — to train the docking / global-
        # arrangement regime the base under-covers.
        shape = (*batch_shape, num_samples)
        ln_sd = math.log(cfg.sigma_data)
        log_base = ln_sd + cfg.p_mean + cfg.p_std * torch.randn(shape, device=device)
        if cfg.sigma_mix_alpha > 0.0:
            if cfg.sigma_mix_kind == "uniform":
                # log-uniform over [σ_data·sigma_mix_smin, σ_data·sigma_mix_smax],
                # i.e. bounded in s = σ/σ_data by the mixture's own smin/smax
                # (decoupled from the inference schedule); α=1.0 → pure sampler.
                ln_lo = math.log(cfg.sigma_data * cfg.sigma_mix_smin)
                ln_hi = math.log(cfg.sigma_data * cfg.sigma_mix_smax)
                log_mix = ln_lo + torch.rand(shape, device=device) * (ln_hi - ln_lo)
            else:  # "lognormal": targeted smooth high-σ bump
                log_mix = ln_sd + cfg.sigma_mix_p_mean + cfg.sigma_mix_p_std * torch.randn(shape, device=device)
            pick_mix = torch.rand(shape, device=device) < cfg.sigma_mix_alpha
            noise_levels = torch.where(pick_mix, log_mix, log_base).exp()  # [B?, S]
        else:
            noise_levels = log_base.exp()  # [B?, S]
        # Floor σ so the EDM weight 1/σ² + 1/σ_d² stays bounded — without this
        # the rare left-tail draw spikes grad_norm by orders of magnitude.
        if cfg.clamp_train_sigma:
            noise_levels = noise_levels.clamp(min=cfg.train_sigma_min)

        # Add noise
        nl = noise_levels
        while nl.dim() < gt_aug.dim():
            nl = nl.unsqueeze(-1)
        noisy_pos = gt_aug + torch.randn_like(gt_aug) * nl  # [B?, S, N, M, 3]

        # Masked / partial diffusion: hand the clean tokens' augmented GT back
        # to the denoiser instead of their noised copy (RePaint-style
        # replacement at the single global σ). Applied AFTER the per-sample
        # augmentation, so the anchor is in the same frame as its sample.
        if clean_mask is not None:
            cm = clean_mask
            while cm.dim() < gt_aug.dim():
                cm = cm.unsqueeze(-1)
            noisy_pos = torch.where(cm, gt_aug, noisy_pos)

        # Denoise.  When ``static`` is provided, conditioning + atom-encoder
        # cache is reused across all S samples (saved compute).  Otherwise
        # ``_denoise_samples`` builds the cache inline once per call.
        if static is not None:
            denoised = self._denoise_samples(noisy_pos, noise_levels, static=static)
        else:
            denoised = self._denoise_samples(
                noisy_pos,
                noise_levels,
                embeddings,
                ref_pos,
                ref_mask,
                ref_element,
                ref_charge,
                ref_atom_name_chars,
                ref_space_uid,
                seq_mask,
                rel_feat,
                use_conditioning=use_conditioning,
            )

        return gt_aug, denoised, noise_levels
