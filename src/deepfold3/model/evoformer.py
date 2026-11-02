"""Evoformer trunk: MSA stack + PairFormer stack.

Haiku scope hierarchy (separator '~/'):
    diffuser~/evoformer~/
        __layer_stack_no_per_layer/msa_stack/layer_{i}/
            msa_attention1~/        → row_attn
            outer_product_mean~/    → outer_product
            triangle_multiplication_outgoing~/  → tri_mul_out
            triangle_multiplication_incoming~/  → tri_mul_in
            pair_attention1~/       → pair_attn_row
            pair_attention2~/       → pair_attn_col
            pair_transition~/       → pair_transition
            msa_transition~/        → msa_transition
        __layer_stack_no_per_layer_1/trunk_pairformer/layer_{i}/
            triangle_multiplication_outgoing~/  → tri_mul_out
            triangle_multiplication_incoming~/  → tri_mul_in
            pair_attention1~/       → pair_attn_row
            pair_attention2~/       → pair_attn_col
            pair_transition~/       → pair_transition
            single_pair_logits_norm~/           → single_to_pair_norm (on pair act)
            single_pair_logits_projection~/     → single_to_pair
            single_attention_~/                 → single_attn
            single_transition~/                 → single_transition

Key dimension facts from AF3 weight file:
    left_single / right_single / single_activations / extra_msa_target_feat:
        input dim = 447  (= 63 raw target_feat + 384 evoformer_conditioning token_act)
    msa_activations:
        input dim = 34   (= 32 msa_1hot + 1 has_deletion + 1 deletion_value)
"""

from __future__ import annotations

import math
from functools import partial as _partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from deepfold3.model import kernel_census
from deepfold3.model.attention import GridSelfAttention, MSAAttention, SingleAttention
from deepfold3.model.config import EvoformerConfig, OptimConfig, PairFormerConfig, TemplateEmbeddingConfig
from deepfold3.model.ops import OuterProductMean, TriangleMulIncoming, TriangleMulOutgoing
from deepfold3.model.optim import make_transition_block
from deepfold3.model.primitives import (
    LayerNorm,
    LinearNoBias,
    block_residual_scale,
    map_row_chunks,
)
from deepfold3.model.template_embedding import TemplateEmbedding
from deepfold3.model.triton.dropout_add import dropout_add_columnwise, dropout_add_rowwise
from deepfold3.model.triton.msa_transition import msa_transition_residual_

_checkpoint = _partial(_torch_checkpoint, use_reentrant=False, determinism_check="none")

# Full target_feat dimension fed into the Evoformer (raw_feat + evo_cond_token_act)
_TARGET_FEAT_DIM = 447
# MSA feature dimension: one_hot(32) + has_deletion(1) + deletion_value(1)
_MSA_FEAT_DIM = 34


def _resolve_branch_residual_gain(layer_idx: int, base_gain: float | None, per_layer_extra: bool) -> float | None:
    """Resolve a per-block residual-gain factor for layer ``layer_idx``.

    Used by both the ``single_transition`` and ``pair_transition`` branches
    (and by the confidence-head PairFormer mirror) to multiply the
    block-wide ``s_ros`` by an additional factor.

    Returns ``None`` (= no override = 1.0) when neither knob is set,
    otherwise the composed scalar
    ``base_gain · (1/sqrt(layer_idx+1) if per_layer_extra else 1)``.
    """
    base = 1.0 if base_gain is None else float(base_gain)
    if per_layer_extra:
        base *= 1.0 / math.sqrt(float(layer_idx + 1))
    if not per_layer_extra and base_gain is None:
        return None
    return base


def _trimul_kwargs(cfg) -> dict:
    """MHR-TriMul kwargs from an Evoformer/PairFormer config."""
    return dict(
        num_heads=cfg.trimul_num_heads,
        head_norm=cfg.trimul_head_norm,
        router=cfg.trimul_router,
        router_kind=cfg.trimul_router_kind,
    )


# Backwards-compatible alias for older imports during the deepnet-removal
# refactor; remove once no callers reference the old name.
_resolve_single_transition_gain = _resolve_branch_residual_gain


class EvoformerBlock(nn.Module):
    """Single layer of the Evoformer MSA stack.

    Haiku: EvoformerIteration, name='msa_stack'
    """

    def __init__(
        self,
        c_msa: int,
        c_pair: int,
        num_msa_heads: int = 8,
        num_pair_heads: int = 4,
        outer_product_hidden: int = 32,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        optim: OptimConfig | None = None,
        residual_out_scale: float = 1.0,
        use_triangle_attention: bool = True,
        trimul_kwargs: dict | None = None,
    ):
        super().__init__()
        # ``residual_out_scale = s`` policy in this block — F-output side for every
        # residual update. Each F's output is multiplied by ``s`` in
        # ``forward`` before being added to the residual stream. Inputs (the
        # LayerNorm at the start of each F) stay unscaled — the absorption
        # target is the final linear projection of each F at inference time.
        self.row_attn = MSAAttention(c_msa, c_pair, num_heads=num_msa_heads, optim=optim)
        self.outer_product = OuterProductMean(c_msa, c_pair, c_hidden=outer_product_hidden)
        if optim is not None and optim.tiled_opm_chunk > 0:
            self.outer_product._chunk_size = optim.tiled_opm_chunk
        self.msa_row_chunk = optim.msa_row_chunk if optim is not None else 0
        self.outer_product._row_chunk = self.msa_row_chunk
        tk = trimul_kwargs or {}
        self.tri_mul_out = TriangleMulOutgoing(c_pair, optim=optim, **tk)
        self.tri_mul_in = TriangleMulIncoming(c_pair, optim=optim, **tk)
        self.use_triangle_attention = bool(use_triangle_attention)
        if self.use_triangle_attention:
            self.pair_attn_row = GridSelfAttention(c_pair, num_heads=num_pair_heads, transpose=False, optim=optim)
            self.pair_attn_col = GridSelfAttention(c_pair, num_heads=num_pair_heads, transpose=True, optim=optim)
        self.pair_transition = make_transition_block(c_pair, config=optim)
        self.msa_transition = make_transition_block(c_msa, config=optim)
        self.use_fused_msa_transition = optim.use_fused_msa_transition_inference if optim is not None else False
        self.msa_dropout = msa_dropout
        self.pair_dropout = pair_dropout
        self.use_fused_dropout = optim.use_fused_dropout_add if optim is not None else True
        # Per-block residual-output multiplier for OPM / TriMul / TriAttn / Transition.
        self.residual_out_scale = float(residual_out_scale)

    def forward(
        self,
        msa: torch.Tensor,  # [B?, S, N, c_msa]
        pair: torch.Tensor,  # [B?, N, N, c_pair]
        msa_mask: torch.Tensor,  # [B?, S, N]
        pair_mask: torch.Tensor,  # [B?, N, N]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p_pair, p_msa, training = self.pair_dropout, self.msa_dropout, self.training
        use_fused = self.use_fused_dropout
        s = self.residual_out_scale
        pair = pair + s * self.outer_product(msa, msa_mask)
        msa = dropout_add_rowwise(msa, s * self.row_attn(msa, msa_mask, pair), p_msa, training, use_fused)
        # ``msa`` is the fresh sum above, so the in-place kernel cannot clobber a caller's tensor.
        # The kernel materialises no intermediate at all; row chunks bound the fallback's.
        fused_msa = msa_transition_residual_(self.msa_transition, msa, s) if self.use_fused_msa_transition else None
        kernel_census.record("EvoformerBlock.msa_transition", "stock" if fused_msa is None else "fused")
        if fused_msa is not None:
            msa = fused_msa
        else:
            msa = msa + s * map_row_chunks(self.msa_transition, msa, self.msa_row_chunk)
        pair = dropout_add_rowwise(pair, s * self.tri_mul_out(pair, pair_mask), p_pair, training, use_fused)
        pair = dropout_add_rowwise(pair, s * self.tri_mul_in(pair, pair_mask), p_pair, training, use_fused)
        if self.use_triangle_attention:
            pair = dropout_add_rowwise(pair, s * self.pair_attn_row(pair, pair_mask), p_pair, training, use_fused)
            pair = dropout_add_columnwise(pair, s * self.pair_attn_col(pair, pair_mask), p_pair, training, use_fused)
        pair = pair + s * self.pair_transition(pair)
        return msa, pair


class PairFormerBlock(nn.Module):
    """Single layer of the PairFormer trunk (with single track).

    Haiku: PairFormerIteration with_single=True, name='trunk_pairformer'

    The pair_logits for single attention come from the pair representation
    (not from single), following AF3: single_pair_logits_projection(single_pair_logits_norm(pair)).
    """

    def __init__(
        self,
        c_pair: int,
        c_single: int,
        num_pair_heads: int = 4,
        num_single_heads: int = 16,
        single_key_dim: int = 384,
        single_value_dim: int = 384,
        pair_dropout: float = 0.25,
        optim: OptimConfig | None = None,
        residual_out_scale: float = 1.0,
        pair_bias_gate: bool = False,
        pair_bias_gate_init_tau: float = 0.25,
        single_transition_linear_out_init_gain: float | None = None,
        single_transition_residual_gain: float | None = None,
        pair_transition_residual_gain: float | None = None,
        branch_scaler: str = "identity",
        use_triangle_attention: bool = True,
        use_single_track: bool = True,
        trimul_kwargs: dict | None = None,
    ):
        super().__init__()
        # ``residual_out_scale`` (= s_ros) policy — F-output side only:
        #  * Every residual update (TriMul, GridSelfAttention pair-track,
        #    Transitions, SingleAttention) has F's output multiplied by
        #    ``s_ros`` in ``forward``. Inputs (LayerNorms inside each F)
        #    are left unscaled, so SingleAttention's q·kᵀ stays at
        #    baseline magnitude and its softmax distribution is unperturbed.
        #  * ``single_to_pair_norm`` (PairBias side input): untouched.
        #  * The per-head learnable PairBias gate ``τ_h = exp(a_h)`` inside
        #    SingleAttention is independent of ``residual_out_scale`` — it
        #    rescales ``pair_logits`` so the network can re-balance the
        #    QK-vs-PairBias contribution to the attention softmax per head.
        #  * ``{single,pair}_transition_residual_gain`` (defaults 1.0)
        #    multiplies ``s_ros`` on the corresponding transition branch
        #    only — used by the trunk transition aligned-gain ablation.
        tk = trimul_kwargs or {}
        self.tri_mul_out = TriangleMulOutgoing(c_pair, optim=optim, **tk)
        self.tri_mul_in = TriangleMulIncoming(c_pair, optim=optim, **tk)
        self.use_triangle_attention = bool(use_triangle_attention)
        if self.use_triangle_attention:
            self.pair_attn_row = GridSelfAttention(c_pair, num_heads=num_pair_heads, transpose=False, optim=optim)
            self.pair_attn_col = GridSelfAttention(c_pair, num_heads=num_pair_heads, transpose=True, optim=optim)
        self.pair_transition = make_transition_block(c_pair, config=optim)
        # The per-token track. ``single`` is a pure sink in this block — it reads
        # the pair representation through ``pair_logits`` and never writes back
        # — so deleting it removes no input from the pair track.
        self.use_single_track = bool(use_single_track)
        if self.use_single_track:
            self.single_to_pair_norm = LayerNorm(c_pair)
            self.single_to_pair = LinearNoBias(c_pair, num_single_heads)  # pair → per-head logits
            self.single_attn = SingleAttention(
                c_single,
                num_heads=num_single_heads,
                key_dim=single_key_dim,
                value_dim=single_value_dim,
                optim=optim,
                pair_bias_gate=pair_bias_gate,
                pair_bias_gate_init_tau=pair_bias_gate_init_tau,
            )
            self.single_transition = make_transition_block(
                c_single, config=optim, linear_out_init_gain=single_transition_linear_out_init_gain
            )
        self.residual_out_scale = float(residual_out_scale)
        # Branch-specific multipliers on the per-transition residual updates
        # (None ≡ 1.0). Compose with the block-wide ``s_ros`` to give the
        # final ``s_single_ffn`` / ``s_pair_ffn``.
        self.single_transition_residual_gain = (
            float(single_transition_residual_gain) if single_transition_residual_gain is not None else 1.0
        )
        self.pair_transition_residual_gain = (
            float(pair_transition_residual_gain) if pair_transition_residual_gain is not None else 1.0
        )
        # Function ``s(F(x))`` applied to every residual-branch output before
        # the per-block γ. See :mod:`deepfold3.model.branch_scaler`.
        from deepfold3.model.branch_scaler import make_branch_scaler

        self.branch_scaler_kind = str(branch_scaler)
        self.branch_scaler = make_branch_scaler(self.branch_scaler_kind)
        self.pair_dropout = pair_dropout
        self.use_fused_dropout = optim.use_fused_dropout_add if optim is not None else True

    def forward(
        self,
        pair: torch.Tensor,  # [B?, N, N, c_pair]
        single: torch.Tensor,  # [B?, N, c_single]
        pair_mask: torch.Tensor,  # [B?, N, N]
        seq_mask: torch.Tensor,  # [B?, N]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p, training = self.pair_dropout, self.training
        use_fused = self.use_fused_dropout
        # All branches multiply F's output by the block-wide ``s_ros``.
        # The two transition branches additionally multiply by their
        # branch-specific residual gain.
        s_ros = self.residual_out_scale
        s_tri = s_ros
        s_pa = s_ros
        s_pf = s_ros * self.pair_transition_residual_gain
        s_single_ffn = s_ros * self.single_transition_residual_gain
        s_single_attn = s_ros
        # Branch-output scaler applied *before* the per-block ``s`` multiplier.
        # When non-identity, the per-branch residual gain knobs must be at
        # default (validated in ``PairFormerConfig.__post_init__``), so
        # ``s_pf == s_ros`` and ``s_single_ffn == s_ros``.
        rmsn = self.branch_scaler
        # Capture each residual-stream branch contribution as a named
        # intermediate so the activation meter can record per-branch RMS
        # / ρ_l. ``_act_meter`` is set by
        # :func:`activation_meter.attach_activation_meter`; the per-branch
        # plumbing is a no-op when the meter is absent or disabled.
        b_tri_out = s_tri * rmsn(self.tri_mul_out(pair, pair_mask))
        pair = dropout_add_rowwise(pair, b_tri_out, p, training, use_fused)
        b_tri_in = s_tri * rmsn(self.tri_mul_in(pair, pair_mask))
        pair = dropout_add_rowwise(pair, b_tri_in, p, training, use_fused)
        if self.use_triangle_attention:
            b_pair_attn_row = s_pa * rmsn(self.pair_attn_row(pair, pair_mask))
            pair = dropout_add_rowwise(pair, b_pair_attn_row, p, training, use_fused)
            b_pair_attn_col = s_pa * rmsn(self.pair_attn_col(pair, pair_mask))
            pair = dropout_add_columnwise(pair, b_pair_attn_col, p, training, use_fused)
        else:
            # Meter the deleted branches as exactly zero rather than dropping the
            # rows — ρ_l stays comparable against a run that has them.
            b_pair_attn_row = b_pair_attn_col = pair.new_zeros(())
        b_pair_ffn = s_pf * rmsn(self.pair_transition(pair))
        pair = pair + b_pair_ffn
        # Single update: pair → logits, then single attention.
        # PairBias is fed in unrescaled — under LNS, ``q·kᵀ ∝ s²`` softens
        # the QK contribution to the softmax, and we accept the resulting
        # PairBias-leaning attention rather than introducing a depth- or
        # head-dependent calibrator.
        # ``single_to_pair`` is *not* a residual-stream branch (its output is
        # consumed as ``pair_logits`` for the SingleAttention PairBias path);
        # leave it un-RMSNormed.
        if self.use_single_track:
            pair_logits = self.single_to_pair(self.single_to_pair_norm(pair))  # [..., N, N, H]
            pair_logits = pair_logits.movedim(-1, -3)  # [..., H, N, N]
            b_single_attn = s_single_attn * rmsn(self.single_attn(single, seq_mask, pair_logits))
            single = single + b_single_attn
            u_single = self.single_transition(single)
            probe = getattr(self, "_probe", None)
            if probe is not None and probe.enabled:
                probe.record_block(self._probe_layer_idx, single, u_single, s_single_ffn)
            b_single_ffn = s_single_ffn * rmsn(u_single)
        else:
            # Same convention as the deleted triangle-attention branches above:
            # meter them as exactly zero rather than dropping the rows, so ρ_l
            # stays comparable against a run that has them.
            b_single_attn = b_single_ffn = pair.new_zeros(())
        act_meter = getattr(self, "_act_meter", None)
        if act_meter is not None and getattr(act_meter, "enabled", False):
            act_meter.record_all_branches(
                self._act_layer_idx,
                b_tri_out=b_tri_out,
                b_tri_in=b_tri_in,
                b_pair_attn_row=b_pair_attn_row,
                b_pair_attn_col=b_pair_attn_col,
                b_pair_ffn=b_pair_ffn,
                b_single_attn=b_single_attn,
                b_single_ffn=b_single_ffn,
            )
        if self.use_single_track:
            single = single + b_single_ffn
        return pair, single


class Evoformer(nn.Module):
    """Full Evoformer trunk: embedders + MSA stack + PairFormer stack.

    Handles recycled pair/single embeddings and the full Evoformer + PairFormer stacks.

    Haiku scope: diffuser~/evoformer~/

    Dimensions (from AF3 weight file):
        target_feat input: 447-dim (raw 63 + evo conditioning 384)
        msa_feat input: 34-dim (one_hot(32) + has_deletion + deletion_value)
    """

    def __init__(
        self,
        cfg: EvoformerConfig,
        pf_cfg: PairFormerConfig,
        template_cfg: TemplateEmbeddingConfig | None = None,
        optim: OptimConfig | None = None,
    ):
        super().__init__()
        self._grad_checkpoint = bool(optim and optim.use_gradient_checkpointing)
        c_msa = cfg.msa_channel
        c_pair = cfg.pair_channel
        c_single = cfg.seq_channel

        # Template embedding (optional — None if not using templates)
        if template_cfg is not None:
            self.template_embedding: TemplateEmbedding | None = TemplateEmbedding(template_cfg, optim=optim)
        else:
            self.template_embedding = None

        # Input embeddings for single and MSA — dimensions from AF3 weight file
        self.msa_activations = LinearNoBias(_MSA_FEAT_DIM, c_msa)
        self.extra_msa_target_feat = LinearNoBias(_TARGET_FEAT_DIM, c_msa)

        # Pair embeddings from sequence (target_feat = 447-dim)
        self.left_single = LinearNoBias(_TARGET_FEAT_DIM, c_pair)
        self.right_single = LinearNoBias(_TARGET_FEAT_DIM, c_pair)

        # Bond embedding
        self.bond_embedding = LinearNoBias(1, c_pair)

        # Previous embedding projections (recycling)
        # AF3: prev_embedding and prev_single_embedding use final_init (zeros)
        self.prev_embedding_layer_norm = LayerNorm(c_pair)
        self.prev_embedding = LinearNoBias(c_pair, c_pair, initializer="zeros")
        # Recycling the single track only means something if something updates it.
        # With ``use_single_track`` off the trunk returns ``s_init`` unchanged,
        # so ``s = s_init + P(LN(s_prev))`` degenerates into a fixed-point
        # iteration on ``s_init`` — a pointwise map dressed up as recycling.
        # Skip it, and the modules with it.
        self.use_single_track = bool(pf_cfg.use_single_track)
        if self.use_single_track:
            self.prev_single_embedding_layer_norm = LayerNorm(c_single)
            self.prev_single_embedding = LinearNoBias(c_single, c_single, initializer="zeros")

        # Position activations for relative encoding
        # rel_feat_dim = rel_pos(66) + rel_token(66) + entity_same(1) + rel_chain(6) = 139
        rel_feat_dim = 2 * (2 * cfg.max_relative_idx + 2) + 1 + (2 * cfg.max_relative_chain + 2)
        self.position_activations = LinearNoBias(rel_feat_dim, c_pair)

        # Single activations from target feat (447-dim input).
        #
        # This survives ``use_single_track=False``, and it is worth saying why:
        # the diffusion *atom* encoder conditions its per-atom activations on
        # the trunk single (``AtomEncoder.compute_static(trunk_single=...)``),
        # and that is its only token-level identity channel — it receives no
        # ``target_feat`` of its own. So s_init keeps a live consumer and a
        # gradient path even when the trunk stops updating it and the diffusion
        # conditioning stops concatenating it.
        self.single_activations = LinearNoBias(_TARGET_FEAT_DIM, c_single)

        # Input embeddings use plain per-Linear Lecun init (OF3/AF3 stock): each
        # LinearNoBias above keeps its default ``"linear"`` initializer. The
        # previous variance-matched sum-fusion override was removed in favour of
        # OF3's per-Linear init.

        # MSA stack
        self.msa_stack = nn.ModuleList(
            [
                EvoformerBlock(
                    c_msa,
                    c_pair,
                    num_msa_heads=cfg.msa_attention.num_head,
                    num_pair_heads=cfg.pair_attention.num_head,
                    outer_product_hidden=cfg.outer_product_num_channel,
                    msa_dropout=cfg.msa_dropout,
                    pair_dropout=cfg.pair_dropout,
                    optim=optim,
                    residual_out_scale=block_residual_scale(
                        i, cfg.num_layer, cfg.residual_output_scaling, cfg.uniform_residual_scale
                    ),
                    use_triangle_attention=cfg.use_triangle_attention,
                    trimul_kwargs=_trimul_kwargs(cfg),
                )
                for i in range(cfg.num_layer)
            ]
        )

        # PairFormer stack
        self.trunk_pairformer = nn.ModuleList(
            [
                PairFormerBlock(
                    c_pair,
                    c_single,
                    num_pair_heads=pf_cfg.pair_attention.num_head,
                    num_single_heads=pf_cfg.single_attention_num_head,
                    single_key_dim=pf_cfg.single_key_dim,
                    single_value_dim=pf_cfg.single_value_dim,
                    pair_dropout=pf_cfg.pair_dropout,
                    optim=optim,
                    residual_out_scale=block_residual_scale(
                        i, pf_cfg.num_layer, pf_cfg.residual_output_scaling, pf_cfg.uniform_residual_scale
                    ),
                    pair_bias_gate=pf_cfg.pair_bias_gate,
                    pair_bias_gate_init_tau=pf_cfg.pair_bias_gate_init_tau,
                    single_transition_linear_out_init_gain=pf_cfg.single_transition_linear_out_init_gain,
                    single_transition_residual_gain=_resolve_branch_residual_gain(
                        i,
                        pf_cfg.single_transition_residual_gain,
                        pf_cfg.single_transition_residual_per_layer_extra,
                    ),
                    pair_transition_residual_gain=_resolve_branch_residual_gain(
                        i,
                        pf_cfg.pair_transition_residual_gain,
                        pf_cfg.pair_transition_residual_per_layer_extra,
                    ),
                    branch_scaler=pf_cfg.branch_scaler,
                    use_triangle_attention=pf_cfg.use_triangle_attention,
                    use_single_track=pf_cfg.use_single_track,
                    trimul_kwargs=_trimul_kwargs(pf_cfg),
                )
                for i in range(pf_cfg.num_layer)
            ]
        )

    def forward(
        self,
        target_feat: torch.Tensor,  # [B?, N, 447]
        msa_feat: torch.Tensor,  # [B?, S, N, 34]
        rel_feat: torch.Tensor,  # [B?, N, N, rel_feat_dim]
        pair_mask: torch.Tensor,  # [B?, N, N]
        seq_mask: torch.Tensor,  # [B?, N]
        msa_mask: torch.Tensor,  # [B?, S, N]
        prev_pair: torch.Tensor,  # [B?, N, N, c_pair]
        prev_single: torch.Tensor,  # [B?, N, c_single]
        bond_matrix: torch.Tensor | None = None,  # [B?, N, N, 1]
        template_aatype: torch.Tensor | None = None,  # [T, N]
        template_pseudo_beta: torch.Tensor | None = None,  # [T, N, 3]
        template_pseudo_beta_mask: torch.Tensor | None = None,  # [T, N]
        template_backbone_frame: torch.Tensor | None = None,  # [T, N, 3, 3]
        template_backbone_mask: torch.Tensor | None = None,  # [T, N]
        multichain_mask: torch.Tensor | None = None,  # [N, N]
        template_ca: torch.Tensor | None = None,  # [T, N, 3]
        template_mask: torch.Tensor | None = None,  # [B?, T] real-vs-pad slot
    ) -> dict[str, torch.Tensor]:
        # Build initial pair activations (negative indices work for both batched/unbatched)
        left = self.left_single(target_feat).unsqueeze(-2)  # [..., N, 1, c_pair]
        right = self.right_single(target_feat).unsqueeze(-3)  # [..., 1, N, c_pair]
        pair = left + right  # [..., N, N, c_pair]

        # Add recycling pair
        pair = pair + self.prev_embedding(self.prev_embedding_layer_norm(prev_pair))

        # Add relative position encoding
        pair = pair + self.position_activations(rel_feat)

        # Add bond embedding
        if bond_matrix is not None:
            pair = pair + self.bond_embedding(bond_matrix)

        # Template embedding (after bonds, before MSA stack — matches AF3 pipeline).
        # ``TemplateEmbedding`` and ``SingleTemplateEmbedding`` natively support
        # batched ``[B?, N, ...]`` inputs (template axis is the dim before N).
        if self.template_embedding is not None and template_aatype is not None:
            t_axis = 1 if template_aatype.dim() == 3 else 0  # [B,T,N] vs [T,N]
            if template_aatype.shape[t_axis] > 0:
                if multichain_mask is None:
                    N = pair.shape[-2]
                    if pair.dim() == 4:
                        multichain_mask = torch.ones(pair.shape[0], N, N, device=pair.device, dtype=pair.dtype)
                    else:
                        multichain_mask = torch.ones(N, N, device=pair.device, dtype=pair.dtype)
                pair = pair + self.template_embedding(
                    query_pair=pair,
                    template_aatype=template_aatype,
                    template_pseudo_beta=template_pseudo_beta,
                    template_pseudo_beta_mask=template_pseudo_beta_mask,
                    template_backbone_frame=template_backbone_frame,
                    template_backbone_mask=template_backbone_mask,
                    pair_mask=pair_mask,
                    multichain_mask=multichain_mask,
                    template_ca=template_ca,
                    template_mask=template_mask,
                )

        # Initial single activations
        single = self.single_activations(target_feat)

        # Add recycling single
        if self.use_single_track:
            single = single + self.prev_single_embedding(self.prev_single_embedding_layer_norm(prev_single))

        # MSA embedding: msa_feat [..., S, N, 34] → [..., S, N, c_msa]
        msa_act = self.msa_activations(msa_feat)
        msa_act = msa_act + self.extra_msa_target_feat(target_feat).unsqueeze(-3)

        # MSA stack (optionally checkpointed per block during training).
        use_ckpt = self._grad_checkpoint and torch.is_grad_enabled()
        for layer in self.msa_stack:
            if use_ckpt:
                msa_act, pair = _checkpoint(layer, msa_act, pair, msa_mask, pair_mask)
            else:
                msa_act, pair = layer(msa_act, pair, msa_mask, pair_mask)

        # PairFormer stack (optionally checkpointed per block during training).
        # ``_act_meter`` (set by :func:`activation_meter.attach_activation_meter`)
        # captures single/pair RMS at the PF input plus each layer output, and
        # the per-layer branch contribution ``Δ = post - pre`` for ρ_l. No-op
        # when ``_act_meter`` is ``None`` or ``meter.enabled`` is ``False``.
        # Pre-block references (``single_pre``/``pair_pre``) are just Python
        # rebinds — the original tensors are already saved by the block as its
        # backward inputs, so keeping a reference adds zero memory.
        act_meter = getattr(self, "_act_meter", None)
        # With no single representation there is nothing to meter on that axis;
        # the pair columns stay live. Note that a *flat* ``single_l2`` in such a
        # run means the quantity is absent, not that the track settled.
        if act_meter is not None:
            act_meter.record(0, single, pair)
        for i, layer in enumerate(self.trunk_pairformer):
            single_pre, pair_pre = single, pair
            if use_ckpt:
                pair, single = _checkpoint(layer, pair, single, pair_mask, seq_mask)
            else:
                pair, single = layer(pair, single, pair_mask, seq_mask)
            if act_meter is not None:
                act_meter.record(i + 1, single, pair)
                act_meter.record_branch(i, single - single_pre, pair - pair_pre)

        return {
            "pair": pair,
            "single": single,
            "msa": msa_act,  # final MSA-stack representation [..., S, N, c_msa] (masked-MSA head)
            "target_feat": target_feat,
        }
