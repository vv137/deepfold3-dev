"""Template embedding: structural templates → pair-space embedding.

Processes T structural templates independently, averages them, and projects
to pair_channel dimensions.  Integrated into the Evoformer trunk between
bond embedding and MSA stack.

Haiku scope hierarchy:
    diffuser/evoformer/template_embedding/
        output_linear                         → output_proj
        single_template_embedding/
            query_embedding_norm              → single_template_emb.query_norm
            template_pair_embedding_0         → single_template_emb.embed_dgram
            template_pair_embedding_1         → single_template_emb.embed_pseudo_beta
            template_pair_embedding_2         → single_template_emb.embed_aatype_row
            template_pair_embedding_3         → single_template_emb.embed_aatype_col
            template_pair_embedding_4         → single_template_emb.embed_unit_vec_x
            template_pair_embedding_5         → single_template_emb.embed_unit_vec_y
            template_pair_embedding_6         → single_template_emb.embed_unit_vec_z
            template_pair_embedding_7         → single_template_emb.embed_backbone_mask
            template_pair_embedding_8         → single_template_emb.embed_query
            output_layer_norm                 → single_template_emb.output_norm
            __layer_stack_no_per_layer/template_embedding_iteration/
                layer {i}:
                    triangle_multiplication_outgoing~/  → pairformer.{i}.tri_mul_out
                    triangle_multiplication_incoming~/  → pairformer.{i}.tri_mul_in
                    pair_attention1~/                   → pairformer.{i}.pair_attn_row
                    pair_attention2~/                   → pairformer.{i}.pair_attn_col
                    pair_transition~/                   → pairformer.{i}.pair_transition

Dense atom indices (from deepfold3.data.const):
    N=0, CA=1, C=2, CB=3
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from deepfold3.model.attention import GridSelfAttention
from deepfold3.model.config import OptimConfig, TemplateEmbeddingConfig
from deepfold3.model.embedders import dgram_from_positions as _dgram_from_positions
from deepfold3.model.ops import TriangleMulIncoming, TriangleMulOutgoing
from deepfold3.model.optim import make_transition_block
from deepfold3.model.primitives import (
    LayerNorm,
    LinearNoBias,
    block_residual_scale,
    map_pair_rows,
    map_pair_transition,
)
from deepfold3.model.triton.dropout_add import dropout_add_columnwise, dropout_add_rowwise

_checkpoint = partial(_torch_checkpoint, use_reentrant=False, determinism_check="none")

# Dense atom indices for backbone atoms
_CA_IDX = 1
_CB_IDX = 3

# GLY residue type index (uses CA instead of CB for pseudo-beta)
_GLY_IDX = 7


def pseudo_beta_from_positions(
    aatype: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pseudo-beta positions and mask.

    Pseudo-beta is CB for all residues except GLY (which uses CA).

    Args:
        aatype: [N] residue type indices.
        positions: [N, num_atoms, 3] dense atom positions.
        mask: [N, num_atoms] dense atom mask.

    Returns:
        Tuple of (pseudo_beta_pos [N, 3], pseudo_beta_mask [N]).
    """
    is_gly = aatype == _GLY_IDX  # [N]
    # Select CB for non-GLY, CA for GLY
    cb_pos = positions[:, _CB_IDX]  # [N, 3]
    ca_pos = positions[:, _CA_IDX]  # [N, 3]
    pb_pos = torch.where(is_gly.unsqueeze(-1), ca_pos, cb_pos)  # [N, 3]

    cb_mask = mask[:, _CB_IDX]  # [N]
    ca_mask = mask[:, _CA_IDX]  # [N]
    pb_mask = torch.where(is_gly, ca_mask, cb_mask)  # [N]

    return pb_pos, pb_mask


def dgram_from_positions_masked(
    positions: torch.Tensor,
    mask: torch.Tensor,
    min_bin: float = 3.25,
    max_bin: float = 50.75,
    num_bins: int = 39,
) -> torch.Tensor:
    """Compute masked pairwise distance histogram from positions.

    Thin wrapper around :func:`~deepfold3.model.confidence.dgram_from_positions`
    that additionally zeros out bins where either position is missing.

    Args:
        positions: [N, 3] atom positions.
        mask: [N] atom mask.
        min_bin: Left edge of first distance bin (Angstroms).
        max_bin: Left edge of last distance bin (Angstroms).
        num_bins: Number of histogram bins.

    Returns:
        Distance histogram [N, N, num_bins] float32.
    """
    dgram = _dgram_from_positions(positions, min_bin=min_bin, max_bin=max_bin, num_bins=num_bins)
    pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # [N, N]
    return dgram * pair_mask.unsqueeze(-1)


def _unit_vectors_from_frames(
    frames: torch.Tensor,
    translations: torch.Tensor,
    frame_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute inter-residue unit vectors in each residue's local frame.

    For each pair (i, j), computes the displacement vector from i's CA to j's CA
    in i's local coordinate frame, then normalizes.

    Args:
        frames: [N, 3, 3] rotation matrices (rows = local frame axes).
        translations: [N, 3] CA positions (frame origins).
        frame_mask: [N] mask for valid frames.

    Returns:
        Tuple of (unit_vec_x, unit_vec_y, unit_vec_z, backbone_mask_2d)
        each [N, N] or [N, N, 1].

    Frame transform + normalisation runs in fp32 (AF3 geometry precision
    convention, matches ``express_coordinates_in_frame``). The featurizer
    produces orthonormal frames, but under bf16 autocast the einsum and the
    subsequent L2 normalise degrade near small displacements.
    """
    eps = 1e-8
    out_dtype = translations.dtype
    frames_f = frames.float()
    translations_f = translations.float()

    # Displacement vectors: j_pos - i_pos → [..., N, N, 3]
    disp = translations_f.unsqueeze(-2) - translations_f.unsqueeze(-3)

    # Transform to local frame of residue i: frames[i] @ disp[i, j]
    # frames: [..., N, 3, 3], disp: [..., N, N, 3]
    # For each i: local_vec[i,j] = frames[i] @ disp[i,j]
    # Ellipsis carries optional leading batch dims through unchanged.
    local_vec = torch.einsum("...iae,...ije->...ija", frames_f, disp)

    # Normalize to unit vectors
    norm = local_vec.norm(dim=-1, keepdim=True).clamp(min=eps)
    unit_vec = local_vec / norm  # [..., N, N, 3]

    # Mask: both residues need valid backbone frames
    backbone_mask_2d = frame_mask.unsqueeze(-1) * frame_mask.unsqueeze(-2)  # [..., N, N]

    # Apply mask, cast back to ambient dtype
    unit_vec = (unit_vec * backbone_mask_2d.unsqueeze(-1).float()).to(out_dtype)

    return (
        unit_vec[..., 0:1],  # x component [N, N, 1]
        unit_vec[..., 1:2],  # y component [N, N, 1]
        unit_vec[..., 2:3],  # z component [N, N, 1]
        backbone_mask_2d,
    )


class TemplatePairFormerBlock(nn.Module):
    """Simplified PairFormer block for template embedding (no single track).

    Contains triangle multiplications, grid self-attention (row + column),
    and a transition block with a smaller intermediate factor.

    Haiku scope: __layer_stack_no_per_layer/template_embedding_iteration/

    Attribute map:
        triangle_multiplication_outgoing~/  → tri_mul_out
        triangle_multiplication_incoming~/  → tri_mul_in
        pair_attention1~/                   → pair_attn_row
        pair_attention2~/                   → pair_attn_col
        pair_transition~/                   → pair_transition
    """

    def __init__(
        self,
        num_channels: int,
        num_heads: int = 4,
        num_intermediate_factor: int = 2,
        pair_dropout: float = 0.25,
        optim: OptimConfig | None = None,
        residual_out_scale: float = 1.0,
    ):
        super().__init__()
        # Pair-only template stack: every residual update is output-scaled by
        # ``residual_out_scale`` in ``forward``. No LNS-path module here (no
        # SingleAttention/MSAAttention).
        self.tri_mul_out = TriangleMulOutgoing(num_channels, optim=optim)
        self.tri_mul_in = TriangleMulIncoming(num_channels, optim=optim)
        self.pair_attn_row = GridSelfAttention(num_channels, num_heads=num_heads, transpose=False, optim=optim)
        self.pair_attn_col = GridSelfAttention(num_channels, num_heads=num_heads, transpose=True, optim=optim)
        self.pair_transition = make_transition_block(num_channels, num_intermediate_factor, config=optim)
        self.num_intermediate_factor = num_intermediate_factor
        self.pair_chunk_budget = optim.pair_chunk_budget_bytes if optim is not None else 0
        self.pair_dropout = pair_dropout
        self.use_fused_dropout = optim.use_fused_dropout_add if optim is not None else True
        self.residual_out_scale = float(residual_out_scale)

    def forward(self, pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through one template PairFormer block.

        Args:
            pair: [N, N, num_channels] pair activations.
            pair_mask: [N, N] pair mask.

        Returns:
            Updated pair activations [N, N, num_channels].
        """
        p, training = self.pair_dropout, self.training
        use_fused = self.use_fused_dropout
        s = self.residual_out_scale
        pair = dropout_add_rowwise(pair, s * self.tri_mul_out(pair, pair_mask), p, training, use_fused)
        pair = dropout_add_rowwise(pair, s * self.tri_mul_in(pair, pair_mask), p, training, use_fused)
        pair = dropout_add_rowwise(pair, s * self.pair_attn_row(pair, pair_mask), p, training, use_fused)
        pair = dropout_add_columnwise(pair, s * self.pair_attn_col(pair, pair_mask), p, training, use_fused)
        pair = pair + s * map_pair_transition(
            self.pair_transition, pair, self.pair_chunk_budget, self.num_intermediate_factor
        )
        return pair


class SingleTemplateEmbedding(nn.Module):
    """Embed a single structural template into pair-space features.

    Constructs 9 input features from template structure data, projects each
    to num_channels via LinearNoBias, sums them, then refines through a
    short PairFormer stack.

    Haiku scope: single_template_embedding/

    Attribute map:
        query_embedding_norm              → query_norm
        template_pair_embedding_0         → embed_dgram
        template_pair_embedding_1         → embed_pseudo_beta
        template_pair_embedding_2         → embed_aatype_row
        template_pair_embedding_3         → embed_aatype_col
        template_pair_embedding_4         → embed_unit_vec_x
        template_pair_embedding_5         → embed_unit_vec_y
        template_pair_embedding_6         → embed_unit_vec_z
        template_pair_embedding_7         → embed_backbone_mask
        template_pair_embedding_8         → embed_query
        output_layer_norm                 → output_norm
        __layer_stack_no_per_layer/...    → pairformer.{i}
    """

    def __init__(self, cfg: TemplateEmbeddingConfig, optim: OptimConfig | None = None):
        super().__init__()
        c = cfg.num_channels
        pc = cfg.pair_channel
        # Per-block activation checkpointing, as the trunk stacks do.
        self._grad_checkpoint = bool(optim and optim.use_gradient_checkpointing)
        self.pair_chunk_budget = optim.pair_chunk_budget_bytes if optim is not None else 0

        # 9 input feature projections (AF3: template_pair_embedding_* uses initializer='relu')
        self.embed_dgram = LinearNoBias(cfg.dgram_num_bins, c, initializer="relu")
        self.embed_pseudo_beta = LinearNoBias(1, c, initializer="relu")
        self.embed_aatype_row = LinearNoBias(cfg.aatype_num_classes, c, initializer="relu")
        self.embed_aatype_col = LinearNoBias(cfg.aatype_num_classes, c, initializer="relu")
        self.embed_unit_vec_x = LinearNoBias(1, c, initializer="relu")
        self.embed_unit_vec_y = LinearNoBias(1, c, initializer="relu")
        self.embed_unit_vec_z = LinearNoBias(1, c, initializer="relu")
        self.embed_backbone_mask = LinearNoBias(1, c, initializer="relu")
        self.embed_query = LinearNoBias(pc, c, initializer="relu")

        # LayerNorm on query embedding before projection
        self.query_norm = LayerNorm(pc)

        # Feature projections keep plain per-Linear He (relu) init, matching
        # AF3's ``template_pair_embedding_*`` (initializer='relu'). The previous
        # 9-way variance-matched sum-fusion override was removed in favour of
        # OF3/AF3 stock per-Linear init.

        # PairFormer stack
        pf = cfg.pairformer
        self.pairformer = nn.ModuleList(
            [
                TemplatePairFormerBlock(
                    num_channels=pf.num_channels,
                    num_heads=pf.num_heads,
                    num_intermediate_factor=pf.num_intermediate_factor,
                    pair_dropout=pf.pair_dropout,
                    optim=optim,
                    residual_out_scale=block_residual_scale(
                        i, pf.num_layer, pf.residual_output_scaling, pf.uniform_residual_scale
                    ),
                )
                for i in range(pf.num_layer)
            ]
        )

        # Output LayerNorm
        self.output_norm = LayerNorm(c)

    def forward(
        self,
        query_pair: torch.Tensor,
        template_aatype: torch.Tensor,
        template_pseudo_beta: torch.Tensor,
        template_pseudo_beta_mask: torch.Tensor,
        template_backbone_frame: torch.Tensor,
        template_backbone_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        multichain_mask: torch.Tensor | None = None,
        template_ca: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed a single template.

        Supports both unbatched (``[N, ...]``) and batched (``[B, N, ...]``)
        inputs. The downstream PairFormer-style ops (triangle multiply, grid
        attention, transition) are already batch-aware via the trunk
        primitives, and ``_unit_vectors_from_frames`` uses ellipsis-style
        einsum, so the only changes for batch support are:
          * use ``shape[-3]`` (negative indexing) for ``N``
          * broadcast ``aatype_1hot`` along an explicit ``unsqueeze`` /
            ``expand`` that doesn't assume rank-2 inputs

        Args:
            query_pair: [B?, N, N, pair_channel] query pair embedding.
            template_aatype: [B?, N] template residue types (int64).
            template_pseudo_beta: [B?, N, 3] pseudo-beta positions.
            template_pseudo_beta_mask: [B?, N] pseudo-beta mask.
            template_backbone_frame: [B?, N, 3, 3] backbone rotation matrices.
            template_backbone_mask: [B?, N] backbone frame mask.
            pair_mask: [B?, N, N] padding mask for attention.
            multichain_mask: [B?, N, N] intra-chain mask (1 for same-chain
                pairs). When provided, restricts template geometry to
                block-diagonal (intra-chain) pairs — per-chain templates carry
                no valid cross-chain frame, so their off-diagonal distances /
                orientations would be garbage.

        Returns:
            Template pair embedding [B?, N, N, num_channels].
        """
        dtype = query_pair.dtype
        N = query_pair.shape[-3]

        # Pseudo-beta pairwise distance histogram (already batch-friendly via
        # ellipsis-style ops in dgram_from_positions).
        dgram = dgram_from_positions_masked(
            template_pseudo_beta,
            template_pseudo_beta_mask,
            min_bin=3.25,
            max_bin=50.75,
            num_bins=self.embed_dgram.in_features,
        )  # [B?, N, N, dgram_bins]
        # Per-template pair mask: 1 where both tokens are covered by this template.
        # Restrict to intra-chain (block-diagonal) pairs via multichain_mask:
        # per-chain templates are placed in each chain's own (different) frame, so
        # their cross-chain distances / orientations are not real interface
        # geometry — mask the off-diagonal so only intra-chain template signal
        # flows. (A whole-complex template would need slot=hit + a common frame
        # to legitimately carry cross-chain features; that is not built here.)
        pb_mask_2d = template_pseudo_beta_mask.unsqueeze(-1) * template_pseudo_beta_mask.unsqueeze(-2)
        if multichain_mask is not None:
            pb_mask_2d = pb_mask_2d * multichain_mask
        dgram = dgram * pb_mask_2d.unsqueeze(-1)

        # Unit vectors from backbone frames (ellipsis einsum, batch-friendly).
        # AF3 §2.8: the template unit-vector feature is the displacement of the
        # CA atom (not the CB pseudo-beta) in each residue's local frame. Use
        # the CA positions carried by ``template_ca``; fall back to pseudo-beta
        # only for legacy callers that don't supply it.
        unit_vec_points = template_ca if template_ca is not None else template_pseudo_beta
        unit_x, unit_y, unit_z, bb_mask_2d = _unit_vectors_from_frames(
            template_backbone_frame,
            unit_vec_points,
            template_backbone_mask,
        )
        if multichain_mask is not None:
            bb_mask_2d = bb_mask_2d * multichain_mask

        # Apply backbone mask to unit vectors
        unit_x = unit_x * bb_mask_2d.unsqueeze(-1)
        unit_y = unit_y * bb_mask_2d.unsqueeze(-1)
        unit_z = unit_z * bb_mask_2d.unsqueeze(-1)

        # Amino acid type one-hot. Shape [B?, N, C].
        aatype_1hot = F.one_hot(template_aatype, self.embed_aatype_row.in_features).to(dtype)
        # Broadcast to [B?, N, N, C] for row and column views. Use unsqueeze
        # at negative indices so the leading batch dim (if present) is
        # preserved automatically.
        aa_row = aatype_1hot.unsqueeze(-3).expand(*aatype_1hot.shape[:-1], N, aatype_1hot.shape[-1])
        aa_col = aatype_1hot.unsqueeze(-2).expand(*aatype_1hot.shape[:-1], N, aatype_1hot.shape[-1])

        # Sum all 9 projected features
        act = self.embed_dgram(dgram.to(dtype))
        act = act + self.embed_pseudo_beta(pb_mask_2d.unsqueeze(-1).to(dtype))
        act = act + self.embed_aatype_row(aa_row)
        act = act + self.embed_aatype_col(aa_col)
        act = act + self.embed_unit_vec_x(unit_x.to(dtype))
        act = act + self.embed_unit_vec_y(unit_y.to(dtype))
        act = act + self.embed_unit_vec_z(unit_z.to(dtype))
        act = act + self.embed_backbone_mask(bb_mask_2d.unsqueeze(-1).to(dtype))
        act = act + map_pair_rows(
            lambda rows: self.embed_query(self.query_norm(rows)),
            query_pair,
            self.pair_chunk_budget,
            4 * query_pair.shape[-1],
        )

        # Template PairFormer stack (TemplatePairFormerBlock uses the same
        # batched primitives as the trunk).
        use_ckpt = self._grad_checkpoint and torch.is_grad_enabled()
        for layer in self.pairformer:
            act = _checkpoint(layer, act, pair_mask) if use_ckpt else layer(act, pair_mask)

        # Output normalization
        act = self.output_norm(act)
        return act


class TemplateEmbedding(nn.Module):
    """Embed a set of structural templates into pair-space.

    Processes each template independently through SingleTemplateEmbedding,
    averages the results, applies ReLU, and projects to pair_channel dims.

    Haiku scope: diffuser/evoformer/template_embedding/

    Attribute map:
        output_linear → output_proj
        single_template_embedding/ → single_template_emb
    """

    def __init__(self, cfg: TemplateEmbeddingConfig, optim: OptimConfig | None = None):
        super().__init__()
        self.cfg = cfg
        self.single_template_emb = SingleTemplateEmbedding(cfg, optim=optim)
        # AF3: output_linear uses initializer='relu'
        self.output_proj = LinearNoBias(cfg.num_channels, cfg.pair_channel, initializer="relu")

    def forward(
        self,
        query_pair: torch.Tensor,
        template_aatype: torch.Tensor,
        template_pseudo_beta: torch.Tensor,
        template_pseudo_beta_mask: torch.Tensor,
        template_backbone_frame: torch.Tensor,
        template_backbone_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        multichain_mask: torch.Tensor | None = None,
        template_ca: torch.Tensor | None = None,
        template_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed T templates and produce pair-space embedding.

        Supports both unbatched and batched inputs. The template axis ``T``
        is always the dim *immediately before* ``N`` in the template feature
        tensors. The leading dim is optional and used as the batch dim.

        Args:
            query_pair: [B?, N, N, pair_channel] query pair embedding from Evoformer.
            template_aatype: [B?, T, N] template residue types.
            template_pseudo_beta: [B?, T, N, 3] pseudo-beta positions.
            template_pseudo_beta_mask: [B?, T, N] pseudo-beta mask.
            template_backbone_frame: [B?, T, N, 3, 3] backbone rotation matrices.
            template_backbone_mask: [B?, T, N] backbone frame mask.
            pair_mask: [B?, N, N] padding mask for attention.
            multichain_mask: [B?, N, N] intra-chain mask (1 for same-chain
                pairs). Restricts template geometry to intra-chain pairs so the
                (frame-inconsistent) cross-chain off-diagonal is masked out.
            template_mask: [B?, T] 1.0 for real template slots, 0.0 for pad
                slots. Always emitted by the featurizer; pad slots arise both
                from ``pad_templates`` and from ``collate_fn`` padding the
                template axis to the batch maximum. Pad slots are zeroed out
                of the sum and excluded from the (per-sample) average
                denominator, so a sample's result does not depend on how many
                templates its batch-mates happened to draw. ``None`` ⇒ treat
                every slot as real (legacy / direct callers).

        Returns:
            Template pair embedding [B?, N, N, pair_channel].
        """
        has_batch = query_pair.dim() == 4  # [B, N, N, c_pair]
        # Template axis position (dim of T): -2 unbatched ([T, N]),
        # 1 batched ([B, T, N]).
        t_axis = 1 if has_batch else 0
        T = template_aatype.shape[t_axis]
        device = query_pair.device
        dtype = query_pair.dtype
        # Match the accumulator dtype: an fp32 mask against a bf16 ``emb``
        # would type-promote ``summed`` to fp32 for the whole loop (and the
        # ReLU), doubling memory traffic on an [B?, N, N, c] tensor.
        if template_mask is not None:
            template_mask = template_mask.to(dtype)

        # Initialize summed embedding with the right batch shape.
        if has_batch:
            B = query_pair.shape[0]
            N = query_pair.shape[1]
            summed = torch.zeros(B, N, N, self.cfg.num_channels, device=device, dtype=dtype)
        else:
            N = query_pair.shape[0]
            summed = torch.zeros(N, N, self.cfg.num_channels, device=device, dtype=dtype)

        for t in range(T):
            if has_batch:
                # Slice along the T dim (=dim 1) of each batched template feature.
                aatype_t = template_aatype[:, t]
                pb_t = template_pseudo_beta[:, t]
                pbm_t = template_pseudo_beta_mask[:, t]
                bbf_t = template_backbone_frame[:, t]
                bbm_t = template_backbone_mask[:, t]
                ca_t = template_ca[:, t] if template_ca is not None else None
                mask_t = template_mask[:, t].view(-1, 1, 1, 1) if template_mask is not None else None
            else:
                aatype_t = template_aatype[t]
                pb_t = template_pseudo_beta[t]
                pbm_t = template_pseudo_beta_mask[t]
                bbf_t = template_backbone_frame[t]
                bbm_t = template_backbone_mask[t]
                ca_t = template_ca[t] if template_ca is not None else None
                mask_t = template_mask[t] if template_mask is not None else None

            emb = self.single_template_emb(
                query_pair=query_pair,
                template_aatype=aatype_t,
                template_pseudo_beta=pb_t,
                template_pseudo_beta_mask=pbm_t,
                template_backbone_frame=bbf_t,
                template_backbone_mask=bbm_t,
                pair_mask=pair_mask,
                multichain_mask=multichain_mask,
                template_ca=ca_t,
            )
            # A pad slot is not self-zeroing: ``single_template_emb`` embeds
            # ``query_pair`` (embed_query) regardless of the per-residue masks,
            # so an all-masked slot still returns a non-zero tensor. Zero it
            # explicitly, or padding would shift the average.
            summed = (summed + emb) if mask_t is None else torch.addcmul(summed, emb, mask_t)

        # Average over the REAL templates, counted PER SAMPLE — see the
        # ``template_mask`` arg doc above for why ``T`` is the wrong divisor.
        # The ``+ 1e-7`` epsilon keeps a 0-template sample at 0/1e-7, not 0/0.
        if template_mask is None:
            # Legacy callers that pass no mask: every slot is real, and an
            # unbatched call cannot be diluted by collate.
            embedding = summed / (T + 1e-7)
        else:
            n_real = template_mask.sum(dim=-1)  # [B?] real slots for THIS sample
            if has_batch:
                n_real = n_real.view(-1, 1, 1, 1)
            embedding = summed / (n_real + 1e-7)

        # ReLU + project to pair_channel
        embedding = F.relu(embedding)
        embedding = self.output_proj(embedding)

        return embedding
