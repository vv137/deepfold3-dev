"""Top-level DeepFold3 model.

Assembles the full pipeline:
    EvoformerConditioningEncoder → raw_target_feat concat → Evoformer trunk
    [recycles] → DiffusionHead → ConfidenceHead

target_feat = concat(raw_feat[N,63], evo_cond_token_act[N,384]) → [N,447]
msa_feat: [S, N, 34]  (one_hot(32) + has_deletion + deletion_value)
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

from deepfold3.model.atom_cross_attn import EvoformerConditioningEncoder
from deepfold3.model.confidence import ConfidenceHead
from deepfold3.model.config import DistogramConfig, ModelConfig, OptimConfig
from deepfold3.model.diffusion import DiffusionHead, GuidanceTerm
from deepfold3.model.embedders import compute_rel_feat_dim, create_relative_encoding
from deepfold3.model.evoformer import Evoformer
from deepfold3.model.masked_msa import MaskedMSAHead, apply_bert_mask
from deepfold3.model.precision import fp32_scope
from deepfold3.model.primitives import TRUNCATED_NORMAL_STDDEV_FACTOR, LayerNorm, LinearNoBias
from deepfold3.model.restraints import Restraint

# Disable the determinism metadata check that compares saved vs recomputed
# tensor shapes.  View reshapes inside checkpointed functions (e.g.
# [1,384,384,128] → [147456,128]) trigger a spurious mismatch warning
# that floods the log.  The values are correct; only the view shape differs.
torch_checkpoint = _partial(_torch_checkpoint, use_reentrant=False, determinism_check="none")

# Raw target_feat dimensions (before evoformer conditioning)
# aatype_1hot(31) + profile(31) + deletion_mean(1) = 63
_RAW_TARGET_FEAT_DIM = 63
# aatype token classes (31 polymer types including unknown and gap)
_AATYPE_NUM_CLASSES = 31
_MSA_NUM_CLASSES = 32  # POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 1 (gap)


@dataclasses.dataclass
class DeepFold3Output:
    atom_positions: torch.Tensor  # [num_samples, N, M, 3]
    atom_mask: torch.Tensor  # [N, M]
    plddt: torch.Tensor | None  # [num_samples, N, M * num_plddt_bins]
    pae: torch.Tensor | None  # [num_samples, N, N, num_pae_bins]
    pde: torch.Tensor | None  # [num_samples, N, N, 64]
    exp_resolved: torch.Tensor | None  # [num_samples, N, M * 2]
    distogram: torch.Tensor | None  # [N, N, 64] logits


class DistogramHead(nn.Module):
    """Distogram head: pair activations → symmetrised distance bin logits.

    AF3 scope: diffuser/distogram_head/

    Attribute map:
        half_logits ← half_logits
    """

    def __init__(
        self,
        pair_channels: int = 128,
        cfg: DistogramConfig | None = None,
        fp32: bool = False,
        use_half_logits_ln: bool | None = None,
    ):
        super().__init__()
        if cfg is None:
            cfg = DistogramConfig()
        self.num_bins = cfg.num_bins
        self.first_break = cfg.first_break
        self.last_break = cfg.last_break
        self.fp32 = fp32
        # Explicit arg wins over cfg; otherwise use the cfg default.
        if use_half_logits_ln is None:
            use_half_logits_ln = bool(getattr(cfg, "use_half_logits_ln", False))
        self.half_logits_ln = LayerNorm(pair_channels) if use_half_logits_ln else None
        # AF3 ships ``final_init = zeros`` so the distogram starts as the
        # uniform 1/64 distribution. Side effect: the distogram CE delivers
        # no gradient back to ``pair`` until the projection accumulates
        # non-zero weight, leaving ``‖pair‖`` unconstrained for the first
        # ~hundreds of steps (the "stuck near log(64)" pattern seen in
        # 051858). We use a *small-normal* init instead:
        #     W ~ trunc_normal(0, (α/√fan_in)²)   default α = 0.05
        # → output stddev ≈ α (≈0.05) per element, softmax sits near uniform,
        # CE starts ≈ log(64), but backward through the Linear delivers
        # non-zero gradient to ``pair`` from step 0. ``α = 0`` recovers
        # AF3's zeros init.
        self.half_logits = LinearNoBias(pair_channels, self.num_bins)
        alpha = float(getattr(cfg, "half_logits_init_alpha", 0.0))
        if alpha == 0.0:
            nn.init.zeros_(self.half_logits.weight)
        else:
            fan_in = self.half_logits.weight.shape[1]
            target_std = alpha / math.sqrt(fan_in)
            stddev = target_std / TRUNCATED_NORMAL_STDDEV_FACTOR
            nn.init.trunc_normal_(self.half_logits.weight, mean=0.0, std=stddev, a=-2.0, b=2.0)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """Symmetrised logits [..., N, N, num_bins]."""
        with fp32_scope(self.fp32):
            z = pair.float() if self.fp32 else pair
            if self.half_logits_ln is not None:
                z = self.half_logits_ln(z)
            h = self.half_logits(z)
        return h + h.transpose(-3, -2)

    def expected_distance(self, pair: torch.Tensor) -> torch.Tensor:
        """Expected Cβ–Cβ distance in Å, shape [..., N, N]."""
        probs = self.forward(pair).softmax(dim=-1)
        device = pair.device
        breaks = torch.linspace(self.first_break, self.last_break, self.num_bins - 1, device=device)
        last_gap = breaks[-1] - breaks[-2]
        bin_lower = torch.cat([torch.zeros(1, device=device), breaks])
        bin_upper = torch.cat([breaks, breaks[-1:] + last_gap])
        bin_centers = (bin_lower + bin_upper) * 0.5
        return (probs * bin_centers).sum(-1)


class DeepFold3(nn.Module):
    """Full AlphaFold3-equivalent model in PyTorch.

    Haiku root scope: diffuser~/
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg

        ec = cfg.evoformer
        pfc = cfg.pairformer
        dc = cfg.diffusion
        cc = cfg.confidence

        # Evoformer conditioning encoder: ref atoms → token acts [N, 384]
        # Maps ref structure to the per-token contribution of target_feat
        self.evo_cond_encoder = EvoformerConditioningEncoder(
            per_atom_channels=dc.atom_cross_att.per_atom_channels,
            per_token_channels=ec.seq_channel,
            per_atom_pair_channels=dc.atom_cross_att.per_atom_pair_channels,
            num_blocks=dc.atom_cross_att.atom_transformer.num_blocks,
            Q=dc.atom_cross_att.queries_subset_size,
            K=dc.atom_cross_att.keys_subset_size,
            num_element_classes=dc.atom_cross_att.num_element_classes,
            num_atom_name_chars=dc.atom_cross_att.num_atom_name_chars,
            atom_name_length=dc.atom_cross_att.atom_name_length,
            ref_element_zero_index=dc.atom_cross_att.ref_element_zero_index,
        )

        # Evoformer trunk (with optional template embedding)
        self.evoformer = Evoformer(ec, pfc, template_cfg=cfg.template, optim=cfg.optim)

        # Diffusion head
        rel_feat_dim = compute_rel_feat_dim(ec.max_relative_idx, ec.max_relative_chain)
        self.diffusion_head = DiffusionHead(
            dc, c_single=ec.seq_channel, c_pair=ec.pair_channel, rel_feat_dim=rel_feat_dim, optim=cfg.optim
        )

        # Confidence head
        self.confidence_head = ConfidenceHead(cc, optim=cfg.optim, fp32=cfg.precision.fp32_confidence_head)

        # Distogram head
        self.distogram_head = DistogramHead(ec.pair_channel, cfg=cfg.distogram, fp32=cfg.precision.fp32_distogram_head)

        # Masked-MSA (BERT) head — optional self-supervised MSA objective.
        # Reads the final MSA-stack representation + trunk single repr.
        self.masked_msa_head = (
            MaskedMSAHead(cfg.masked_msa, c_msa=ec.msa_channel, c_single=ec.seq_channel)
            if cfg.masked_msa.enabled
            else None
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load state dict.

        All fused Triton wrapper modules are stateless after the 2026-04-20
        refactor — they read the parent module's parameters at call time and
        own none themselves. Old checkpoints may still contain keys under
        those names (`_ln_linear.*`, `_fused_pair_bias.*`, etc.); filter them
        out so strict-mode loading still works for the legacy weights.
        """
        markers = OptimConfig.FUSED_STATE_DICT_MARKERS
        filtered = {k: v for k, v in state_dict.items() if not any(m in k for m in markers)}
        return super().load_state_dict(filtered, strict=strict, assign=assign)

    def run_evoformer(
        self,
        features: dict[str, torch.Tensor],
        num_recycles: int,
        seed: int = 0,
        msa_precropped: bool = False,
        crop_offset: int = 0,
        mask_msa: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run feature building + evoformer trunk with recycling.

        Supports both unbatched ``[N, ...]`` and batched ``[B, N, ...]`` features.

        For each recycle, a fresh random subset of MSA rows is sampled
        (``msa_crop_size`` rows, always including the query row 0).
        This follows AF3 §3.3 (Algorithm 8, line 2).

        Returns:
            (embeddings, rel_feat, target_feat, seq_mask, pair_mask)
        """
        device = features["res_type"].device
        has_batch = features["res_type"].dim() >= 2
        N = features["res_type"].shape[-1]

        # Build fallback tensors matching input shape
        token_index = features.get("token_index")
        if token_index is None:
            token_index = torch.arange(N, dtype=torch.long, device=device)
            if has_batch:
                token_index = token_index.unsqueeze(0).expand_as(features["residue_index"])
        sym_id = features.get("sym_id")
        if sym_id is None:
            sym_id = torch.zeros_like(features["residue_index"])

        # Build relative position features
        rel_feat = create_relative_encoding(
            token_index=token_index,
            residue_index=features["residue_index"],
            asym_id=features["asym_id"],
            entity_id=features["entity_id"],
            sym_id=sym_id,
            max_relative_idx=self.cfg.evoformer.max_relative_idx,
            max_relative_chain=self.cfg.evoformer.max_relative_chain,
        )

        # Build raw target_feat [..., N, 63]
        raw_feat = self._build_raw_target_feat(features)

        # Evoformer conditioning: ref atoms → token_act (natively batched).
        ref_pos = features["ref_pos"]
        ref_mask = features["ref_mask"].float()
        ref_element = features["ref_element"]
        ref_charge = features["ref_charge"].float()
        ref_atom_name_chars = features["ref_atom_name_chars"]
        ref_space_uid = features.get("ref_space_uid")

        use_ckpt = torch.is_grad_enabled()
        # Match OF3 ``input_embedders.py:127`` — wrap the atom conditioning
        # encoder in fp32 autocast when ``cfg.precision.fp32_evo_cond_encoder``
        # is set. The 3D ref_pos projection + offset / inverse-distance pair
        # features have high dynamic range and bf16 accumulates noticeable
        # rounding error there. Cast the output back to surrounding dtype so
        # the downstream concat with ``raw_feat`` is shape- and dtype-clean.
        fp32_ace = self.cfg.precision.fp32_evo_cond_encoder
        amp_ctx = (
            torch.amp.autocast(device_type=ref_pos.device.type, dtype=torch.float32) if fp32_ace else nullcontext()
        )
        with amp_ctx:
            if use_ckpt:
                evo_cond_act = torch_checkpoint(
                    self.evo_cond_encoder,
                    ref_pos,
                    ref_mask,
                    ref_element,
                    ref_charge,
                    ref_atom_name_chars,
                    ref_space_uid,
                )
            else:
                evo_cond_act = self.evo_cond_encoder(
                    ref_pos=ref_pos,
                    ref_mask=ref_mask,
                    ref_element=ref_element,
                    ref_charge=ref_charge,
                    ref_atom_name_chars=ref_atom_name_chars,
                    ref_space_uid=ref_space_uid,
                )
        if fp32_ace:
            evo_cond_act = evo_cond_act.to(raw_feat.dtype)

        # Full target_feat [..., N, 447] = concat(raw[63], evo_cond[384])
        target_feat = torch.cat([raw_feat, evo_cond_act], dim=-1)

        seq_mask = features["seq_mask"].float()
        pair_mask = features["pair_mask"].float()

        # Full MSA tensors [..., S_full, N]
        full_msa = features["msa"]
        full_msa_mask = features["msa_mask"]
        full_del_matrix = features.get("deletion_matrix")

        # Initialize recycling buffers (or resume from features if warm-started)
        c_pair = self.cfg.evoformer.pair_channel
        c_single = self.cfg.evoformer.seq_channel
        # Match dtype of model parameters to avoid mixed-precision mismatch under autocast
        param_dtype = target_feat.dtype
        if "_prev_pair" in features:
            prev_pair = features["_prev_pair"]
            prev_single = features["_prev_single"]
        elif has_batch:
            B = features["res_type"].shape[0]
            prev_pair = torch.zeros(B, N, N, c_pair, device=device, dtype=param_dtype)
            prev_single = torch.zeros(B, N, c_single, device=device, dtype=param_dtype)
        else:
            prev_pair = torch.zeros(N, N, c_pair, device=device, dtype=param_dtype)
            prev_single = torch.zeros(N, c_single, device=device, dtype=param_dtype)

        # Extract template features (optional — T=0 tensors when no templates)
        template_kwargs: dict[str, torch.Tensor] = {}
        if "template_aatype" in features:
            template_kwargs["template_aatype"] = features["template_aatype"]
            template_kwargs["template_pseudo_beta"] = features["template_pseudo_beta"].float()
            template_kwargs["template_pseudo_beta_mask"] = features["template_pseudo_beta_mask"].float()
            template_kwargs["template_backbone_frame"] = features["template_backbone_frame"].float()
            template_kwargs["template_backbone_mask"] = features["template_backbone_mask"].float()
            # CA positions for the unit-vector feature (AF3 §2.8); absent on
            # legacy cached features → embedding falls back to pseudo-beta.
            if "template_ca" in features:
                template_kwargs["template_ca"] = features["template_ca"].float()
            # Real-vs-pad slot mask. The featurizer emits it unconditionally,
            # so the guard here is only for hand-built / legacy feature dicts;
            # absent ⇒ every slot is treated as real.
            if "template_mask" in features:
                template_kwargs["template_mask"] = features["template_mask"].float()

        # Bond matrix: polymer-ligand and ligand-ligand covalent bonds [..., N, N, 1]
        bond_matrix = features.get("bond_matrix")
        if bond_matrix is not None:
            bond_matrix = bond_matrix.to(device).float()

        # Multichain mask: same-chain pairs for template masking [..., N, N]
        asym_id_tensor = features["asym_id"].to(device)
        multichain_mask = (asym_id_tensor.unsqueeze(-1) == asym_id_tensor.unsqueeze(-2)).float()

        # Per-recycle MSA subsampling (AF3 §3.3, Algorithm 8). Two-stage:
        #   (1) Subsample the full MSA pool (up to ``max_msa_seqs``, e.g. 16384)
        #       down to a *fixed* tensor of ``msa_crop_size`` rows (default 1024).
        #       Index selection content is random per recycle, but tensor shape
        #       stays fixed → DDP-static-graph- / compile-friendly.
        #   (2) AF3 §2.3 random-``k`` mask. During training, ``k = Uniform[1, crop]``
        #       and rows ``k..crop-1`` are zeroed via ``msa_mask`` (tensor shape
        #       unchanged). At eval, ``k = crop`` so no extra masking.
        # Query row (index 0) is always retained.
        S_full = full_msa.shape[-2]  # MSA rows dimension
        crop = self.cfg.msa_crop_size
        rng = torch.Generator(device="cpu").manual_seed(seed)
        msa_rows_dim = full_msa.dim() - 2  # 0 for [S,N], 1 for [B,S,N]

        if msa_precropped:
            # Fail loud on a featurizer/model crop-size mismatch: the slices
            # below require S_full >= (crop_offset + num_recycles + 1) * crop.
            # (S_full = n_crops*crop from the featurizer; crop = cfg.msa_crop_size.)
            need = (crop_offset + num_recycles + 1) * crop
            if S_full < need:
                raise ValueError(
                    f"msa_precropped: MSA pool {S_full} < required {need} "
                    f"(crop={crop}, crop_offset={crop_offset}, num_recycles={num_recycles}). "
                    "Featurizer msa_crop_size / n_msa_crops must match the model."
                )

        # Masked-MSA (BERT) capture — populated on the masked recycle so the
        # head/loss can supervise reconstruction of the corrupted entries.
        # ``bert_valid_mask`` is the per-entry validity mask the head uses to
        # exclude padding / inactive MSA rows from its cross-attention.
        bert_true_msa: torch.Tensor | None = None
        bert_mask_cap: torch.Tensor | None = None
        bert_valid_mask: torch.Tensor | None = None

        # Recycle. ``msa_precropped`` and the resample path differ only in how
        # this recycle's (msa_sub, msa_mask_sub, del_sub) are selected; the
        # evoformer call + recycling-buffer update below are shared.
        for recycle_idx in range(num_recycles + 1):
            if msa_precropped:
                # The featurizer pre-built ``[n_crops * crop, N]`` (one crop per
                # recycle, already subsampled + Stage-2-masked on CPU) so the GPU
                # MSA shape is constant. Just slice this recycle's crop — no
                # on-GPU resample. ``crop_offset`` keeps the no_grad warmup pass
                # (crops 0..num_recycles-1) and the final grad pass (crop
                # num_recycles) on distinct crops, mirroring the resample path's
                # seed offset.
                base = (crop_offset + recycle_idx) * crop
                msa_sub = full_msa.narrow(msa_rows_dim, base, crop)
                msa_mask_sub = full_msa_mask.narrow(msa_rows_dim, base, crop)
                del_sub = full_del_matrix.narrow(msa_rows_dim, base, crop) if full_del_matrix is not None else None
            else:
                # --- Stage 1: subsample to a fixed-size pool of ``crop`` rows ---
                if S_full <= crop:
                    msa_sub = full_msa
                    msa_mask_sub = full_msa_mask
                    del_sub = full_del_matrix
                else:
                    perm = torch.randperm(S_full - 1, generator=rng) + 1  # indices 1..S_full-1
                    sel = perm[: crop - 1]
                    idx = torch.cat([torch.zeros(1, dtype=torch.long), sel]).sort().values.to(device)
                    msa_sub = full_msa.index_select(msa_rows_dim, idx)
                    msa_mask_sub = full_msa_mask.index_select(msa_rows_dim, idx)
                    del_sub = full_del_matrix.index_select(msa_rows_dim, idx) if full_del_matrix is not None else None

                # --- Stage 2: AF3 §2.3 random-``k`` mask ---
                # ``k = Uniform[1, n_full]`` and **at random** ``k`` rows are kept
                # (always including the query row 0). When ``k >= pool_size``
                # (≈94% of training steps when n_full=16384, crop=1024) no rows
                # are masked. When ``k < pool_size`` the kept rows are scattered
                # at random across the pool — *not* a leading prefix — so we do
                # not bias toward the closest homologs that Stage 1's sort puts
                # at low indices. Tensor shape unchanged → DDP static_graph safe.
                # Matches Protenix ``sample_indices``: ``randperm(n)[:k][:cutoff]``.
                if self.training:
                    pool_size = msa_sub.shape[msa_rows_dim]
                    k = int(torch.randint(1, S_full + 1, (1,), generator=rng).item())
                    if k < pool_size:
                        perm_k = torch.randperm(pool_size - 1, generator=rng) + 1
                        keep = torch.cat([torch.zeros(1, dtype=torch.long), perm_k[: k - 1]]).to(device)
                        active = torch.zeros(pool_size, dtype=msa_mask_sub.dtype, device=device)
                        active[keep] = 1
                        bcast_shape = [1] * msa_mask_sub.dim()
                        bcast_shape[msa_rows_dim] = pool_size
                        msa_mask_sub = msa_mask_sub * active.view(*bcast_shape)

            # Masked-MSA (BERT) corruption — applied to the gradient pass only
            # (``mask_msa=True``), before the MSA stack sees the crop. The
            # original tokens + corruption mask are captured for the head/loss;
            # the masked tokens are fed in via ``msa_feat`` below.
            if mask_msa and self.masked_msa_head is not None:
                mask_gen = torch.Generator(device=device).manual_seed(int(seed) + recycle_idx + 0x5EED)
                bert_msa, bert_true_msa, bert_mask_cap = apply_bert_mask(
                    msa_sub,
                    msa_mask_sub,
                    features.get("profile"),
                    self.cfg.masked_msa,
                    generator=mask_gen,
                )
                msa_sub = bert_msa
                # Validity mask for the head's cross-attention (excludes padding /
                # inactive rows). Captured before the ``.float()`` below.
                bert_valid_mask = msa_mask_sub

            # Build MSA features for this recycle [..., S_crop, N, 34]
            msa_feat = self._build_msa_feat_from_tensors(msa_sub, del_sub)
            msa_mask = msa_mask_sub.float()

            embeddings = self.evoformer(
                target_feat=target_feat,
                msa_feat=msa_feat,
                rel_feat=rel_feat,
                pair_mask=pair_mask,
                seq_mask=seq_mask,
                msa_mask=msa_mask,
                prev_pair=prev_pair,
                prev_single=prev_single,
                bond_matrix=bond_matrix,
                multichain_mask=multichain_mask,
                **template_kwargs,
            )
            if recycle_idx < num_recycles:
                prev_pair = embeddings["pair"].detach()
                prev_single = embeddings["single"].detach()

        # Expose the masked-MSA targets alongside the (grad-pass) ``msa``
        # representation so ``forward_training`` can run the head + loss.
        if bert_mask_cap is not None:
            embeddings["true_msa"] = bert_true_msa
            embeddings["bert_mask"] = bert_mask_cap
            embeddings["bert_msa_valid_mask"] = bert_valid_mask

        return embeddings, rel_feat, target_feat, seq_mask, pair_mask

    @staticmethod
    def _ref_kwargs(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Extract ref atom tensors needed by diffusion head."""
        return {
            "ref_pos": features["ref_pos"],
            "ref_mask": features["ref_mask"].float(),
            "ref_element": features["ref_element"],
            "ref_charge": features["ref_charge"].float(),
            "ref_atom_name_chars": features["ref_atom_name_chars"],
            "ref_space_uid": features.get("ref_space_uid"),
        }

    def forward_training(
        self,
        features: dict[str, torch.Tensor],
        num_recycles: int,
        diffusion_batch_size: int,
        mini_rollout_steps: int = 20,
        mini_rollout_samples: int = 1,
        seed: int = 0,
        fused_ce: bool = False,
        diffusion_chunk_size: int = 0,
        backward_fn=None,
        loss_cfg=None,
        compute_confidence: bool = True,
        msa_precropped: bool = False,
        chain_perm_gate_k: int = 1,
        permute_gt_before_noise: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Training forward pass with recycling, diffusion, confidence, and distogram.

        Encapsulates the full training pipeline:

        1. Evoformer trunk with recycling (gradient only on last cycle).
        2. Mini-rollout (no gradient, reduced steps/samples) — runs before
           diffusion so the chain/ligand permutation in the diffusion MSE
           can read ``mini_rollout_pos`` from ``pred``.
        3. Single-step diffusion training (augment GT, add noise, denoise).
        4. Confidence head on mini-rollout positions.
        5. Distogram head.

        Args:
            features: Batched feature dict from the dataloader.
            num_recycles: Number of recycles (gradient flows through last only).
            diffusion_batch_size: Number of noise samples for diffusion training.
            mini_rollout_steps: Denoising steps for the confidence mini-rollout.
            mini_rollout_samples: Diffusion samples in the mini-rollout.
            seed: Random seed for MSA subsampling.
            fused_ce: Whether to defer cross-entropy projection (for fused loss).
            diffusion_chunk_size: If >0 and divides ``diffusion_batch_size``,
                run diffusion in chunks of this size, computing the per-chunk
                loss and back-propping it via ``backward_fn`` before the next
                chunk's forward. Reduces peak memory by ~``diffusion_batch_size
                / diffusion_chunk_size``×. Requires ``backward_fn`` and
                ``loss_cfg`` to be set.
            backward_fn: Callable ``(loss: Tensor) -> None`` used by the
                chunked path to back-prop each chunk's diffusion loss. The
                trainer wraps ``self.scaler.scale(loss / accumulate).backward``.
            loss_cfg: ``TrainingConfig`` instance — needed by the chunked path
                to compute per-chunk diffusion losses via ``diffusion_only_loss``.

        Returns:
            Dict. In the unchunked path, contains ``denoised``, ``gt_augmented``,
            ``noise_levels`` plus the confidence/distogram outputs. In the
            chunked path, the diffusion-loss components are stashed under
            ``__diffusion_logged__`` (already back-propped); ``denoised`` etc.
            are not returned.
        """
        ref_kwargs = self._ref_kwargs(features)

        # --- 1. Evoformer trunk (gradient only on the FINAL pass) ---
        # AF3 convention: ``num_recycles`` is the number of *additional*
        # trunk passes after the first. Total trunk forwards = num_recycles + 1.
        #   num_recycles=0 → 1 pass (just initial, no recycling)
        #   num_recycles=N → N warmup passes (no_grad) + 1 grad pass
        # Warmup passes update ``prev_pair``/``prev_single`` to seed the
        # final grad pass.
        if num_recycles > 0:
            with torch.no_grad():
                embeddings_det, _, _, _, _ = self.run_evoformer(
                    features, num_recycles - 1, seed=seed, msa_precropped=msa_precropped, crop_offset=0
                )
            # Pass warm-start buffers via features for the final (grad-enabled) pass.
            # No .detach() needed — tensors from no_grad() context already lack grad_fn.
            features["_prev_pair"] = embeddings_det["pair"]
            features["_prev_single"] = embeddings_det["single"]

        # Masked-MSA (BERT) corruption is applied on this (gradient) pass only,
        # and only while training (eval runs the head-free clean forward).
        do_mask_msa = self.masked_msa_head is not None and self.training
        embeddings, rel_feat, target_feat, seq_mask, pair_mask = self.run_evoformer(
            features,
            0,
            seed=seed + num_recycles,
            msa_precropped=msa_precropped,
            crop_offset=num_recycles,
            mask_msa=do_mask_msa,
        )
        # Clean up side-channel keys to avoid stale references
        features.pop("_prev_pair", None)
        features.pop("_prev_single", None)
        embeddings["target_feat"] = target_feat

        pred: dict[str, torch.Tensor] = {}
        # Expose the attached single / pair representations for the magnitude
        # regularizers / observers in ``training_loss``.  The reference alone
        # does not extend the autograd graph; only the loss call does.
        pred["single"] = embeddings["single"]
        pred["pair"] = embeddings["pair"]

        # --- 1b. Masked-MSA (BERT) head ---
        # Predicts the original residue identity at corrupted MSA entries from
        # the (attached) final MSA-stack repr + single repr, so the loss trains
        # the MSA stack + single track. Optionally checkpointed (the head's
        # [S, N, hidden] activations are large at S=msa_crop_size).
        if self.masked_msa_head is not None and "bert_mask" in embeddings:
            msa_valid = embeddings["bert_msa_valid_mask"]
            if torch.is_grad_enabled() and self.cfg.optim.use_gradient_checkpointing:
                msa_logits = torch_checkpoint(self.masked_msa_head, embeddings["msa"], embeddings["single"], msa_valid)
            else:
                msa_logits = self.masked_msa_head(embeddings["msa"], embeddings["single"], msa_valid)
            pred["masked_msa_logits"] = msa_logits
            pred["true_msa"] = embeddings["true_msa"]
            pred["bert_mask"] = embeddings["bert_mask"]

        # --- 2. Mini-rollout (no gradient) ---
        # Runs before diffusion so the chunked diffusion MSE can read
        # ``mini_rollout_pos`` for chain/ligand permutation (AF3 §4.3).
        # eval() + detach inputs is defence-in-depth against any submodule
        # that tracks grad through leaves under no_grad.
        with torch.no_grad():
            diff_cfg = self.diffusion_head.cfg
            orig_steps, orig_samples = diff_cfg.steps, diff_cfg.num_samples
            diff_cfg.steps = mini_rollout_steps
            # Chain-perm stability gate needs ``chain_perm_gate_k`` independent
            # rollout draws; the confidence head still consumes only the first
            # ``mini_rollout_samples`` of them (sliced below), so its behaviour is
            # unchanged. gate_k <= mini_rollout_samples → no extra samples.
            rollout_samples = max(mini_rollout_samples, chain_perm_gate_k)
            diff_cfg.num_samples = rollout_samples
            was_training = self.diffusion_head.training
            self.diffusion_head.eval()
            try:
                mini_embeddings = {k: v.detach() for k, v in embeddings.items()}
                mini_result = self.diffusion_head.sample(
                    embeddings=mini_embeddings,
                    seq_mask=seq_mask,
                    rel_feat=rel_feat.detach(),
                    guidance=False,  # the confidence head trains on unguided samples
                    **{k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in ref_kwargs.items()},
                )
            finally:
                diff_cfg.steps, diff_cfg.num_samples = orig_steps, orig_samples
                if was_training:
                    self.diffusion_head.train()

        mini_full = mini_result["atom_positions"].detach()  # [B, rollout_samples, N, M, 3]
        if chain_perm_gate_k > 1:
            # Keep ``mini_rollout_pos`` at its original ``mini_rollout_samples``
            # count (confidence / tm_true / atom-symmetry consumers assume it);
            # stash the full K-sample rollout separately for the chain-perm vote.
            pred["chain_perm_gate_pos"] = mini_full
            pred["mini_rollout_pos"] = mini_full[:, :mini_rollout_samples]
        else:
            pred["mini_rollout_pos"] = mini_full

        # --- 2b. Permute-then-noise (AF3 §4.2/4.3; matches OpenFold3/Protenix) ---
        # Resolve chain + atom/ligand symmetry ONCE from the mini-rollout and apply
        # it to the GT that gets NOISED below, so the denoiser's noised input and
        # its MSE target share one atom labelling (no midpoint-collapse / chain
        # latch from a swapped rollout). The losses re-derive the same permutation
        # for the shared un-augmented GT and skip re-permuting ``gt_augmented``
        # (guarded on ``features["__gt_prepermuted__"]``).
        diffusion_gt_pos = features["atom_gt_positions"]
        diffusion_gt_mask = features["atom_gt_mask"]
        if permute_gt_before_noise and pred.get("mini_rollout_pos") is not None:
            from deepfold3.training.collectives import slice_batch_dict
            from deepfold3.training.loss import permute_gt_for_diffusion

            B_gt, N_tok = diffusion_gt_pos.shape[0], diffusion_gt_pos.shape[1]
            device = diffusion_gt_pos.device
            # empty_like (not clone): every slice is overwritten in the loop below.
            perm_pos = torch.empty_like(diffusion_gt_pos)
            perm_mask = torch.empty_like(diffusion_gt_mask)
            # Batched chain-perm index (identity default) stashed for the loss to
            # reuse instead of re-deriving the assignment a second time — see
            # loss._resolve_chain_perm_index.
            cp_index = torch.arange(N_tok, dtype=torch.long, device=device).expand(B_gt, N_tok).contiguous()
            cp_keep = torch.ones(B_gt, N_tok, dtype=torch.bool, device=device)
            for b in range(B_gt):
                # Curated pred slice: the helper reads only these two keys, so avoid
                # slicing the (large) trunk tensors that slice_batch_dict(pred) would.
                pred_b = {"mini_rollout_pos": pred["mini_rollout_pos"][b]}
                if "chain_perm_gate_pos" in pred:
                    pred_b["chain_perm_gate_pos"] = pred["chain_perm_gate_pos"][b]
                features_b = slice_batch_dict(features, b, B_gt)
                gp, gm, ci, ck = permute_gt_for_diffusion(pred_b, features_b, stability_gate_k=chain_perm_gate_k)
                perm_pos[b] = gp
                perm_mask[b] = gm
                if ci is not None:
                    cp_index[b] = ci
                    cp_keep[b] = ck
            diffusion_gt_pos = perm_pos
            diffusion_gt_mask = perm_mask
            features["__gt_prepermuted__"] = True
            features["chain_perm_index"] = cp_index
            features["chain_perm_keep"] = cp_keep

        # --- 3. Diffusion training step ---
        # Trunk-conditioning dropout (OF3 ``use_conditioning_prob``): ONE draw
        # per forward, shared by every diffusion sample, exactly as OF3 does it
        # (`projects/of3_all_atom/model.py:487`). Training-only. Zeroing an
        # input still reaches every parameter, so DDP ``static_graph`` is safe.
        diff_cfg_t = self.diffusion_head.cfg
        use_conditioning = True
        if self.training and diff_cfg_t.use_conditioning_prob < 1.0:
            use_conditioning = bool(torch.rand(()) < diff_cfg_t.use_conditioning_prob)
        # A fired drop may instead zero a random subset of inter-chain pair
        # blocks (``cond_drop_interface_frac``). Expressed as a masked copy of
        # the trunk pair with ``use_conditioning`` back on; crops without an
        # interface keep the full drop. No RNG draw unless a drop fired.
        cond_pair_keep = None
        if (
            not use_conditioning
            and diff_cfg_t.cond_drop_interface_frac > 0.0
            and features.get("asym_id") is not None
            and bool(torch.rand(()) < diff_cfg_t.cond_drop_interface_frac)
        ):
            from deepfold3.model.diffusion import draw_interface_pair_keep

            cond_pair_keep = draw_interface_pair_keep(features["asym_id"], seq_mask)
            if cond_pair_keep is not None:
                use_conditioning = True
        if self.training and diff_cfg_t.use_conditioning_prob < 1.0:
            from deepfold3.model.diffusion import cond_drop_stats

            # Read by ``training_step`` into ``train/cond_drop/*``. Emitted on
            # every forward whenever the drop is configured, so all ranks log
            # the same key set.
            pred["__cond_drop__"] = cond_drop_stats(
                features.get("asym_id"), seq_mask, use_conditioning=use_conditioning, pair_keep=cond_pair_keep
            )

        # Masked / partial diffusion (motif scaffolding) — see
        # ``DiffusionConfig.diffusion_mask_prob``. Built here because it needs
        # ``asym_id``, and AFTER the chain permutation above so a "clean chain"
        # names the same copy the GT was permuted to.
        clean_mask = None
        if self.training and diff_cfg_t.diffusion_mask_prob > 0.0 and features.get("asym_id") is not None:
            from deepfold3.model.diffusion import build_diffusion_clean_mask

            clean_mask = build_diffusion_clean_mask(
                features["asym_id"],
                seq_mask,
                diffusion_batch_size,
                prob=diff_cfg_t.diffusion_mask_prob,
                mode=diff_cfg_t.diffusion_mask_mode,
                min_frac=diff_cfg_t.diffusion_mask_min_frac,
                max_frac=diff_cfg_t.diffusion_mask_max_frac,
            )

        chunked = (
            diffusion_chunk_size > 0
            and backward_fn is not None
            and loss_cfg is not None
            and diffusion_batch_size % diffusion_chunk_size == 0
        )
        if chunked:
            # Detach trunk outputs into leaves so each chunk's backward can
            # run with ``retain_graph=False`` and free its activations
            # without dragging the (much larger) trunk graph with it. After
            # the chunk loop, accumulated leaf grads are flowed back through
            # the trunk in one walk. ``retain_graph=True`` keeps the trunk
            # graph alive for the distogram backward (which still uses the
            # attached ``embeddings["pair"]``).
            from deepfold3.training.loss import diffusion_only_loss

            pair_real = embeddings["pair"]
            single_real = embeddings["single"]
            target_feat_real = target_feat

            pair_det = pair_real.detach().requires_grad_(True)
            single_det = single_real.detach().requires_grad_(True)
            target_feat_det = target_feat_real.detach().requires_grad_(True)

            emb_det = {
                "pair": pair_det,
                "single": single_det,
                "target_feat": target_feat_det,
            }
            if cond_pair_keep is not None:
                from deepfold3.model.diffusion import mask_trunk_pair

            num_chunks = diffusion_batch_size // diffusion_chunk_size
            logged: dict[str, torch.Tensor] = {}
            weighted_total: torch.Tensor = pair_real.new_zeros(())
            # ``diffusion_only_loss`` assumes unbatched ``[N, ...]`` features.
            # Always slice per batch element (batch dim is always kept).
            B_pair = pair_real.shape[0]  # [B, N, N, C]

            for chunk_i in range(num_chunks):
                # ``clean_mask`` is [B, diffusion_batch_size, N]; hand each chunk
                # its own slice so the per-sample fire decisions are preserved.
                cm_c = None
                if clean_mask is not None:
                    cm_c = clean_mask[:, chunk_i * diffusion_chunk_size : (chunk_i + 1) * diffusion_chunk_size]
                # Mask per chunk: each chunk's backward frees its graph, and a
                # mask node built once before the loop would be part of it.
                emb_c = emb_det if cond_pair_keep is None else mask_trunk_pair(emb_det, cond_pair_keep)
                gt_aug_c, denoised_c, noise_levels_c = self.diffusion_head.sample_training(
                    gt_positions=diffusion_gt_pos,
                    gt_mask=diffusion_gt_mask,
                    embeddings=emb_c,
                    seq_mask=seq_mask,
                    rel_feat=rel_feat,
                    num_samples=diffusion_chunk_size,
                    clean_mask=cm_c,
                    use_conditioning=use_conditioning,
                    **ref_kwargs,
                )
                # chunk_pred shapes:
                #   denoised, gt_augmented: [B, S_chunk, N, M, 3]
                #   noise_levels: [B, S_chunk]
                chunk_loss = pair_real.new_zeros(())
                chunk_components: dict[str, torch.Tensor] = {}
                mini_pos_batched = pred["mini_rollout_pos"]  # [B, S_mini, N, M, 3]
                for b in range(B_pair):
                    chunk_pred_b = {
                        "denoised": denoised_c[b],
                        "gt_augmented": gt_aug_c[b],
                        "noise_levels": noise_levels_c[b],
                        "mini_rollout_pos": mini_pos_batched[b],
                    }
                    if cm_c is not None:
                        chunk_pred_b["diffusion_clean_mask"] = cm_c[b]
                    if "chain_perm_gate_pos" in pred:
                        chunk_pred_b["chain_perm_gate_pos"] = pred["chain_perm_gate_pos"][b]
                    features_b = {
                        k: (v[b] if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == B_pair else v)
                        for k, v in features.items()
                    }
                    loss_b, comp_b = diffusion_only_loss(chunk_pred_b, features_b, loss_cfg)
                    chunk_loss = chunk_loss + loss_b / B_pair
                    for k, v in comp_b.items():
                        chunk_components[k] = chunk_components.get(k, v.new_zeros(())) + v / B_pair
                # Each chunk contributes 1/num_chunks of the full diffusion
                # loss; the accumulated grads on emb_det / diffusion params
                # equal the unchunked equivalent.
                backward_fn(chunk_loss / num_chunks)
                weighted_total = weighted_total + chunk_loss.detach() / num_chunks
                for k, v in chunk_components.items():
                    logged[k] = logged.get(k, v.new_zeros(())) + v / num_chunks

            # Defensive: ``torch.autograd.backward`` cannot take ``None`` as a
            # grad_tensor for a non-scalar — it raises "grad can be implicitly
            # created only for scalar outputs" — so a leaf the diffusion path
            # never read would kill the run on its first backward rather than
            # being ignored. All three are read today (``single_real`` by the
            # atom encoder's per-atom conditioning, even when the trunk single
            # track and the conditioning concat are both off), so this filter is
            # a guard against a future consumer being removed, not a live path.
            flow = [
                (t, d.grad)
                for t, d in ((pair_real, pair_det), (single_real, single_det), (target_feat_real, target_feat_det))
                if d.grad is not None
            ]
            if flow:
                torch.autograd.backward(
                    tensors=[t for t, _ in flow],
                    grad_tensors=[g for _, g in flow],
                    retain_graph=True,
                )

            pred["__diffusion_total__"] = weighted_total
            pred["__diffusion_logged__"] = logged
        else:
            diff_embeddings = embeddings
            if cond_pair_keep is not None:
                from deepfold3.model.diffusion import mask_trunk_pair

                diff_embeddings = mask_trunk_pair(embeddings, cond_pair_keep)
            gt_aug, denoised, noise_levels = self.diffusion_head.sample_training(
                gt_positions=diffusion_gt_pos,
                gt_mask=diffusion_gt_mask,
                embeddings=diff_embeddings,
                seq_mask=seq_mask,
                rel_feat=rel_feat,
                num_samples=diffusion_batch_size,
                clean_mask=clean_mask,
                use_conditioning=use_conditioning,
                **ref_kwargs,
            )
            pred["denoised"] = denoised
            pred["gt_augmented"] = gt_aug
            pred["noise_levels"] = noise_levels
            if clean_mask is not None:
                # Read by ``training_loss`` so the diffusion MSE scores only the
                # atoms whose input was noise. [B, S, N] — ``slice_batch_dict``
                # peels the batch dim to [S, N] for the per-example loss.
                pred["diffusion_clean_mask"] = clean_mask

        # --- 4. Confidence head ---
        # ``compute_confidence`` gates only the confidence head forward so
        # the caller can disable confidence supervision (e.g. warmup
        # phase). Mini-rollout stays on to keep chain-perm deterministic.
        # Skipping the head entirely — not just the loss — is required
        # because a NaN/Inf in the confidence logits would contaminate
        # ``total_loss`` even when multiplied by ``alpha_confidence=0``
        # (``0 * NaN == NaN``). The flag must stay constant across a run
        # so DDP ``static_graph=True`` sees an invariant reached-parameter
        # set.
        if compute_confidence:
            # AF3 §3.7: confidence head trains over fixed (detached) trunk
            # outputs and the (already-detached) mini-rollout positions, so
            # confidence-loss gradients update only confidence-head parameters.
            # OF3 ``use_confidence_emb_prob`` (`head_modules.py:195-196`):
            # zero the trunk PAIR embedding into the confidence head with some
            # probability, training-only, pair-only. The distogram head below
            # reads the un-zeroed ``embeddings["pair"]``, matching OF3's
            # ordering, so the distogram loss is never affected.
            conf_pair = embeddings["pair"].detach()
            if self.training and self.cfg.confidence.use_pair_emb_prob < 1.0:
                if not bool(torch.rand(()) < self.cfg.confidence.use_pair_emb_prob):
                    conf_pair = conf_pair * 0
            conf_out = self.confidence_head(
                pair=conf_pair,
                single=embeddings["single"].detach(),
                pair_mask=pair_mask,
                seq_mask=seq_mask,
                target_feat=target_feat.detach(),
                atom_positions=pred["mini_rollout_pos"],
                rep_atom_index=features.get("token_to_rep_atom"),
                return_pre_logits=fused_ce,
            )

            pred["plddt"] = conf_out["plddt_logits"]
            pred["exp_resolved"] = conf_out["exp_resolved_logits"]

            if fused_ce:
                pred["pae_features"] = conf_out["pae_features"]
                pred["pde_features"] = conf_out["pde_features"]
                pred["pae_weight"] = self.confidence_head.pae_logits.weight
                pred["pde_weight"] = self.confidence_head.pde_half_logits.weight
            else:
                pred["pae"] = conf_out["pae_logits"]
                pred["pde"] = conf_out["pde_logits"]

        # --- 5. Distogram head (independent of mini-rollout) ---
        if fused_ce:
            pred["distogram_features"] = embeddings["pair"]
            pred["distogram_weight"] = self.distogram_head.half_logits.weight
        else:
            pred["distogram"] = self.distogram_head(embeddings["pair"])

        return pred

    def forward(
        self,
        features: dict[str, torch.Tensor],
        seed: int = 0,
        num_recycles: int | None = None,
        num_samples: int | None = None,
        guidance_terms: Sequence[GuidanceTerm] | None = None,
        restraints: Sequence[Restraint] = (),
        restraint_weight: float = 0.0,
        restraint_rigid: bool = True,
        restraint_max_step: float = 2.0,
        restraint_after_step: bool = False,
        restraint_after_sigma: float = 16.0,
    ) -> DeepFold3Output:
        """Run full forward pass.

        Args:
            features: dict from featurizer with keys:
                token_index, residue_index, asym_id, entity_id, sym_id,
                mol_type, res_type, ref_pos, ref_mask, ref_element, ref_charge,
                ref_atom_name_chars, atom_to_token, token_to_rep_atom,
                msa, msa_mask, profile, deletion_mean, pair_mask, seq_mask
            seed: random seed for diffusion sampling
            num_recycles: override default number of recycles
            num_samples: override default number of diffusion samples
            guidance_terms: classifier-free guidance terms, replacing the
                ``cfg.diffusion.guidance_*`` settings
            restraints: distance restraints steered into the sampler
                (:mod:`deepfold3.model.restraints`); unlike guidance these act
                on coordinates, not on the conditioning
            restraint_weight: how much of each restraint's violation to remove
                per step
            restraint_rigid: move each restrained chain as a rigid body rather
                than pulling the named residues alone
            restraint_max_step: angstrom cap on the displacement one step may
                apply
            restraint_after_step: move the coordinates after the Euler step
                instead of the prediction before it, so the denoiser cannot
                undo it
            restraint_after_sigma: with ``restraint_after_step``, stop pushing
                below this sigma-hat and let the tail of the schedule repair
                and finish the structure
        """
        if num_recycles is None:
            num_recycles = self.cfg.num_recycles

        # Temporarily override num_samples if provided, restoring after
        orig_num_samples = self.diffusion_head.cfg.num_samples
        if num_samples is not None:
            self.diffusion_head.cfg.num_samples = num_samples

        try:
            embeddings, rel_feat, target_feat, seq_mask, pair_mask = self.run_evoformer(
                features, num_recycles, seed=seed
            )

            # Diffusion sampling
            device = features["res_type"].device
            generator = torch.Generator(device=device).manual_seed(seed)

            diff_cfg = self.diffusion_head.cfg
            guidance_pair_keep = None
            if guidance_terms is None and diff_cfg.guidance_scale != 1.0 and diff_cfg.guidance_drop == "interface":
                from deepfold3.model.diffusion import interface_pair_keep

                guidance_pair_keep = interface_pair_keep(features["asym_id"].to(device))
            sample_result = self.diffusion_head.sample(
                embeddings=embeddings,
                seq_mask=seq_mask,
                rel_feat=rel_feat,
                generator=generator,
                guidance_pair_keep=guidance_pair_keep,
                guidance_terms=guidance_terms,
                restraints=restraints,
                restraint_weight=restraint_weight,
                restraint_rigid=restraint_rigid,
                restraint_max_step=restraint_max_step,
                restraint_after_step=restraint_after_step,
                restraint_after_sigma=restraint_after_sigma,
                **self._ref_kwargs(features),
            )

            # Confidence estimates (per-sample, matching AF3)
            conf_out = self.confidence_head(
                pair=embeddings["pair"],
                single=embeddings["single"],
                pair_mask=pair_mask,
                seq_mask=seq_mask,
                target_feat=target_feat,
                atom_positions=sample_result["atom_positions"],
                rep_atom_index=features.get("token_to_rep_atom"),
            )

            disto_logits = self.distogram_head(embeddings["pair"])

            return DeepFold3Output(
                atom_positions=sample_result["atom_positions"],
                atom_mask=sample_result["mask"],
                plddt=conf_out["plddt_logits"],
                pae=conf_out["pae_logits"],
                pde=conf_out["pde_logits"],
                exp_resolved=conf_out["exp_resolved_logits"],
                distogram=disto_logits,
            )
        finally:
            self.diffusion_head.cfg.num_samples = orig_num_samples

    def _build_raw_target_feat(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build raw token feature vector [N, 63].

        Matches AF3 create_target_feat(batch, append_per_atom_features=False):
            aatype_1hot(31) + profile(31) + deletion_mean(1) = 63
        """
        feat_parts = []

        # aatype one-hot: 31 classes (POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP)
        aatype_1hot = F.one_hot(features["res_type"], _AATYPE_NUM_CLASSES).float()
        feat_parts.append(aatype_1hot)

        # Profile: per-position amino acid frequency (31-dim)
        if "profile" in features:
            prof = features["profile"].float()
            # Profile may be 22-dim (legacy) or 31-dim; handle both
            if prof.shape[-1] < _AATYPE_NUM_CLASSES:
                pad = torch.zeros(*prof.shape[:-1], _AATYPE_NUM_CLASSES - prof.shape[-1], device=prof.device)
                prof = torch.cat([prof, pad], dim=-1)
            feat_parts.append(prof[..., :_AATYPE_NUM_CLASSES])
        else:
            feat_parts.append(aatype_1hot)  # use one-hot as profile for single-seq

        # Deletion mean: [N, 1]
        if "deletion_mean" in features:
            feat_parts.append(features["deletion_mean"].float().unsqueeze(-1))
        else:
            feat_parts.append(torch.zeros(*features["res_type"].shape, 1, device=features["res_type"].device))

        return torch.cat(feat_parts, dim=-1)  # [N, 63]

    def _build_msa_feat(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build MSA feature tensor [S, N, 34] from the full features dict."""
        return self._build_msa_feat_from_tensors(features["msa"], features.get("deletion_matrix"))

    def _build_msa_feat_from_tensors(
        self,
        msa: torch.Tensor,
        deletion_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build MSA feature tensor [..., S, N, 34].

        Matches AF3 create_msa_feat:
            msa_1hot(32) + has_deletion(1) + deletion_value(1) = 34

        Supports both unbatched [S, N] and batched [B, S, N] input.
        """
        msa_1hot = F.one_hot(msa, _MSA_NUM_CLASSES).float()  # [..., S, N, 32]

        parts = [msa_1hot]
        if deletion_matrix is not None:
            del_mat = deletion_matrix.float()
            has_del = del_mat.clamp(0.0, 1.0).unsqueeze(-1)
            del_val = (torch.arctan(del_mat / 3.0) * (2.0 / torch.pi)).unsqueeze(-1)
            parts.append(has_del)
            parts.append(del_val)
        else:
            parts.append(torch.zeros(*msa.shape, 2, device=msa.device))

        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def compute_distogram(
        self,
        features: dict[str, torch.Tensor],
        num_recycles: int | None = None,
    ) -> torch.Tensor:
        """Run evoformer + distogram head only (no diffusion).

        Fast validation path — useful for checking whether evoformer
        embeddings are correct without paying the cost of 200-step diffusion.

        Returns distogram logits [N, N, 64].
        """
        if num_recycles is None:
            num_recycles = self.cfg.num_recycles

        embeddings, _, _, _, _ = self.run_evoformer(features, num_recycles)
        return self.distogram_head(embeddings["pair"])  # [N, N, 64]
