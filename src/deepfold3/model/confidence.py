"""Confidence head: _embed_features + 4-layer PairFormer → pLDDT, PAE, PDE, exp_resolved.

Haiku scope: diffuser~/confidence_head~/

The confidence pairformer uses full PairFormerIteration (with_single=True),
same as the trunk pairformer. Both pair and single tracks are updated.

Before the PairFormer, ``_embed_features`` adds to pair_act:
  1. Outer-sum of target_feat projections (left + right)
  2. Distogram from predicted representative-atom positions (39 bins)

After the PairFormer:
  - pLDDT:  LayerNorm → Linear on single track  [S, N, M*50]
  - PAE:    LayerNorm → Linear on pair track     [S, N, N, 64]
  - PDE:    LayerNorm → symmetrised half-logits   [S, N, N, 64]
  - Experimentally resolved: LayerNorm → Linear on single track [S, N, M*2]

Attribute names match the Haiku → PyTorch converter mapping.
"""

from __future__ import annotations

from functools import partial as _partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from deepfold3.model.config import ConfidenceConfig, OptimConfig
from deepfold3.model.embedders import DGRAM_NUM_BINS, dgram_from_positions
from deepfold3.model.evoformer import PairFormerBlock, _resolve_branch_residual_gain
from deepfold3.model.precision import fp32_scope
from deepfold3.model.primitives import LayerNorm, LinearNoBias, block_residual_scale, budget_rows, map_row_chunks

torch_checkpoint = _partial(_torch_checkpoint, use_reentrant=False, determinism_check="none")


class ConfidenceHead(nn.Module):
    """Shallow 4-layer PairFormer (with single track) for confidence prediction.

    Takes trunk pair + single representations, target_feat, and predicted atom
    positions. Outputs per-sample pLDDT and PAE.

    Haiku scope: diffuser~/confidence_head~/
    Layer scope: __layer_stack_no_per_layer/confidence_pairformer/

    Attribute names (for converter):
        layers   → layers (ModuleList of PairFormerBlock)
        plddt_logits_ln → plddt_logits_ln (LayerNorm on single before pLDDT)
        plddt_logits → plddt_logits (LinearNoBias, single → max_atoms*num_plddt_bins)
        pae_logits_ln → pae_logits_ln (LayerNorm on pair before PAE)
        pae_logits   → pae_logits   (LinearNoBias, pair → num_distogram_bins)
        pde_logits_ln → pde_logits_ln (LayerNorm on pair before PDE)
        pde_half_logits → pde_half_logits (LinearNoBias, pair → num_distogram_bins)
        exp_resolved_ln → exp_resolved_ln (LayerNorm on single)
        exp_resolved_logits → exp_resolved_logits (LinearNoBias, single → max_atoms*2)
        left_target_feat_project → left_target_feat_project (LinearNoBias)
        right_target_feat_project → right_target_feat_project (LinearNoBias)
        distogram_feat_project → distogram_feat_project (LinearNoBias)
    """

    def __init__(
        self,
        cfg: ConfidenceConfig,
        target_feat_dim: int = 447,
        optim: OptimConfig | None = None,
        fp32: bool = False,
    ):
        super().__init__()
        c_pair = cfg.pair_channel
        c_single = cfg.seq_channel
        self.fp32 = fp32
        # Checkpoint each PairFormer layer as well as the whole sample (below):
        # with the sample alone, its backward recompute holds all four layers'
        # intermediates at once, which ran a 640-token crop out of a 95 GB GH200.
        self._grad_checkpoint = bool(optim and optim.use_gradient_checkpointing)
        self.pair_chunk_budget = optim.pair_chunk_budget_bytes if optim is not None else 0

        # _embed_features: target_feat outer-sum + distogram from positions
        self.left_target_feat_project = LinearNoBias(target_feat_dim, c_pair)
        self.right_target_feat_project = LinearNoBias(target_feat_dim, c_pair)
        self.distogram_feat_project = LinearNoBias(DGRAM_NUM_BINS, c_pair)

        # target_feat / distogram projections use plain per-Linear Lecun init
        # (OF3/AF3 stock). The previous 3-way variance-matched sum-fusion
        # override was removed.

        # 4-layer PairFormer with single track (same as trunk)
        self.layers = nn.ModuleList(
            [
                PairFormerBlock(
                    c_pair=c_pair,
                    c_single=c_single,
                    num_pair_heads=cfg.pair_attention_num_head,
                    num_single_heads=cfg.single_attention_num_head,
                    single_key_dim=c_single,
                    single_value_dim=c_single,
                    optim=optim,
                    residual_out_scale=block_residual_scale(
                        i, cfg.num_layer, cfg.residual_output_scaling, cfg.uniform_residual_scale
                    ),
                    pair_bias_gate=cfg.pair_bias_gate,
                    pair_bias_gate_init_tau=cfg.pair_bias_gate_init_tau,
                    single_transition_linear_out_init_gain=cfg.single_transition_linear_out_init_gain,
                    single_transition_residual_gain=_resolve_branch_residual_gain(
                        i,
                        cfg.single_transition_residual_gain,
                        cfg.single_transition_residual_per_layer_extra,
                    ),
                    pair_transition_residual_gain=_resolve_branch_residual_gain(
                        i,
                        cfg.pair_transition_residual_gain,
                        cfg.pair_transition_residual_per_layer_extra,
                    ),
                    branch_scaler=cfg.branch_scaler,
                )
                for i in range(cfg.num_layer)
            ]
        )

        # Output heads: LayerNorm before logit projections (matching AF3)
        # AF3: all logit projections use final_init (zeros)
        max_atoms = cfg.max_atoms_per_token

        self.plddt_logits_ln = LayerNorm(c_single)
        self.plddt_logits = LinearNoBias(c_single, max_atoms * cfg.num_plddt_bins, initializer="zeros")

        self.pae_logits_ln = LayerNorm(c_pair)
        self.pae_logits = LinearNoBias(c_pair, cfg.num_distogram_bins, initializer="zeros")

        # PDE head: symmetrised half-logits on pair track
        self.pde_logits_ln = LayerNorm(c_pair)
        self.pde_half_logits = LinearNoBias(c_pair, cfg.num_distogram_bins, initializer="zeros")

        # Experimentally resolved head: per-atom 2-class on single track
        self.exp_resolved_ln = LayerNorm(c_single)
        self.exp_resolved_logits = LinearNoBias(c_single, max_atoms * 2, initializer="zeros")

    def _embed_features(
        self,
        target_feat: torch.Tensor,  # [N, target_feat_dim]
        positions: torch.Tensor,  # [N, 3] representative atom positions
        pair_mask: torch.Tensor,  # [N, N]
    ) -> torch.Tensor:
        """Compute additional pair features from target_feat and positions.

        Returns tensor to add to pair_act [N, N, c_pair].
        """
        # Target feat outer sum: left[i] + right[j] → [..., N, N, c_pair]
        out = self.left_target_feat_project(target_feat).unsqueeze(-2)  # [..., N, 1, c_pair]
        out = out + self.right_target_feat_project(target_feat).unsqueeze(-3)  # [..., 1, N, c_pair]

        # Distogram from predicted positions. fp32 guards the squared-distance
        # computation against bf16 catastrophic cancellation at small gaps.
        with fp32_scope(self.fp32):
            dgram = dgram_from_positions(positions.float() if self.fp32 else positions)
            dgram = dgram * pair_mask[..., None]

        return out + self.distogram_feat_project(dgram.to(out.dtype))

    def forward(
        self,
        pair: torch.Tensor,  # [B?, N, N, c_pair]
        single: torch.Tensor,  # [B?, N, c_single]
        pair_mask: torch.Tensor,  # [B?, N, N]
        seq_mask: torch.Tensor,  # [B?, N]
        target_feat: torch.Tensor,  # [B?, N, target_feat_dim]
        atom_positions: torch.Tensor,  # [B?, num_samples, N, M, 3]
        rep_atom_index: torch.Tensor | None = None,  # [B?, N] index of representative atom per token
        return_pre_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run confidence head per diffusion sample.

        Supports both unbatched and batched inputs. The PairFormer + LN +
        projection ops inside ``_process_one_sample`` already support
        ``[B?, N, ...]`` shapes natively, so the only batched-specific work
        here is gathering ``positions`` from ``atom_positions`` along the
        sample axis (which is ``dim=0`` unbatched / ``dim=1`` batched).

        Args:
            return_pre_logits: If True, return post-LN features instead of
                projected logits for pair-scale heads (PAE, PDE).  Used with
                fused chunked linear+CE to avoid materializing [S,N,N,64] logits.
                pLDDT and exp_resolved always return logits (small tensors).

        Returns:
            When ``return_pre_logits=False`` (default):
                plddt_logits:        [B?, num_samples, N, max_atoms * num_plddt_bins]
                pae_logits:          [B?, num_samples, N, N, num_distogram_bins]
                pde_logits:          [B?, num_samples, N, N, num_distogram_bins]
                exp_resolved_logits: [B?, num_samples, N, max_atoms * 2]
            When ``return_pre_logits=True``:
                pae_logits → pae_features [B?, num_samples, N, N, pair_channel]
                pde_logits → pde_features [B?, num_samples, N, N, pair_channel]
        """
        has_batch = pair.dim() == 4  # [B, N, N, c_pair]
        N = pair.shape[-3]
        device = pair.device

        # Sample-axis position: 0 unbatched, 1 batched (after a leading B).
        sample_axis = 1 if has_batch else 0
        num_samples = atom_positions.shape[sample_axis]

        # Default: use atom index 1 (Cα for proteins) as representative.
        if rep_atom_index is None:
            if has_batch:
                rep_atom_index = torch.ones(pair.shape[0], N, dtype=torch.long, device=device)
            else:
                rep_atom_index = torch.ones(N, dtype=torch.long, device=device)

        # Pre-compute the index helpers used to gather rep-atom positions.
        if has_batch:
            B = pair.shape[0]
            b_arange = torch.arange(B, device=device).unsqueeze(-1).expand(B, N)
            n_arange = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        else:
            n_arange = torch.arange(N, device=device)

        # Sample-axis position in the outputs: dim 0 unbatched, dim 1 batched
        # (the convention ``[B?, num_samples, N, ...]``).
        stack_dim = 1 if has_batch else 0
        # Inference writes each sample straight into a preallocated output. A
        # per-sample list followed by ``torch.stack`` holds the PAE and PDE
        # ``[N, N, 64]`` maps twice at the end (~31 GiB fp32 at N=2560, 5 samples);
        # this holds them once. With grad enabled the list + stack path is kept.
        prealloc = not torch.is_grad_enabled()
        outs: list[list[torch.Tensor] | torch.Tensor] = [[], [], [], []]

        use_ckpt = torch.is_grad_enabled()
        for s in range(num_samples):
            if has_batch:
                # atom_positions [B, num_samples, N, M, 3] → positions [B, N, 3]
                positions = atom_positions[b_arange, s, n_arange, rep_atom_index]
            else:
                # atom_positions [num_samples, N, M, 3] → [N, 3]
                positions = atom_positions[s, n_arange, rep_atom_index]

            if use_ckpt:
                sample_out = torch_checkpoint(
                    self._process_one_sample,
                    pair,
                    single,
                    target_feat,
                    positions,
                    pair_mask,
                    seq_mask,
                    return_pre_logits,
                )
            else:
                sample_out = self._process_one_sample(
                    pair,
                    single,
                    target_feat,
                    positions,
                    pair_mask,
                    seq_mask,
                    return_pre_logits,
                )

            for k, x in enumerate(sample_out):
                if not prealloc:
                    outs[k].append(x)
                    continue
                if s == 0:
                    shape = list(x.shape)
                    shape.insert(stack_dim, num_samples)
                    outs[k] = x.new_empty(shape)
                outs[k].select(stack_dim, s).copy_(x)
            del sample_out, x

        if not prealloc:
            outs = [torch.stack(xs, dim=stack_dim) for xs in outs]
        plddt, exp_res, pae, pde = outs
        result = {"plddt_logits": plddt, "exp_resolved_logits": exp_res}
        if return_pre_logits:
            result["pae_features"] = pae
            result["pde_features"] = pde
        else:
            result["pae_logits"] = pae
            result["pde_logits"] = pde
        return result

    def _process_one_sample(
        self,
        pair: torch.Tensor,
        single: torch.Tensor,
        target_feat: torch.Tensor,
        positions: torch.Tensor,
        pair_mask: torch.Tensor,
        seq_mask: torch.Tensor,
        return_pre_logits: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run PairFormer + logit projections for one confidence sample.

        Extracted as a separate method so it can be wrapped with
        ``torch.utils.checkpoint`` to free per-sample intermediates.
        """
        pair_act = pair.clone()
        single_act = single.clone()

        pair_act = pair_act + self._embed_features(target_feat, positions, pair_mask)

        use_ckpt = self._grad_checkpoint and torch.is_grad_enabled()
        for layer in self.layers:
            if use_ckpt:
                pair_act, single_act = torch_checkpoint(layer, pair_act, single_act, pair_mask, seq_mask)
            else:
                pair_act, single_act = layer(pair_act, single_act, pair_mask, seq_mask)

        # fp32_scope is what actually holds LN + matmul in fp32: a bare
        # ``.float()`` upcast is otherwise re-downcast by autocast on entry.
        # Over the pair chunk budget, the PAE / PDE heads run row-chunked, so the
        # fp32 copy of the pair and its LayerNorm exist one chunk at a time.
        rows = 0
        if not return_pre_logits and not torch.is_grad_enabled():
            rows = budget_rows(self.pair_chunk_budget, pair_act.shape[-2], 4 * pair_act.shape[-1])
        with fp32_scope(self.fp32):
            single_act = single_act.float()

            plddt = self.plddt_logits(self.plddt_logits_ln(single_act))
            exp_res = self.exp_resolved_logits(self.exp_resolved_ln(single_act))

            if rows:
                pae = map_row_chunks(lambda r: self.pae_logits(self.pae_logits_ln(r.float())), pair_act, rows)
                pde_half = map_row_chunks(lambda r: self.pde_half_logits(self.pde_logits_ln(r.float())), pair_act, rows)
                return plddt, exp_res, pae, pde_half + pde_half.transpose(-3, -2)

            pair_act = pair_act.float()
            if return_pre_logits:
                pae = self.pae_logits_ln(pair_act)
                pde = self.pde_logits_ln(pair_act)
            else:
                pae = self.pae_logits(self.pae_logits_ln(pair_act))
                pde_half = self.pde_half_logits(self.pde_logits_ln(pair_act))
                pde = pde_half + pde_half.transpose(-3, -2)

        return plddt, exp_res, pae, pde
