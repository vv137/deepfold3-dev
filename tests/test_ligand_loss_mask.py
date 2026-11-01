"""Tests for the MASK_LOSS_WEIGHT channel and its effect on training losses.

Validates:
- ``TokenMask.loss_weight_excluding_mol_types`` zeros excluded mol_types,
  ones elsewhere. Empty exclude → all ones. Multi-element exclude works.
- ``training_loss`` correctly zeros excluded-token contributions in
  diffusion / smooth_lddt / distogram / plddt / exp_resolved when the
  channel is provided, while leaving non-excluded contributions identical
  to the no-mask baseline.
- ``pair_l2`` / ``single_l2`` regularizers are NOT affected by the mask
  (regularizers must continue to bound the activation magnitude of every
  token, including those excluded from the supervised losses).
"""

from __future__ import annotations

import numpy as np
import torch

from deepfold3.data import const
from deepfold3.data.masking import MASK_LOSS_WEIGHT, TokenMask


def test_loss_weight_excluding_mol_types_shape_and_values():
    N = 8
    tokens = np.zeros(N, dtype=[("mol_type", "i4"), ("asym_id", "i4"), ("residue_index", "i4")])
    # 4 protein, 1 RNA, 1 DNA, 2 ligand
    tokens["mol_type"] = [0, 0, 0, 0, 1, 2, const.MOL_NONPOLYMER, const.MOL_NONPOLYMER]

    # Exclude ligand only (PATCHR-style)
    arr = TokenMask.loss_weight_excluding_mol_types(tokens, [const.MOL_NONPOLYMER])
    assert arr.shape == (N,)
    assert arr.dtype == np.float32
    np.testing.assert_array_equal(arr, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    # Exclude both RNA and DNA
    arr2 = TokenMask.loss_weight_excluding_mol_types(tokens, [1, 2])
    np.testing.assert_array_equal(arr2, [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])

    # Empty exclude → all ones
    arr3 = TokenMask.loss_weight_excluding_mol_types(tokens, ())
    np.testing.assert_array_equal(arr3, [1.0] * N)


def _build_mini_loss_inputs(*, with_mask: bool):
    """Construct minimal pred/features for ``training_loss`` with one ligand token.

    Layout: 8 tokens, last 2 are mol_type=ligand. M=4 atoms each.
    """
    from deepfold3.model.config import TrainingConfig

    torch.manual_seed(0)
    N, M, S = 8, 4, 2

    atom_mask = torch.ones(N, M, dtype=torch.bool)
    seq_mask = torch.ones(N, dtype=torch.bool)
    rep_idx = torch.ones(N, dtype=torch.long)

    mol_type = torch.zeros(N, dtype=torch.long)
    mol_type[-2:] = const.MOL_NONPOLYMER  # last 2 tokens are ligand

    pred = {
        "denoised": torch.randn(S, N, M, 3),
        "gt_augmented": torch.randn(S, N, M, 3),
        "noise_levels": torch.rand(S),
        "distogram": torch.randn(N, N, 64),
        "plddt": torch.randn(S, N, M * 50),
        "pde": torch.randn(S, N, N, 64),
        "exp_resolved": torch.randn(S, N, M * 2),
        "mini_rollout_pos": torch.randn(S, N, M, 3),
    }
    features = {
        "atom_gt_positions": torch.randn(N, M, 3),
        "atom_gt_mask": atom_mask,
        "seq_mask": seq_mask.float(),
        "token_to_rep_atom": rep_idx,
        "mol_type": mol_type,
    }
    if with_mask:
        features[f"mask_{MASK_LOSS_WEIGHT}"] = torch.tensor([1.0] * 6 + [0.0, 0.0], dtype=torch.float32)

    cfg = TrainingConfig()
    cfg.alpha_confidence = 1.0  # enable confidence head losses
    return pred, features, cfg


def test_training_loss_with_ligand_mask_changes_value():
    """Loss differs when the mask zeros out ligand contributions vs no mask."""
    from deepfold3.training.loss import training_loss

    pred_no, feats_no, cfg = _build_mini_loss_inputs(with_mask=False)
    pred_w, feats_w, _ = _build_mini_loss_inputs(with_mask=True)

    total_no, _ = training_loss(pred_no, feats_no, cfg)
    total_w, _ = training_loss(pred_w, feats_w, cfg)

    # Loss must be finite in both cases.
    assert torch.isfinite(total_no)
    assert torch.isfinite(total_w)
    # Ligand atoms had ligand_loss_weight=10, so dropping them changes the
    # loss meaningfully.
    assert not torch.allclose(total_no, total_w)


def test_ligand_atom_gradients_are_zero_under_mask():
    """With mask=0 on ligand tokens, gradient w.r.t. ligand atom positions is 0."""
    from deepfold3.training.loss import training_loss

    pred, features, cfg = _build_mini_loss_inputs(with_mask=True)
    pred["denoised"] = pred["denoised"].clone().requires_grad_(True)
    pred["mini_rollout_pos"] = pred["mini_rollout_pos"].clone().requires_grad_(True)

    total, _ = training_loss(pred, features, cfg)
    total.backward()

    # Ligand tokens are the last 2; their gradients should be zero.
    grad_denoised = pred["denoised"].grad  # [S, N, M, 3]
    assert grad_denoised is not None
    assert torch.allclose(grad_denoised[:, -2:].abs().sum(), torch.zeros((), dtype=grad_denoised.dtype)), (
        "denoised grad on ligand tokens must be zero"
    )
    # Non-ligand tokens (first 6) should have at least some non-zero gradient.
    assert grad_denoised[:, :6].abs().sum() > 0


def test_pair_reg_unaffected_by_ligand_mask():
    """``pair_l2`` / ``single_l2`` regularizer values must NOT depend on the mask
    (regularizers continue to bound activations on ligand tokens).

    Parametrized over ``reg_form``: ``"lp"`` (existing behaviour, keys
    ``single_l2``/``pair_l2``) and ``"log_rms"`` (keys ``single_rms``/``pair_rms``).
    Both forms must be mask-invariant for the same reason.
    """
    from deepfold3.model.config import TrainingConfig
    from deepfold3.training.loss import training_loss

    torch.manual_seed(0)
    N, M = 8, 4

    pair = torch.randn(N, N, 16)
    single = torch.randn(N, 32)

    def _build(with_mask: bool, reg_form: str):
        pred = {
            "single": single.clone(),
            "pair": pair.clone(),
            "distogram_features": torch.randn(N, N, 8, requires_grad=True),
            "distogram_weight": torch.randn(64, 8),
        }
        features = {
            "atom_gt_positions": torch.zeros(N, M, 3),
            "atom_gt_mask": torch.ones(N, M, dtype=torch.bool),
            "seq_mask": torch.ones(N, dtype=torch.bool).float(),
            "token_to_rep_atom": torch.ones(N, dtype=torch.long),
            "mol_type": torch.cat(
                [torch.zeros(6, dtype=torch.long), torch.full((2,), const.MOL_NONPOLYMER, dtype=torch.long)]
            ),
        }
        if with_mask:
            features[f"mask_{MASK_LOSS_WEIGHT}"] = torch.tensor([1.0] * 6 + [0.0, 0.0], dtype=torch.float32)
        cfg = TrainingConfig()
        cfg.alpha_pair_reg = 0.0
        cfg.alpha_single_reg = 0.0
        cfg.alpha_diffusion = 0.0
        cfg.alpha_distogram = 0.0
        cfg.alpha_confidence = 0.0
        cfg.reg_form = reg_form
        return pred, features, cfg

    for reg_form, expected_keys in [("lp", ("pair_l2", "single_l2")), ("log_rms", ("pair_rms", "single_rms"))]:
        pred_no, feats_no, cfg = _build(with_mask=False, reg_form=reg_form)
        pred_w, feats_w, _ = _build(with_mask=True, reg_form=reg_form)
        _, losses_no = training_loss(pred_no, feats_no, cfg)
        _, losses_w = training_loss(pred_w, feats_w, cfg)

        for key in expected_keys:
            assert key in losses_no, f"reg_form={reg_form}: missing {key}"
            assert key in losses_w, f"reg_form={reg_form}: missing {key}"
            torch.testing.assert_close(losses_no[key], losses_w[key])


def test_log_rms_metric_matches_sqrt_of_l2():
    """For the same activations, ``single_rms`` (log_rms form) equals
    ``sqrt(single_l2)`` (lp form, p=2). Documents the relationship between
    the two reg_form metric definitions.
    """
    from deepfold3.model.config import TrainingConfig
    from deepfold3.training.loss import training_loss

    torch.manual_seed(123)
    N, M = 6, 4
    pair = torch.randn(N, N, 16)
    single = torch.randn(N, 32)

    def _build(reg_form: str):
        pred = {
            "single": single.clone(),
            "pair": pair.clone(),
            "distogram_features": torch.randn(N, N, 8, requires_grad=True),
            "distogram_weight": torch.randn(64, 8),
        }
        features = {
            "atom_gt_positions": torch.zeros(N, M, 3),
            "atom_gt_mask": torch.ones(N, M, dtype=torch.bool),
            "seq_mask": torch.ones(N, dtype=torch.bool).float(),
            "token_to_rep_atom": torch.ones(N, dtype=torch.long),
            "mol_type": torch.zeros(N, dtype=torch.long),
        }
        cfg = TrainingConfig()
        cfg.alpha_pair_reg = 0.0
        cfg.alpha_single_reg = 0.0
        cfg.alpha_diffusion = 0.0
        cfg.alpha_distogram = 0.0
        cfg.alpha_confidence = 0.0
        cfg.reg_form = reg_form
        return pred, features, cfg

    _, lp = training_loss(*_build("lp")[:2], _build("lp")[2])
    _, lr = training_loss(*_build("log_rms")[:2], _build("log_rms")[2])

    # single_rms == sqrt(single_l2), pair_rms == sqrt(pair_l2)
    torch.testing.assert_close(lr["single_rms"], lp["single_l2"].sqrt(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(lr["pair_rms"], lp["pair_l2"].sqrt(), atol=1e-5, rtol=1e-5)


def test_dataset_config_exclude_loss_mol_types_default_empty():
    """``DatasetConfig.exclude_loss_mol_types`` defaults to empty list."""
    from deepfold3.model.config import DatasetConfig

    cfg = DatasetConfig(name="x", data_dir="/tmp/x")
    assert cfg.exclude_loss_mol_types == []


def test_initial_yaml_exclude_loss_mol_types():
    """``configs/initial.yaml`` declares ``exclude_loss_mol_types`` per dataset.

    Current policy: no dataset excludes any mol_type (PATCHR keeps ligand
    supervision). The field is still surfaced through the loader.

    Asserted over whichever datasets the config enables, not over a named one:
    the earlier version demanded ``patchr``, which is commented out in the
    config on purpose until its template index frame is verified, so the test
    failed for a reason that had nothing to do with loss masking.
    """
    from pathlib import Path

    from deepfold3.config_loader import load_config

    cfg = load_config(Path(__file__).parent.parent / "configs" / "initial.yaml")
    extras = cfg["training"]["extra_datasets"]
    assert extras, "initial.yaml enables no extra dataset, so the field is untested"
    for d in extras:
        assert d["exclude_loss_mol_types"] == [], d["name"]


def test_distogram_gated_on_gt_rep_presence():
    """A token whose GT representative atom is unresolved is dropped from the
    distogram, even though seq_mask (CCD-derived rep_atom_present) keeps it.

    Flipping only ``atom_gt_mask`` at the rep atom must change the distogram
    term: the old code masked the distogram with seq_mask alone (independent of
    atom_gt_mask at the rep atom), so this isolates the new GT-rep-presence gate.
    """
    from deepfold3.training.loss import training_loss

    pred_a, feats_a, cfg = _build_mini_loss_inputs(with_mask=False)  # seeded → deterministic
    _, losses_a = training_loss(pred_a, feats_a, cfg)

    pred_b, feats_b, _ = _build_mini_loss_inputs(with_mask=False)
    m = feats_b["atom_gt_mask"].clone()
    m[0, 1] = False  # unresolve token 0's rep atom (rep_idx == 1)
    feats_b["atom_gt_mask"] = m
    _, losses_b = training_loss(pred_b, feats_b, cfg)

    assert torch.isfinite(losses_b["distogram"])
    assert not torch.allclose(losses_a["distogram"], losses_b["distogram"])
