"""Tests for training utilities: EMA and LR scheduler."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from deepfold3.model.config import TrainingConfig
from deepfold3.training.distributed import DistInfo
from deepfold3.training.ema import EMA
from deepfold3.training.scheduler import AF3LRScheduler, CosineAnnealingScheduler, WSDSScheduler, build_scheduler
from deepfold3.training.signals import ShutdownCoordinator


def _simple_model():
    torch.manual_seed(42)
    return nn.Linear(4, 2)


def _make_records(n: int):
    """Minimal valid Records for sampler tests — single-chain, integer cluster."""
    from deepfold3.data.types import ChainInfo, Record

    return [
        Record(
            id=f"id_{i}",
            chains=[
                ChainInfo(
                    chain_id=0,
                    chain_name="A",
                    mol_type=0,  # protein
                    cluster_id=i % 5,  # int cluster_id triggers WeightedClusterSampler path
                    msa_id=i,
                    num_residues=64,
                ),
            ],
            interfaces=[],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------------------


class TestEMA:
    def test_init_matches_model(self):
        model = _simple_model()
        ema = EMA(model, decay=0.999)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow
                assert torch.equal(ema.shadow[name], param.data)

    def test_update_moves_toward_model(self):
        model = _simple_model()
        ema = EMA(model, decay=0.999)
        old_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Change model params
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        ema.update(model)

        for name in ema.shadow:
            # Shadow should have moved toward the new param values
            old_dist = (old_shadow[name] - model.state_dict()[name]).norm()
            new_dist = (ema.shadow[name] - model.state_dict()[name]).norm()
            assert new_dist < old_dist

    def test_apply_restore(self):
        model = _simple_model()
        ema = EMA(model, decay=0.999)

        # Change model params so they differ from shadow
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update(model)

        original_params = {n: p.data.clone() for n, p in model.named_parameters()}
        shadow_before_apply = {k: v.clone() for k, v in ema.shadow.items()}

        # Apply EMA weights — swaps model params with shadow
        ema.apply(model)
        for name, param in model.named_parameters():
            if name in shadow_before_apply:
                assert torch.equal(param.data, shadow_before_apply[name])

        # Restore original weights — swaps back
        ema.restore(model)
        for name, param in model.named_parameters():
            assert torch.equal(param.data, original_params[name])

    def test_state_dict_roundtrip(self):
        model = _simple_model()
        ema = EMA(model, decay=0.99)
        ema.update(model)
        ema.update(model)

        state = ema.state_dict()

        ema2 = EMA(model, decay=0.5)  # different decay
        ema2.load_state_dict(state)

        assert ema2.decay == ema.decay
        assert ema2.num_updates == ema.num_updates
        for k in ema.shadow:
            assert torch.equal(ema2.shadow[k], ema.shadow[k])

    def test_load_state_dict_clones_tensors(self):
        model = _simple_model()
        ema = EMA(model, decay=0.999)

        state = ema.state_dict()
        original_tensor = next(iter(state["shadow"].values())).clone()

        ema2 = EMA(model, decay=0.999)
        ema2.load_state_dict(state)

        # Mutate the state dict shadow — EMA should be unaffected
        for v in state["shadow"].values():
            v.zero_()

        for _k, v in ema2.shadow.items():
            assert not torch.equal(v, torch.zeros_like(v)) or torch.equal(
                original_tensor, torch.zeros_like(original_tensor)
            )


# ---------------------------------------------------------------------------
# AF3LRScheduler tests
# ---------------------------------------------------------------------------


class TestAF3LRScheduler:
    def _make_scheduler(self, warmup=10, start_decay=20, decay_every=5, decay_factor=0.95, lr=1e-3):
        model = _simple_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        cfg = TrainingConfig(
            lr=lr,
            warmup_steps=warmup,
            start_decay_after=start_decay,
            decay_every=decay_every,
            decay_factor=decay_factor,
        )
        scheduler = AF3LRScheduler(optimizer, cfg)
        return optimizer, scheduler

    def test_step_zero(self):
        optimizer, scheduler = self._make_scheduler()
        # At step 0, warmup multiplier = 0/10 = 0
        lr = scheduler.get_last_lr()[0]
        assert lr == 0.0

    def test_warmup_phase(self):
        optimizer, scheduler = self._make_scheduler(warmup=10, lr=1e-3)
        lrs = []
        for _ in range(10):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()
        # LR should increase monotonically during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1], f"LR should increase during warmup: {lrs}"

    def test_plateau_phase(self):
        optimizer, scheduler = self._make_scheduler(warmup=10, start_decay=20, lr=1e-3)
        # Advance through warmup
        for _ in range(10):
            optimizer.step()
            scheduler.step()
        # During plateau (steps 10-19), LR should be constant at max_lr
        for _ in range(10):
            lr = scheduler.get_last_lr()[0]
            assert abs(lr - 1e-3) < 1e-9, f"LR should be max_lr during plateau: {lr}"
            optimizer.step()
            scheduler.step()

    def test_decay_phase(self):
        optimizer, scheduler = self._make_scheduler(warmup=10, start_decay=20, decay_every=5, decay_factor=0.5, lr=1e-3)
        # Advance to decay phase
        for _ in range(20):
            optimizer.step()
            scheduler.step()

        lr_at_decay_start = scheduler.get_last_lr()[0]
        assert abs(lr_at_decay_start - 1e-3) < 1e-9

        # After 5 more steps, should decay by factor 0.5
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        lr_after_first_decay = scheduler.get_last_lr()[0]
        assert abs(lr_after_first_decay - 0.5e-3) < 1e-9, f"Expected 0.5e-3, got {lr_after_first_decay}"


# ---------------------------------------------------------------------------
# WSDSScheduler tests
# ---------------------------------------------------------------------------


class TestWSDSScheduler:
    def _make_scheduler(self, *, warmup=5, windows=None, min_ratio=0.01, lr=1e-3):
        model = _simple_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        cfg = TrainingConfig(
            lr=lr,
            warmup_steps=warmup,
            wsds_decay_windows=windows or [],
            wsds_min_lr_ratio=min_ratio,
        )
        scheduler = WSDSScheduler(optimizer, cfg)
        return optimizer, scheduler

    def _step_to(self, optimizer, scheduler, target_step: int) -> float:
        """Drive the optimizer to ``target_step`` and return current LR."""
        # scheduler.last_epoch is the last completed step. We step until it
        # equals target_step.
        while scheduler.last_epoch < target_step:
            optimizer.step()
            scheduler.step()
        return scheduler.get_last_lr()[0]

    def test_warmup_then_stable(self):
        optimizer, scheduler = self._make_scheduler(warmup=10, windows=[[100, 110]], lr=1e-3)
        # Mid warmup
        lr5 = self._step_to(optimizer, scheduler, 5)
        assert abs(lr5 - 5e-4) < 1e-9
        # End of warmup → stable
        lr10 = self._step_to(optimizer, scheduler, 10)
        assert abs(lr10 - 1e-3) < 1e-9
        # Stable phase before decay window
        lr50 = self._step_to(optimizer, scheduler, 50)
        assert abs(lr50 - 1e-3) < 1e-9

    def test_decay_endpoints(self):
        # Window [100, 110), 10 steps
        optimizer, scheduler = self._make_scheduler(warmup=5, windows=[[100, 110]], min_ratio=0.1, lr=1e-3)
        lr_start = self._step_to(optimizer, scheduler, 100)  # s=0 → max_lr
        assert abs(lr_start - 1e-3) < 1e-9, f"At s=0 lr should be max: {lr_start}"
        lr_end = self._step_to(optimizer, scheduler, 109)  # s=9, L=10
        # 1/lr = 0.9*10 + 0.1 = 9.1 → lr ≈ 1e-3 / 9.1 ≈ 1.0989e-4
        assert abs(lr_end - 1e-3 / 9.1) < 1e-7, f"Near-end lr: {lr_end}"

    def test_resume_after_decay(self):
        optimizer, scheduler = self._make_scheduler(warmup=5, windows=[[100, 110]], min_ratio=0.1, lr=1e-3)
        # Step 110: past the decay window → back to max_lr (no rollback)
        lr110 = self._step_to(optimizer, scheduler, 110)
        assert abs(lr110 - 1e-3) < 1e-9, f"Post-decay should resume at max_lr: {lr110}"

    def test_multiple_windows(self):
        optimizer, scheduler = self._make_scheduler(warmup=5, windows=[[100, 110], [200, 210]], min_ratio=0.5, lr=1e-3)
        # Stable between the two windows
        lr150 = self._step_to(optimizer, scheduler, 150)
        assert abs(lr150 - 1e-3) < 1e-9
        # Mid-decay of second window: s=5, L=10, u=0.5
        # 1/lr_mult = 0.5/0.5 + 0.5 = 1.5 → lr = 1e-3 / 1.5
        lr205 = self._step_to(optimizer, scheduler, 205)
        assert abs(lr205 - 1e-3 / 1.5) < 1e-9, f"Mid-decay of 2nd window: {lr205}"

    def test_inverse_proportional_form(self):
        # 1/lr should be linear-in-step within a decay window
        optimizer, scheduler = self._make_scheduler(warmup=0, windows=[[0, 100]], min_ratio=0.01, lr=1.0)
        # collect LR at steps 0, 25, 50, 75 → 1/lr should be linear
        lrs = []
        for step in (0, 25, 50, 75):
            lrs.append(self._step_to(optimizer, scheduler, step))
        inv_lrs = [1.0 / lr for lr in lrs]
        # diffs of 1/lr should be (approximately) constant
        diffs = [inv_lrs[i + 1] - inv_lrs[i] for i in range(len(inv_lrs) - 1)]
        for i in range(1, len(diffs)):
            assert abs(diffs[i] - diffs[0]) < 1e-9, f"Inverse-LR not linear: {diffs}"

    def test_overlapping_windows_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="overlap"):
            self._make_scheduler(windows=[[100, 200], [150, 250]])

    def test_invalid_window_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="must be > start"):
            self._make_scheduler(windows=[[200, 100]])  # end < start

    def test_invalid_ratio_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="min_lr_ratio"):
            self._make_scheduler(windows=[[100, 200]], min_ratio=1.5)

    def test_build_scheduler_dispatches_wsds(self):
        model = _simple_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        cfg = TrainingConfig(
            lr=1e-3,
            finetune_scheduler="wsds",
            warmup_steps=5,
            wsds_decay_windows=[[100, 110]],
            wsds_min_lr_ratio=0.1,
        )
        sched = build_scheduler(optimizer, cfg)
        assert isinstance(sched, WSDSScheduler)

    def test_build_scheduler_unknown_name_rejected(self):
        import pytest

        model = _simple_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        cfg = TrainingConfig(lr=1e-3, finetune_scheduler="bogus")
        with pytest.raises(ValueError, match="finetune_scheduler"):
            build_scheduler(optimizer, cfg)


# ---------------------------------------------------------------------------
# AdamWScheduleFree tests
# ---------------------------------------------------------------------------


class TestAdamWScheduleFree:
    def _build(self, *, lr=1e-3, warmup=0, weight_decay=0.0, betas=(0.9, 0.999)):
        from deepfold3.training.schedule_free import AdamWScheduleFree

        model = _simple_model()
        optimizer = AdamWScheduleFree(
            model.parameters(),
            lr=lr,
            betas=betas,
            warmup_steps=warmup,
            weight_decay=weight_decay,
        )
        return model, optimizer

    @staticmethod
    def _set_grads(model, value=0.1):
        for p in model.parameters():
            p.grad = torch.full_like(p, value)

    def test_step_requires_train_mode(self):
        import pytest

        _, optimizer = self._build()
        # Default group['train_mode'] is False in __init__.
        with pytest.raises(RuntimeError, match="train mode"):
            optimizer.step()

    def test_train_eval_swap_round_trip(self):
        model, optimizer = self._build(lr=1e-2, warmup=0)
        optimizer.train()
        # Two steps: first step has ckp1 = 1.0 (= 1/(k+1) with weight_lr_power
        # giving constant weight) so x = z = y degenerately; second step gives
        # ckp1 = 0.5 and a non-trivial average.
        for _ in range(2):
            self._set_grads(model)
            optimizer.step()

        # Snapshot y at this iterate (live param after step).
        y_snapshot = [p.detach().clone() for p in model.parameters()]

        optimizer.eval()
        # Now p should equal x for each param. After 2 steps, x ≠ y.
        x_snapshot = [p.detach().clone() for p in model.parameters()]
        any_differ = any(not torch.allclose(x, y) for x, y in zip(x_snapshot, y_snapshot, strict=True))
        assert any_differ, "x should differ from y after multiple steps"

        optimizer.train()
        # Now back to y — must match the pre-eval snapshot bit-exactly (lerp is
        # invertible: train→eval→train is the identity on p).
        for p, y in zip(model.parameters(), y_snapshot, strict=True):
            assert torch.allclose(p, y, atol=1e-6), "train→eval→train should restore y"

    def test_extract_x_matches_eval_swap(self):
        model, optimizer = self._build(lr=1e-2)
        optimizer.train()
        self._set_grads(model)
        optimizer.step()

        # Materialize x via the helper (no live mutation).
        x_helper = [optimizer.extract_x(p).detach().clone() for p in model.parameters()]
        # Live snapshot of y (extract_x must not have changed live params).
        y_after_helper = [p.detach().clone() for p in model.parameters()]

        # Now compare against eval-then-read.
        optimizer.eval()
        for p, x_h in zip(model.parameters(), x_helper, strict=True):
            assert torch.allclose(p, x_h, atol=1e-6)
        optimizer.train()

        # Confirm extract_x was non-destructive.
        for p, y in zip(model.parameters(), y_after_helper, strict=True):
            assert torch.allclose(p, y, atol=1e-6)

    def test_state_dict_roundtrip_resume(self):
        # Two parallel runs: one continues straight, the other saves+loads
        # mid-way. Final params must agree.
        model_a, opt_a = self._build(lr=1e-2)
        model_b, opt_b = self._build(lr=1e-2)
        # Sync init weights so the two runs start identically.
        with torch.no_grad():
            for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True):
                pb.copy_(pa)
        opt_a.train()
        opt_b.train()

        for _ in range(5):
            self._set_grads(model_a, value=0.1)
            self._set_grads(model_b, value=0.1)
            opt_a.step()
            opt_b.step()

        # Save B inside eval mode (matches CheckpointManager.save behavior).
        opt_b.eval()
        b_state = opt_b.state_dict()
        b_model_state = {k: v.detach().clone() for k, v in model_b.state_dict().items()}
        opt_b.train()

        # Fresh model_c + opt_c reconstructed from B's saved state.
        from deepfold3.training.schedule_free import AdamWScheduleFree

        model_c = _simple_model()
        # Random init — overwrite with saved x (= what was in model_b at eval-save).
        model_c.load_state_dict(b_model_state)
        opt_c = AdamWScheduleFree(model_c.parameters(), lr=1e-2)
        opt_c.load_state_dict(b_state)
        opt_c.train()  # lerp p ← y

        # Continue 5 more steps for both A and C.
        for _ in range(5):
            self._set_grads(model_a, value=0.1)
            self._set_grads(model_c, value=0.1)
            opt_a.step()
            opt_c.step()

        for pa, pc in zip(model_a.parameters(), model_c.parameters(), strict=True):
            assert torch.allclose(pa, pc, atol=1e-5), "save+load round-trip diverged"

    def test_warmup_lr_curve(self):
        # Paper Algorithm 1 line 8: γ_t = γ · √(1−β₂^t) · min(1, t/T_warm).
        # Use β₂ = 0.0 so the bias correction term is identically 1 for
        # t≥1 and we can verify the warmup factor in isolation. With β₂=0
        # the second moment is just g² (no momentum) — fine for testing.
        model, optimizer = self._build(lr=1e-2, warmup=5, betas=(0.9, 0.0))
        optimizer.train()
        observed = []
        for _ in range(8):
            self._set_grads(model)
            optimizer.step()
            observed.append(optimizer.param_groups[0]["scheduled_lr"])
        expected_warmup = [(i + 1) / 5 * 1e-2 for i in range(5)]
        for got, exp in zip(observed[:5], expected_warmup, strict=True):
            assert abs(got - exp) < 1e-12, f"warmup lr curve: got {got}, expected {exp}"
        # Steps 5+: flat at lr.
        for got in observed[5:]:
            assert abs(got - 1e-2) < 1e-12

    def test_lr_includes_bias_correction(self):
        # γ_t / γ at step t with sched = 1.0 must equal √(1 − β₂^t).
        model, optimizer = self._build(lr=1.0, warmup=0, betas=(0.9, 0.95))
        optimizer.train()
        for t in range(1, 10):
            self._set_grads(model)
            optimizer.step()
            expected = (1 - 0.95**t) ** 0.5
            got = optimizer.param_groups[0]["scheduled_lr"]
            assert abs(got - expected) < 1e-12, f"step {t}: got {got}, expected {expected}"

    def test_skip_step_does_not_advance_iterates(self):
        model, optimizer = self._build(lr=1e-2)
        optimizer.train()
        # One real step to materialize state (z, exp_avg_sq, k=1).
        self._set_grads(model)
        optimizer.step()
        z_before = [optimizer.state[p]["z"].detach().clone() for p in model.parameters()]
        v_before = [optimizer.state[p]["exp_avg_sq"].detach().clone() for p in model.parameters()]
        k_before = optimizer.param_groups[0]["k"]

        # Simulate WrappedOptimizer skip path: zero grads, do NOT call step.
        for p in model.parameters():
            p.grad = None

        # State must be unchanged.
        for p, z in zip(model.parameters(), z_before, strict=True):
            assert torch.equal(optimizer.state[p]["z"], z)
        for p, v in zip(model.parameters(), v_before, strict=True):
            assert torch.equal(optimizer.state[p]["exp_avg_sq"], v)
        assert optimizer.param_groups[0]["k"] == k_before

    def test_build_optimizer_dispatches_schedule_free(self):
        from deepfold3.training.optim import build_optimizer
        from deepfold3.training.schedule_free import AdamWScheduleFree

        model = _simple_model()
        cfg = TrainingConfig(lr=1e-3, optimizer_type="schedule_free", warmup_steps=10)
        optimizer = build_optimizer(model, cfg)
        assert isinstance(optimizer, AdamWScheduleFree)
        # build_optimizer must leave SF in train mode (step() requires it).
        assert optimizer.param_groups[0]["train_mode"] is True

    def test_identity_scheduler_for_schedule_free(self):
        from torch.optim.lr_scheduler import LambdaLR

        from deepfold3.training.scheduler import build_scheduler

        model = _simple_model()
        cfg = TrainingConfig(lr=1e-3, optimizer_type="schedule_free", warmup_steps=10)
        # Use a real SF optimizer so the scheduler can attach.
        from deepfold3.training.optim import build_optimizer

        optimizer = build_optimizer(model, cfg)
        sched = build_scheduler(optimizer, cfg)
        assert isinstance(sched, LambdaLR)
        # Identity multiplier: lr unchanged across many .step() calls.
        before = optimizer.param_groups[0]["lr"]
        for _ in range(20):
            sched.step()
        assert abs(optimizer.param_groups[0]["lr"] - before) < 1e-12

    def test_optimizer_eval_mode_noop_for_adamw(self):
        # The free helper must be safe to call on plain AdamW (no eval/train methods).
        from deepfold3.training.schedule_free import optimizer_eval_mode

        model = _simple_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        snapshot = [p.detach().clone() for p in model.parameters()]
        with optimizer_eval_mode(optimizer):
            for p, s in zip(model.parameters(), snapshot, strict=True):
                assert torch.equal(p, s)
        for p, s in zip(model.parameters(), snapshot, strict=True):
            assert torch.equal(p, s)


class TestCheckpointResume:
    """Round-trip: epoch + RNG state + scaler are persisted and restored.

    Single-rank only — multi-rank gather/scatter needs an MPI launcher,
    which we don't run in unit tests.
    """

    def _make_world(self, tmp_path, *, optimizer_type="adamw"):
        """Build a minimal model+optim+scheduler+state and a CheckpointManager."""
        import random as _random

        import numpy as np

        from deepfold3.model.config import ModelConfig, TrainingConfig
        from deepfold3.training.checkpointing import CheckpointManager
        from deepfold3.training.distributed import DistInfo
        from deepfold3.training.optim import build_optimizer
        from deepfold3.training.scheduler import build_scheduler
        from deepfold3.training.state import TrainState

        cfg = TrainingConfig(lr=1e-3, optimizer_type=optimizer_type, warmup_steps=10)
        model_cfg = ModelConfig()
        model = _simple_model()
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)
        state = TrainState()
        state.global_step = 200
        state.effective_step = 100
        state.epoch = 4
        state.best_eval_loss = 0.5
        ckpt_mgr = CheckpointManager(max_checkpoints=0, strict=True, device=torch.device("cpu"))
        dist_info = DistInfo(rank=0, world_size=1, local_rank=0, device=torch.device("cpu"))

        # Burn some RNG draws so each source has a non-default state.
        torch.randn(7)
        np.random.rand(11)
        _random.random()

        return cfg, model_cfg, model, optimizer, scheduler, state, ckpt_mgr, dist_info

    def test_round_trip_persists_epoch_and_rng(self, tmp_path):
        """epoch + python/numpy/torch RNG states round-trip through save/load."""
        import random as _random

        import numpy as np

        cfg, model_cfg, model, optimizer, scheduler, state, ckpt_mgr, dist_info = self._make_world(tmp_path)
        path = tmp_path / "ckpt.pt"

        # Snapshot the RNG state we expect to be preserved.
        torch_state_pre = torch.get_rng_state().clone()
        py_state_pre = _random.getstate()
        np_state_pre = np.random.get_state()

        ckpt_mgr.save(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=None,
            scaler=None,
            state=state,
            model_cfg=model_cfg,
            train_cfg=cfg,
            dist_info=dist_info,
        )

        # Burn the RNGs so state diverges before load.
        for _ in range(10):
            torch.randn(3)
            np.random.rand(5)
            _random.random()
        from deepfold3.training.state import TrainState

        new_state = TrainState()
        ckpt_mgr.load(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=None,
            scaler=None,
            state=new_state,
            train_cfg=cfg,
            dist_info=dist_info,
        )
        assert new_state.epoch == 4
        assert new_state.effective_step == 100
        assert new_state.global_step == 200
        # RNG state restored — first draw should match what would have
        # come next at save time.
        torch.set_rng_state(torch_state_pre)
        expected_torch = torch.randn(3)
        ckpt_mgr.load(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=None,
            scaler=None,
            state=new_state,
            train_cfg=cfg,
            dist_info=dist_info,
        )
        actual_torch = torch.randn(3)
        assert torch.equal(expected_torch, actual_torch)

        _random.setstate(py_state_pre)
        expected_py = _random.random()
        ckpt_mgr.load(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=None,
            scaler=None,
            state=new_state,
            train_cfg=cfg,
            dist_info=dist_info,
        )
        actual_py = _random.random()
        assert expected_py == actual_py

        np.random.set_state(np_state_pre)
        expected_np = np.random.rand(5)
        ckpt_mgr.load(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=None,
            scaler=None,
            state=new_state,
            train_cfg=cfg,
            dist_info=dist_info,
        )
        actual_np = np.random.rand(5)
        assert (expected_np == actual_np).all()

    def test_old_checkpoint_derives_epoch_from_steps(self, tmp_path):
        """A blob without ``epoch`` / ``rng_states`` falls back gracefully."""
        from deepfold3.training.state import TrainState

        cfg, model_cfg, model, optimizer, scheduler, state, ckpt_mgr, dist_info = self._make_world(tmp_path)
        path = tmp_path / "old.pt"

        # Hand-rolled "old format" blob — no epoch / rng_states / scaler keys.
        old_blob = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": 1300 * 2,
            "effective_step": 1300,
            "best_eval_loss": 1.0,
            "model_cfg": model_cfg,
            "train_cfg": cfg,
        }
        torch.save(old_blob, path)

        new_state = TrainState()
        # steps_per_epoch=391 mimics epoch_size=100000 / effective_batch=256.
        ckpt_mgr.load(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=None,
            scaler=None,
            state=new_state,
            train_cfg=cfg,
            dist_info=dist_info,
            steps_per_epoch=391,
        )
        assert new_state.effective_step == 1300
        assert new_state.epoch == 1300 // 391  # == 3

    def test_sampler_iter_position_round_trip(self, tmp_path):
        """Sampler ``sample_idx`` is captured + restored, so resume yields the tail.

        The end-to-end check: yield N samples, snapshot, build a fresh
        sampler with the same seed/epoch, ``load_state_dict``, then
        yield the rest — total sequence must equal a single uncut iter.
        """
        from deepfold3.data.sample.weighted import WeightedClusterSampler

        records = _make_records(30)
        sampler_a = WeightedClusterSampler(records=records, epoch_size=20, num_replicas=1, rank=0, seed=7)
        sampler_a.set_epoch(2)

        full = list(iter(sampler_a))  # consume entire epoch
        assert len(full) == 20

        # Re-run, stop midway, save state, hand off to a fresh sampler.
        sampler_b = WeightedClusterSampler(records=records, epoch_size=20, num_replicas=1, rank=0, seed=7)
        sampler_b.set_epoch(2)
        prefix = []
        for i, idx in enumerate(iter(sampler_b)):
            prefix.append(idx)
            if i + 1 == 7:  # stop after 7 samples
                break
        snap = sampler_b.state_dict()
        assert snap["sample_idx"] == 7

        sampler_c = WeightedClusterSampler(records=records, epoch_size=20, num_replicas=1, rank=0, seed=7)
        sampler_c.load_state_dict(snap)
        # Trainer always calls set_epoch before iter — must be a no-op
        # when epoch matches the loaded value.
        sampler_c.set_epoch(snap["epoch"])
        tail = list(iter(sampler_c))

        assert prefix + tail == full
        assert sampler_c.sample_idx == 20  # exhausted

    def test_sampler_set_epoch_resets_idx_only_on_change(self, tmp_path):
        """``set_epoch(e)`` resets ``sample_idx`` only when ``e`` changes."""
        from deepfold3.data.sample.weighted import WeightedClusterSampler

        records = _make_records(20)
        sampler = WeightedClusterSampler(records=records, epoch_size=10, num_replicas=1, rank=0, seed=1)
        sampler.set_epoch(5)
        sampler.sample_idx = 4  # simulate having yielded 4 samples
        sampler.set_epoch(5)  # same epoch — must NOT reset
        assert sampler.sample_idx == 4
        sampler.set_epoch(6)  # new epoch — resets
        assert sampler.sample_idx == 0

    def test_full_round_trip_through_checkpoint_manager(self, tmp_path):
        """End-to-end: sampler state goes through CheckpointManager save/load."""
        from deepfold3.data.sample.weighted import WeightedClusterSampler
        from deepfold3.training.state import TrainState

        records = _make_records(30)
        cfg, model_cfg, model, optimizer, scheduler, state, ckpt_mgr, dist_info = self._make_world(tmp_path)
        sampler = WeightedClusterSampler(records=records, epoch_size=20, num_replicas=1, rank=0, seed=11)
        sampler.set_epoch(2)
        # Consume 6 of 20.
        it = iter(sampler)
        for _ in range(6):
            next(it)
        assert sampler.sample_idx == 6

        path = tmp_path / "ckpt.pt"
        ckpt_mgr.save(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=None,
            scaler=None,
            state=state,
            model_cfg=model_cfg,
            train_cfg=cfg,
            dist_info=dist_info,
            train_sampler=sampler,
        )

        # Fresh sampler, same construction args, restore via the manager.
        sampler2 = WeightedClusterSampler(records=records, epoch_size=20, num_replicas=1, rank=0, seed=11)
        new_state = TrainState()
        ckpt_mgr.load(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=None,
            scaler=None,
            state=new_state,
            train_cfg=cfg,
            dist_info=dist_info,
            train_sampler=sampler2,
        )
        assert sampler2.epoch == 2
        assert sampler2.sample_idx == 6

    def test_multi_dataset_sampler_iter_position_round_trip(self, tmp_path):
        """``MultiDatasetSampler.sample_idx`` is captured + restored across resume.

        Mirror of ``test_sampler_iter_position_round_trip`` for the
        multi-dataset path that the trainer takes when extra datasets
        (e.g. PDB + distillation) are configured.
        """
        from deepfold3.data.sample.multi import MultiDatasetSampler

        records = _make_records(30)
        configs = [{"records": records, "dataset_len": 30, "weight": 1.0, "strategy": "uniform"}]

        sampler_a = MultiDatasetSampler(dataset_configs=configs, epoch_size=20, num_replicas=1, rank=0, seed=7)
        sampler_a.set_epoch(2)
        full = list(iter(sampler_a))
        assert len(full) == 20

        sampler_b = MultiDatasetSampler(dataset_configs=configs, epoch_size=20, num_replicas=1, rank=0, seed=7)
        sampler_b.set_epoch(2)
        prefix = []
        for i, idx in enumerate(iter(sampler_b)):
            prefix.append(idx)
            if i + 1 == 7:
                break
        snap = sampler_b.state_dict()
        assert snap["sample_idx"] == 7
        assert snap["epoch"] == 2
        assert snap["seed"] == 7

        sampler_c = MultiDatasetSampler(dataset_configs=configs, epoch_size=20, num_replicas=1, rank=0, seed=7)
        sampler_c.load_state_dict(snap)
        sampler_c.set_epoch(snap["epoch"])
        tail = list(iter(sampler_c))

        assert prefix + tail == full
        assert sampler_c.sample_idx == 20

    def test_multi_dataset_sampler_set_epoch_resets_idx_only_on_change(self, tmp_path):
        """``set_epoch`` only resets ``sample_idx`` when epoch number actually changes."""
        from deepfold3.data.sample.multi import MultiDatasetSampler

        records = _make_records(20)
        configs = [{"records": records, "dataset_len": 20, "weight": 1.0, "strategy": "uniform"}]
        sampler = MultiDatasetSampler(dataset_configs=configs, epoch_size=10, num_replicas=1, rank=0, seed=1)
        sampler.set_epoch(5)
        sampler.sample_idx = 4
        sampler.set_epoch(5)  # same epoch — keep
        assert sampler.sample_idx == 4
        sampler.set_epoch(6)  # new epoch — reset
        assert sampler.sample_idx == 0

    def test_multi_dataset_sampler_infinite_crosses_chunk_boundary(self):
        """Infinite iterator never exhausts and reproduces the finite per-chunk draws.

        The first ``epoch_size`` indices of the infinite stream must equal the
        finite epoch-0 shard, and the next ``epoch_size`` must equal the finite
        epoch-1 shard — so workers prefetch across the boundary with no drain
        while the sampling distribution is unchanged.
        """
        from deepfold3.data.sample.multi import MultiDatasetSampler

        records = _make_records(40)
        configs = [{"records": records, "dataset_len": 40, "weight": 1.0, "strategy": "uniform"}]

        finite = MultiDatasetSampler(dataset_configs=configs, epoch_size=10, num_replicas=1, rank=0, seed=3)
        finite.set_epoch(0)
        chunk0 = list(iter(finite))
        finite.set_epoch(1)
        chunk1 = list(iter(finite))

        inf = MultiDatasetSampler(dataset_configs=configs, epoch_size=10, num_replicas=1, rank=0, seed=3, infinite=True)
        it = iter(inf)
        stream = [next(it) for _ in range(20)]  # spans two chunks, never raises

        assert stream[:10] == chunk0
        assert stream[10:] == chunk1
        assert inf.epoch == 1  # advanced as a logging/resume counter

    def test_multi_dataset_sampler_infinite_resume_across_chunk(self):
        """Resume mid-stream in infinite mode continues the exact sequence.

        Captures state after the chunk boundary, restores into a fresh sampler,
        and checks the tail matches the uninterrupted stream — the resume
        protocol the trainer relies on for the multi-dataset path.
        """
        from deepfold3.data.sample.multi import MultiDatasetSampler

        records = _make_records(40)
        configs = [{"records": records, "dataset_len": 40, "weight": 1.0, "strategy": "uniform"}]

        full = MultiDatasetSampler(dataset_configs=configs, epoch_size=8, num_replicas=1, rank=0, seed=5, infinite=True)
        it = iter(full)
        ref = [next(it) for _ in range(20)]

        a = MultiDatasetSampler(dataset_configs=configs, epoch_size=8, num_replicas=1, rank=0, seed=5, infinite=True)
        it_a = iter(a)
        prefix = [next(it_a) for _ in range(11)]  # 8 + 3 → mid second chunk
        snap = a.state_dict()
        assert snap["epoch"] == 1
        assert snap["sample_idx"] == 3

        b = MultiDatasetSampler(dataset_configs=configs, epoch_size=8, num_replicas=1, rank=0, seed=5, infinite=True)
        b.load_state_dict(snap)
        b.set_epoch(snap["epoch"])  # trainer calls this; same epoch → no reset
        it_b = iter(b)
        tail = [next(it_b) for _ in range(9)]

        assert prefix + tail == ref

    def test_multi_dataset_sampler_finite_checkpoint_resumes_infinite_identically(self):
        """Cross-version resume: a checkpoint saved by the OLD finite sampler
        continues bit-identically under the NEW infinite sampler.

        The live arms were checkpointed under ``infinite=False``; resuming them
        under ``infinite=True`` must not perturb the sample stream. ``_draw_shard``
        is shared and the ``{epoch, sample_idx, seed}`` schema is unchanged, so
        the continuation from (epoch, sample_idx) is identical — only the
        epoch-boundary prefetch behaviour differs.
        """
        from deepfold3.data.sample.multi import MultiDatasetSampler

        records = _make_records(40)
        configs = [{"records": records, "dataset_len": 40, "weight": 1.0, "strategy": "uniform"}]

        # Old run: finite sampler, mid-epoch-E checkpoint at sample_idx I.
        old = MultiDatasetSampler(
            dataset_configs=configs,
            epoch_size=6,
            num_replicas=1,
            rank=0,
            seed=9,  # infinite=False
        )
        old.set_epoch(2)
        it_old = iter(old)
        consumed = [next(it_old) for _ in range(4)]  # mid epoch 2
        snap = old.state_dict()
        assert snap == {"epoch": 2, "sample_idx": 4, "seed": 9}
        rest_of_epoch2 = list(it_old)  # finite: drains to end of epoch 2

        # What the finite run would draw next (epoch 3) after the trainer's
        # while-level epoch increment + set_epoch reset.
        old.set_epoch(3)
        epoch3 = list(iter(old))

        # New run: same checkpoint, resumed under the infinite sampler.
        new = MultiDatasetSampler(dataset_configs=configs, epoch_size=6, num_replicas=1, rank=0, seed=9, infinite=True)
        new.load_state_dict(snap)
        new.set_epoch(snap["epoch"])
        it_new = iter(new)
        cont = [next(it_new) for _ in range(len(rest_of_epoch2) + len(epoch3))]

        # Identical continuation across the version switch and the E→E+1 boundary.
        assert cont == rest_of_epoch2 + epoch3
        assert consumed + cont == consumed + rest_of_epoch2 + epoch3

    def test_multi_dataset_sampler_full_round_trip_through_checkpoint_manager(self, tmp_path):
        """End-to-end: MultiDatasetSampler state goes through CheckpointManager save/load."""
        from deepfold3.data.sample.multi import MultiDatasetSampler
        from deepfold3.training.state import TrainState

        records = _make_records(30)
        configs = [{"records": records, "dataset_len": 30, "weight": 1.0, "strategy": "uniform"}]

        cfg, model_cfg, model, optimizer, scheduler, state, ckpt_mgr, dist_info = self._make_world(tmp_path)
        sampler = MultiDatasetSampler(dataset_configs=configs, epoch_size=20, num_replicas=1, rank=0, seed=11)
        sampler.set_epoch(2)
        it = iter(sampler)
        for _ in range(6):
            next(it)
        assert sampler.sample_idx == 6

        path = tmp_path / "ckpt_multi.pt"
        ckpt_mgr.save(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=None,
            scaler=None,
            state=state,
            model_cfg=model_cfg,
            train_cfg=cfg,
            dist_info=dist_info,
            train_sampler=sampler,
        )

        sampler2 = MultiDatasetSampler(dataset_configs=configs, epoch_size=20, num_replicas=1, rank=0, seed=11)
        new_state = TrainState()
        ckpt_mgr.load(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=None,
            scaler=None,
            state=new_state,
            train_cfg=cfg,
            dist_info=dist_info,
            train_sampler=sampler2,
        )
        assert sampler2.epoch == 2
        assert sampler2.sample_idx == 6

    def test_sf_invariant_violation_raises(self, tmp_path):
        """Loading an SF blob whose optimizer state has train_mode=True must error."""
        import pytest

        from deepfold3.training.state import TrainState

        cfg, model_cfg, model, optimizer, scheduler, state, ckpt_mgr, dist_info = self._make_world(
            tmp_path, optimizer_type="schedule_free"
        )
        path = tmp_path / "bad.pt"
        ckpt_mgr.save(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=None,
            scaler=None,
            state=state,
            model_cfg=model_cfg,
            train_cfg=cfg,
            dist_info=dist_info,
        )
        # Tamper with the saved blob: flip train_mode to True.
        blob = torch.load(path, weights_only=False)
        for g in blob["optimizer"]["param_groups"]:
            g["train_mode"] = True
        torch.save(blob, path)

        new_state = TrainState()
        with pytest.raises(RuntimeError, match="train_mode=True"):
            ckpt_mgr.load(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=None,
                scaler=None,
                state=new_state,
                train_cfg=cfg,
                dist_info=dist_info,
            )


def _make_anchor_records():
    """Heteromeric, multi-anchor records with varied cluster sizes.

    Cluster "A" is shared across 4 chains (downweighted); "B"/"C" are unique.
    Interfaces {A,C} recur (het2 has two), so their interface-cluster size is 2.
    """
    from deepfold3.data.types import ChainInfo, InterfaceInfo, Record

    def _rec(rid, chains, ifaces):
        return Record(
            id=rid,
            chains=[
                ChainInfo(chain_id=c, chain_name=chr(65 + c), mol_type=0, cluster_id=cl, msa_id=c, num_residues=64)
                for (c, cl) in chains
            ],
            interfaces=[InterfaceInfo(chain_1=a, chain_2=b) for (a, b) in ifaces],
        )

    return [
        _rec("het1", [(0, "A"), (1, "B")], [(0, 1)]),
        _rec("het2", [(0, "A"), (1, "A"), (2, "C")], [(0, 2), (1, 2)]),
        _rec("mono", [(0, "A")], []),
    ]


class TestTwoStageAnchorSampling:
    """AF3 §2.5.1 two-stage decomposition: record marginal × anchor conditional."""

    def test_chain_rule_joint_matches_flat_table(self):
        """p(R)·p(a|R) reproduces the flat per-anchor joint p(R,a) ∝ w_a exactly."""
        from deepfold3.data.sample.weighted import build_record_anchor_tables, compute_cluster_weights

        records = _make_anchor_records()
        tables = build_record_anchor_tables(records)
        p_record = compute_cluster_weights(records)

        # Flat reference: normalise all per-anchor weights globally.
        flat = {(ri, a): wa for ri, (anchors, ws) in tables.items() for a, wa in zip(anchors, ws, strict=True)}
        total = sum(flat.values())
        flat_joint = {k: v / total for k, v in flat.items()}

        # Two-stage: marginal times per-record conditional.
        two_stage = {}
        for ri, (anchors, ws) in tables.items():
            s = float(sum(ws))
            for a, wa in zip(anchors, ws, strict=True):
                two_stage[(ri, a)] = p_record[ri] * (wa / s)

        assert set(flat_joint) == set(two_stage)
        assert max(abs(flat_joint[k] - two_stage[k]) for k in flat_joint) < 1e-12

    def test_conditional_is_cluster_weighted(self):
        """Within het2, the unique-cluster chain outweighs the shared-cluster ones."""
        from deepfold3.data.sample.weighted import build_record_anchor_tables

        records = _make_anchor_records()
        anchors, weights = build_record_anchor_tables(records)[1]  # het2
        w = dict(zip(anchors, weights, strict=True))
        # chain 2 (cluster "C", size 1) vs chains 0/1 (cluster "A", size 4)
        assert w[("chain", 2, -1)] > w[("chain", 0, -1)]
        assert w[("chain", 0, -1)] == w[("chain", 1, -1)]
        # interfaces (beta_interface=1.0, two mol types) dominate a single chain
        assert w[("interface", 0, 2)] > w[("chain", 2, -1)]

    def test_records_without_anchors_are_omitted(self):
        """A record with no valid chain contributes no table entry (cropper self-seeds)."""
        from deepfold3.data.sample.weighted import build_record_anchor_tables
        from deepfold3.data.types import ChainInfo, Record

        records = [
            Record(
                id="dead",
                chains=[
                    ChainInfo(
                        chain_id=0, chain_name="A", mol_type=0, cluster_id=0, msa_id=0, num_residues=64, valid=False
                    )
                ],
                interfaces=[],
            ),
            _make_anchor_records()[0],
        ]
        tables = build_record_anchor_tables(records)
        assert 0 not in tables and 1 in tables

    def test_draw_crop_anchor_deterministic_and_none(self):
        """_draw_crop_anchor: deterministic per idx, None when no table / no anchor."""
        import torch

        from deepfold3.data.module.dataset import _draw_crop_anchor
        from deepfold3.data.sample.weighted import build_record_anchor_tables

        tables = build_record_anchor_tables(_make_anchor_records())
        torch.manual_seed(0)
        first = _draw_crop_anchor(tables, 1)
        torch.manual_seed(0)
        assert _draw_crop_anchor(tables, 1) == first  # reproducible under fixed seed
        assert first[0] in ("chain", "interface")
        assert _draw_crop_anchor(None, 0) is None  # uniform dataset
        assert _draw_crop_anchor(tables, 999) is None  # record absent from table

    def test_weighted_sampler_yields_record_indices(self):
        """WeightedClusterSampler now yields record indices in [0, n_records)."""
        from deepfold3.data.sample.weighted import WeightedClusterSampler

        records = _make_anchor_records()
        sampler = WeightedClusterSampler(records=records, epoch_size=40, num_replicas=1, rank=0, seed=3)
        idxs = list(iter(sampler))
        assert len(idxs) == 40
        assert all(0 <= i < len(records) for i in idxs)

    def test_weighted_sampler_ddp_shards_are_disjoint_and_complete(self):
        """Union of per-rank shards equals the single global draw (no overlap, no loss)."""
        import numpy as np

        from deepfold3.data.sample.weighted import WeightedClusterSampler

        records = _make_anchor_records()
        world, ep, seed = 4, 80, 9
        shards = [
            list(iter(WeightedClusterSampler(records=records, epoch_size=ep, num_replicas=world, rank=r, seed=seed)))
            for r in range(world)
        ]
        weights = WeightedClusterSampler(records=records, epoch_size=ep, seed=seed).weights
        glob = np.random.default_rng(seed).choice(len(records), size=ep, replace=True, p=weights)
        assert sum(len(s) for s in shards) == ep
        assert sorted(sum(shards, [])) == sorted(glob.tolist())


# --- per-key loss reporting -------------------------------------------------


def test_aggregate_sample_losses_uses_per_key_denominator():
    """A key emitted by one of two samples must not be halved.

    ``training_loss`` returns early for distillation samples, so in a mixed
    batch only the PDB sample reports plddt/pae/pde/exp_resolved. Dividing
    those by ``batch_size`` under-reported them by n_present/batch_size.
    """
    import torch

    from deepfold3.training.step import aggregate_sample_losses

    pdb_sample = {"total": torch.tensor(2.0), "plddt": torch.tensor(3.0)}
    distill_sample = {"total": torch.tensor(4.0)}  # no confidence keys

    means, counts = aggregate_sample_losses([pdb_sample, distill_sample])

    assert counts == {"total": 2, "plddt": 1}
    assert means["total"].item() == 3.0  # (2 + 4) / 2 — every sample reported
    assert means["plddt"].item() == 3.0  # 3 / 1, NOT 3 / 2


def test_meters_weight_by_per_key_count():
    """Cross-rank averaging must weight by how many samples contributed."""
    import torch

    from deepfold3.training.meters import DistributedAverageMeters

    m = DistributedAverageMeters(torch.device("cpu"))
    # Step 1: both batch elements reported plddt, mean 2.0
    m.update({"plddt": torch.tensor(2.0)}, n={"plddt": 2})
    # Step 2: only one did, value 8.0
    m.update({"plddt": torch.tensor(8.0)}, n={"plddt": 1})

    averaged, _ = m.sync()
    # Weighted by contribution: (2*2 + 8*1) / 3 == 4.0, not the unweighted 5.0
    assert averaged["plddt"] == pytest.approx(4.0)


def test_meters_scalar_n_still_supported():
    """The int form must keep working for callers that pass a single weight."""
    import torch

    from deepfold3.training.meters import DistributedAverageMeters

    m = DistributedAverageMeters(torch.device("cpu"))
    m.update({"loss": torch.tensor(1.0)})
    m.update({"loss": torch.tensor(3.0)})
    averaged, _ = m.sync()
    assert averaged["loss"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Eval dataloader — training-only regularizers must not run at eval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("atomize_eval,expected", [(False, 0.0), (True, 0.5)])
def test_eval_atomization_follows_its_flag(atomize_eval, expected):
    """Eval atomizes only when ``data.atomize_eval`` asks for it.

    Default off: inference never atomizes standard residues, so an eval set
    that did would be scored on a distribution the model is never served — and
    the per-sample Bernoulli draw would add noise to the metric on top of that.
    On: eval inherits the training atomize_* settings verbatim.
    """
    import inspect

    from deepfold3.model.config import ModelConfig
    from deepfold3.training.data import build_dataloaders
    from deepfold3.training.distributed import DistInfo

    cfg = TrainingConfig()
    cfg.data_dir = "/nonexistent"  # no sample is ever read, only the dataset built
    cfg.num_workers = 0
    cfg.eval_num_workers = 0
    model_cfg = ModelConfig()
    model_cfg.data.atomize_prob = 0.5
    model_cfg.data.atomize_eval = atomize_eval

    train_dl, eval_dl = build_dataloaders(
        cfg=cfg,
        model_cfg=model_cfg,
        dist=DistInfo(rank=0, world_size=1, local_rank=0, device=torch.device("cpu")),
        records=_make_records(8),
        eval_records=_make_records(2),
        distillation_records=[],
    )
    assert train_dl.dataset.atomize_prob == 0.5  # train is never touched by the flag
    assert eval_dl.dataset.atomize_prob == expected

    # ``cfg.eval_datasets`` specs are built by the same helper as the training
    # extras and need on-disk discovery to instantiate, so pin the routing flag
    # that selects the eval kwargs for them.
    src = inspect.getsource(inspect.getmodule(build_dataloaders))
    assert "keep_after=True, for_eval=True" in src, "eval specs no longer request the eval kwargs"


# ---------------------------------------------------------------------------
# Scheduler resume-override tests
# ---------------------------------------------------------------------------


class TestSchedulerResumeOverride:
    """``LambdaLR.state_dict()`` is ``self.__dict__`` minus ``optimizer`` /
    ``lr_lambdas``, so the schedule fields ride along into every checkpoint and
    ``load_state_dict`` restores them verbatim — shadowing a ``--config``
    change. ``Trainer._run_inner`` forces the yaml value back in on resume.
    """

    def _make(self, start_decay, decay_every, lr=1e-3, warmup=10, decay_factor=0.95):
        model = _simple_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        cfg = TrainingConfig(
            lr=lr,
            warmup_steps=warmup,
            start_decay_after=start_decay,
            decay_every=decay_every,
            decay_factor=decay_factor,
        )
        return optimizer, AF3LRScheduler(optimizer, cfg), cfg

    def test_state_dict_shadows_the_schedule(self):
        _, saved_sched, _ = self._make(start_decay=50, decay_every=50)
        _, new_sched, _ = self._make(start_decay=10, decay_every=50)
        new_sched.load_state_dict(saved_sched.state_dict())
        # The checkpoint wins — this is the shadowing the override exists for.
        assert new_sched.start_decay_after == 50

    def test_override_restores_yaml_and_keeps_the_step_counter(self):
        from deepfold3.training.trainer import _SCHEDULER_CFG_FIELDS

        saved_opt, saved_sched, _ = self._make(start_decay=50, decay_every=50)
        for _ in range(60):
            saved_opt.step()
            saved_sched.step()
        optimizer, sched, cfg = self._make(start_decay=10, decay_every=50)
        sched.load_state_dict(saved_sched.state_dict())

        for sched_attr, cfg_attr in _SCHEDULER_CFG_FIELDS:
            if hasattr(sched, sched_attr) and hasattr(cfg, cfg_attr):
                setattr(sched, sched_attr, getattr(cfg, cfg_attr))

        assert sched.start_decay_after == 10
        assert sched.last_epoch == 60  # step counter stays aligned with global_step
        # 10 + 50 = 60 -> one decay window has elapsed.
        assert sched.lr_lambdas[0](60) == pytest.approx(0.95)
        assert sched.lr_lambdas[0](59) == pytest.approx(1.0)

    def test_af3_schedule_fields_are_all_covered(self):
        from deepfold3.training.trainer import _SCHEDULER_CFG_FIELDS

        covered = {sched_attr for sched_attr, _ in _SCHEDULER_CFG_FIELDS}
        assert {"warmup_steps", "start_decay_after", "decay_every", "decay_factor"} <= covered

    def test_resume_path_applies_the_override(self):
        import inspect

        from deepfold3.training.trainer import Trainer

        src = inspect.getsource(Trainer._run_inner)
        assert "_SCHEDULER_CFG_FIELDS" in src, "resume path no longer forces the yaml schedule"


class TestShutdownStopFile:
    @staticmethod
    def _dist(rank: int = 0):
        return DistInfo(rank=rank, world_size=1, local_rank=0, device=torch.device("cpu"))

    def test_stop_file_requests_shutdown(self, tmp_path, monkeypatch):
        stop = tmp_path / ".stop"
        monkeypatch.setenv("DF3_STOP_FILE", str(stop))
        shutdown = ShutdownCoordinator(self._dist())
        assert shutdown.tick() is False
        stop.touch()
        assert shutdown.tick() is True

    def test_only_rank_zero_polls(self, tmp_path, monkeypatch):
        stop = tmp_path / ".stop"
        stop.touch()
        monkeypatch.setenv("DF3_STOP_FILE", str(stop))
        assert ShutdownCoordinator(self._dist(rank=1)).tick() is False

    def test_unset_is_inert(self, monkeypatch):
        monkeypatch.delenv("DF3_STOP_FILE", raising=False)
        assert ShutdownCoordinator(self._dist()).tick() is False


# ---------------------------------------------------------------------------
# Warmup -> stable -> decay shapes (cosine / linear / 1-sqrt)
# ---------------------------------------------------------------------------


class TestDecayShapes:
    def _lam(self, shape, *, warmup=10, decay_start=0, max_steps=110, min_ratio=0.0):
        optimizer = torch.optim.Adam(_simple_model().parameters(), lr=1e-3)
        cfg = TrainingConfig(
            lr=1e-3,
            finetune_scheduler=shape,
            warmup_steps=warmup,
            finetune_decay_start=decay_start,
            finetune_max_steps=max_steps,
            finetune_min_lr_ratio=min_ratio,
        )
        sched = build_scheduler(optimizer, cfg)
        assert isinstance(sched, CosineAnnealingScheduler)
        return sched.lr_lambdas[0]

    @pytest.mark.parametrize("shape,mid", [("cosine", 0.5), ("linear", 0.5), ("1-sqrt", 1.0 - 0.5**0.5)])
    def test_endpoints_and_midpoint(self, shape, mid):
        lam = self._lam(shape)
        assert lam(5) == pytest.approx(0.5)  # warmup
        assert lam(10) == pytest.approx(1.0)  # decay starts at the end of warmup
        assert lam(60) == pytest.approx(mid)  # half way through [10, 110]
        assert lam(110) == pytest.approx(0.0)
        assert lam(500) == pytest.approx(0.0)  # held at the floor

    def test_min_ratio_floor(self):
        lam = self._lam("linear", min_ratio=0.1)
        assert lam(60) == pytest.approx(0.1 + 0.9 * 0.5)
        assert lam(110) == pytest.approx(0.1)

    def test_stable_phase(self):
        lam = self._lam("1-sqrt", warmup=10, decay_start=60, max_steps=110)
        assert lam(10) == pytest.approx(1.0)
        assert lam(59) == pytest.approx(1.0)
        assert lam(60) == pytest.approx(1.0)
        assert lam(85) == pytest.approx(1.0 - 0.5**0.5)
        assert lam(110) == pytest.approx(0.0)

    def test_cosine_default_unchanged(self):
        # finetune_decay_start=0 keeps the old warmup -> cosine schedule.
        lam = self._lam("cosine", warmup=10, max_steps=110, min_ratio=0.1)
        assert lam(35) == pytest.approx(0.1 + 0.9 * (1 + math.cos(math.pi * 0.25)) / 2)

    def test_end_before_start_rejected(self):
        with pytest.raises(ValueError, match="finetune_max_steps"):
            self._lam("linear", warmup=10, decay_start=200, max_steps=110)

    def test_resume_override_switches_shape(self):
        from deepfold3.training.trainer import _scheduler_cfg_fields

        optimizer = torch.optim.Adam(_simple_model().parameters(), lr=1e-3)
        saved = build_scheduler(
            optimizer,
            TrainingConfig(
                lr=1e-3, warmup_steps=10, start_decay_after=10, finetune_scheduler="cosine", finetune_max_steps=110
            ),
        )
        cfg = TrainingConfig(
            lr=1e-3,
            warmup_steps=10,
            start_decay_after=10,
            finetune_scheduler="linear",
            finetune_max_steps=110,
            finetune_decay_start=50,
        )
        sched = build_scheduler(optimizer, cfg)
        sched.load_state_dict(saved.state_dict())
        assert sched.shape == "cosine"  # the checkpoint shadows the yaml
        for a, c in _scheduler_cfg_fields(sched):
            if hasattr(sched, a):
                setattr(sched, a, getattr(cfg, c))
        assert (sched.shape, sched.decay_start) == ("linear", 50)

    def test_wsds_min_ratio_maps_to_its_own_field(self):
        from deepfold3.training.trainer import _scheduler_cfg_fields

        optimizer = torch.optim.Adam(_simple_model().parameters(), lr=1e-3)
        cfg = TrainingConfig(
            lr=1e-3, finetune_scheduler="wsds", wsds_decay_windows=[[100, 110]], wsds_min_lr_ratio=0.01
        )
        fields = dict(_scheduler_cfg_fields(build_scheduler(optimizer, cfg)))
        assert fields["min_ratio"] == "wsds_min_lr_ratio"
        assert "shape" not in fields


# ---------------------------------------------------------------------------
# freeze_modules
# ---------------------------------------------------------------------------


class TestFreezeModules:
    def _model(self):
        torch.manual_seed(0)
        m = nn.Module()
        m.trunk = nn.Linear(4, 4)
        m.head = nn.Linear(4, 2)
        return m

    def test_freezes_named_children_only(self):
        from deepfold3.training.optim import build_optimizer, freeze_modules

        m = self._model()
        freeze_modules(m, ["trunk"])
        assert not any(p.requires_grad for p in m.trunk.parameters())
        assert all(p.requires_grad for p in m.head.parameters())
        # No EMA shadow and no optimizer slot for frozen parameters.
        assert set(EMA(m).shadow) == {"head.weight", "head.bias"}
        opt = build_optimizer(m, TrainingConfig(lr=1e-3))
        n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
        assert n_opt == sum(p.numel() for p in m.head.parameters())

    def test_frozen_trunk_gets_no_grad(self):
        from deepfold3.training.optim import freeze_modules

        m = self._model()
        freeze_modules(m, ["trunk"])
        m.head(m.trunk(torch.randn(3, 4))).sum().backward()
        assert all(p.grad is None for p in m.trunk.parameters())
        assert all(p.grad is not None for p in m.head.parameters())

    def test_unknown_name_rejected(self):
        from deepfold3.training.optim import freeze_modules

        with pytest.raises(ValueError, match="no top-level module"):
            freeze_modules(self._model(), ["pairformer"])

    def test_stage2_frozen_trunk_names_exist(self):
        from deepfold3.config_loader import resolve_config
        from deepfold3.model.model import DeepFold3

        names = resolve_config("configs/stage2-frozen-trunk.yaml")["training"]["freeze_modules"]
        assert names == ["evo_cond_encoder", "evoformer"]
        # Building DeepFold3 on CPU is slow; the names must be assigned in __init__.
        import inspect

        src = inspect.getsource(DeepFold3.__init__)
        for n in names:
            assert f"self.{n} = " in src


def test_cosine_spellings_still_read():
    """Configs and pickled train_cfg written before the finetune_* names."""
    import pickle

    # scripts/train.py: default config first, yaml keys setattr()'d afterwards.
    cfg = TrainingConfig()
    cfg.cosine_max_steps, cfg.cosine_min_lr_ratio = 5000, 0.0
    assert (cfg.effective_finetune_max_steps, cfg.effective_finetune_min_lr_ratio) == (5000, 0.0)
    cfg.finetune_max_steps = 7000
    assert cfg.effective_finetune_max_steps == 7000  # the new name wins

    old = TrainingConfig(cosine_max_steps=3000)
    state = dict(old.__dict__)
    for k in ("finetune_max_steps", "finetune_min_lr_ratio", "finetune_decay_start"):
        state.pop(k)
    restored = TrainingConfig.__new__(TrainingConfig)
    restored.__setstate__(state)
    assert (restored.effective_finetune_max_steps, restored.finetune_decay_start) == (3000, 0)
    assert pickle.loads(pickle.dumps(restored)).effective_finetune_max_steps == 3000
