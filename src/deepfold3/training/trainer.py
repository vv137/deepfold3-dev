"""DDP trainer for DeepFold3.

Native PyTorch ``DistributedDataParallel`` training loop following
Protenix's procedure with Boltz-style code organization.  No Lightning
dependency.

Example::

    trainer = Trainer(model_cfg, train_cfg)
    trainer.run()
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import math
import os
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as torch_dist
import torch.nn as nn
import yaml
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from deepfold3.data.io_cache import set_drop_page_cache
from deepfold3.data.module.collate import collate_fn
from deepfold3.model.config import ModelConfig, TrainingConfig
from deepfold3.model.model import DeepFold3
from deepfold3.training.accumulator import AccumulationLoop
from deepfold3.training.callbacks import CallbackHub
from deepfold3.training.callbacks.ema_callback import EMACallback
from deepfold3.training.checkpointing import CheckpointManager, load_pretrained_weights
from deepfold3.training.data import build_dataloaders
from deepfold3.training.distributed import DistInfo, destroy_process_group_safely, init_process_group
from deepfold3.training.ema import EMA
from deepfold3.training.eval import run_eval
from deepfold3.training.logging_payload import build_step_payload
from deepfold3.training.meters import DistributedAverageMeters
from deepfold3.training.optim import (
    WrappedOptimizer,
    build_optimizer,
    freeze_for_confidence,
    freeze_modules,
    is_chunked_diffusion_active,
    register_reduce_hook,
    wrap_ddp,
)
from deepfold3.training.rank_zero import rank_zero_info, rank_zero_warn
from deepfold3.training.scheduler import AF3LRScheduler, WSDSScheduler, build_scheduler
from deepfold3.training.signals import CheckpointRequestHandler, ShutdownCoordinator
from deepfold3.training.state import TrainState, seed_everything
from deepfold3.training.step import scaled_backward, training_step
from deepfold3.training.step_timer import StepTimeMeter
from deepfold3.training.tracking import StdoutTracker, Tracker, TrackerHub, WandBTracker

# Bucket-padded collation keeps Triton kernel input shapes (and the
# persistent autotune cache keys) deterministic across microsteps.
_collate_bucket_pad = partial(collate_fn, bucket_pad=True)

logger = logging.getLogger(__name__)

# yaml-controlled scheduler schedule fields: (scheduler attribute, TrainingConfig
# field). ``LambdaLR.state_dict`` is ``self.__dict__`` minus ``optimizer`` /
# ``lr_lambdas``, so each of these is baked into every checkpoint and restored
# verbatim by ``load_state_dict`` — shadowing a --config change unless it is
# forced back in on resume. Shared by the resume override and the resume audit
# so the two lists cannot drift apart.
_SCHEDULER_CFG_FIELDS: tuple[tuple[str, str], ...] = (
    ("max_lr", "lr"),
    ("warmup_steps", "warmup_steps"),
    ("start_decay_after", "start_decay_after"),
    ("decay_every", "decay_every"),
    ("decay_factor", "decay_factor"),
    ("max_steps", "effective_finetune_max_steps"),
    ("min_ratio", "effective_finetune_min_lr_ratio"),
    ("decay_start", "finetune_decay_start"),
    ("shape", "finetune_scheduler"),
)


def _scheduler_cfg_fields(scheduler) -> tuple[tuple[str, str], ...]:
    """``_SCHEDULER_CFG_FIELDS`` for this scheduler's class.

    ``WSDSScheduler.min_ratio`` is ``wsds_min_lr_ratio``, not the cosine field
    of the same attribute name.
    """
    if isinstance(scheduler, WSDSScheduler):
        return tuple(
            (a, "wsds_min_lr_ratio" if a == "min_ratio" else c) for a, c in _SCHEDULER_CFG_FIELDS if a != "shape"
        )
    return _SCHEDULER_CFG_FIELDS


def _yaml_safe(value):
    """Recursively convert tuples to lists so the result is `yaml.safe_dump` clean."""
    if isinstance(value, tuple):
        return [_yaml_safe(v) for v in value]
    if isinstance(value, list):
        return [_yaml_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _yaml_safe(v) for k, v in value.items()}
    return value


def serialize_run_config(train_cfg: TrainingConfig, model_cfg: ModelConfig) -> dict:
    """Build a round-trippable resolved-config dict.

    Mirrors the input YAML layout accepted by :func:`deepfold3.config_loader.load_config`:
    top-level ``training``, ``model``, optional ``datasets`` and ``eval_datasets``.
    The result can be re-fed via ``--config`` to reproduce the run.
    """
    train_dict = _yaml_safe(dataclasses.asdict(train_cfg))
    extras = train_dict.pop("extra_datasets", []) or []
    evals = train_dict.pop("eval_datasets", []) or []
    out: dict = {
        "training": train_dict,
        "model": _yaml_safe(dataclasses.asdict(model_cfg)),
    }
    if extras:
        out["datasets"] = extras
    if evals:
        out["eval_datasets"] = evals
    return out


class Trainer:
    """DeepFold3 distributed trainer.

    Orchestrates DDP setup, training loop, checkpointing, EMA, and evaluation.

    Args:
        model_cfg: Model architecture config.
        train_cfg: Training hyperparameters.
        records: Training data records.
        eval_records: Optional evaluation data records.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        records: list | None = None,
        eval_records: list | None = None,
        distillation_records: list | None = None,
        provenance: dict | None = None,
    ):
        self.model_cfg = model_cfg
        self.cfg = train_cfg
        self.records = records or []
        self.eval_records = eval_records or []
        self.distillation_records = distillation_records or []
        self.provenance = provenance

        self.model: DeepFold3 | None = None
        self.ddp_model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: AF3LRScheduler | None = None
        self.ema: EMA | None = None
        self.scaler: GradScaler | None = None
        self.trackers: TrackerHub | None = None
        self.accumulator: AccumulationLoop | None = None
        self.ckpt_manager: CheckpointManager | None = None
        self.wrapped_optimizer: WrappedOptimizer | None = None
        self.callbacks: CallbackHub = CallbackHub()
        self.dist: DistInfo | None = None
        self.state = TrainState()
        self._train_sampler: object | None = None

    # ------------------------------------------------------------------
    # Distributed setup
    # ------------------------------------------------------------------

    def _init_distributed(self) -> None:
        """Bring up the process group — delegates to :func:`init_process_group`."""
        self.dist = init_process_group(timeout_s=self.cfg.collective_timeout_s)

    def _seed_everything(self) -> None:
        """Seed all RNGs with a per-rank offset."""
        seed_everything(self.cfg.seed, self.dist.rank, deterministic=self.cfg.deterministic)

    def _init_memory_policy(self) -> None:
        """Host-memory policy for a long DDP run under a container cgroup.

        Two pressures, both proportional to samples consumed rather than to
        wall-clock, which is why they surface as step-periodic stalls:

        * **Page cache.** Records are drawn at random from 10^6–10^7-structure
          datasets, so a cached NPZ is never read again — the cache only grows
          until allocations start hitting direct reclaim. Training opts every
          dataset-path read into ``POSIX_FADV_DONTNEED``; offline CLIs sharing
          those loaders keep normal caching.
        * **Copy-on-write.** Dataloader workers fork from this process and
          inherit its heap. CPython writes to an object's GC header on every
          collection pass, so a full collection in a worker un-shares pages
          across the whole inherited heap. ``gc.freeze`` at the fork point
          moves everything allocated so far into a permanent generation the
          collector never visits. Registering it via ``register_at_fork``
          rather than calling it inline catches objects created after setup
          (a resumed checkpoint blob, tracker state) that an inline call
          would miss. Refcount updates still dirty individual objects on
          first touch, so this bounds the growth rather than eliminating it.
        """
        set_drop_page_cache(True)
        os.register_at_fork(before=gc.freeze)

    # ------------------------------------------------------------------
    # Model and optimizer setup
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        """Create model, optimizer, scheduler, EMA, and wrap in DDP."""
        self.model = DeepFold3(self.model_cfg).to(self.dist.device)

        # Load pretrained weights if specified
        if self.cfg.params_path:
            self._load_pretrained(self.cfg.params_path)

        # Freeze parameters for confidence-only training (stage 3)
        if self.cfg.train_confidence_only and self.cfg.freeze_modules:
            raise ValueError("train_confidence_only and freeze_modules are exclusive; set one")
        if self.cfg.train_confidence_only:
            freeze_for_confidence(self.model)
        if self.cfg.freeze_modules:
            freeze_modules(self.model, self.cfg.freeze_modules)

        self.ddp_model, use_static_graph = wrap_ddp(self.model, self.dist, self.cfg)
        self._install_precision_policy()
        self.accumulator = AccumulationLoop(
            ddp_model=self.ddp_model,
            dist=self.dist,
            num_steps=self.cfg.accumulate_grad_batches,
            per_rank_clip=self.cfg.per_rank_grad_clip,
        )
        self.optimizer = build_optimizer(self.model, self.cfg)
        self.scheduler = build_scheduler(self.optimizer, self.cfg)

        # EMA — owned by the trainer (for eval swap + checkpoint save/load),
        # driven by ``EMACallback`` on every successful optimizer step. The
        # callback wraps ``ema.update`` in ``optimizer_eval_mode`` for the
        # SF case so the shadow tracks ``x`` not ``y``; AdamW is unaffected.
        if self.cfg.ema_decay > 0:
            self.ema = EMA(self.model, decay=self.cfg.ema_decay)
            self.callbacks.append(EMACallback(self.ema, optimizer=self.optimizer))

        # Mixed precision scaler (only for fp16; bf16 doesn't need scaling)
        self.scaler = GradScaler("cuda", enabled=(self.cfg.dtype == "fp16"))

        self.wrapped_optimizer = WrappedOptimizer(
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            model=self.model,
            dist_info=self.dist,
            cfg=self.cfg,
        )

        self.ckpt_manager = CheckpointManager(
            max_checkpoints=self.cfg.max_checkpoints,
            strict=self.cfg.load_strict,
            device=self.dist.device,
            staging_dir=self.cfg.checkpoint_staging_dir or None,
        )

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        rank_zero_info(f"Model parameters: {n_params:.1f}M total, {n_trainable:.1f}M trainable")
        rank_zero_info(
            f"DDP: world_size={self.dist.world_size}, static_graph={use_static_graph}, "
            f"accumulate_grad_batches={self.cfg.accumulate_grad_batches}"
        )

    def _install_precision_policy(self) -> None:
        """Validate cross-config precision invariants and install the comm hook.

        Runs after ``wrap_ddp`` (needs the DDP-wrapped model). Fails
        loud on combinations that would otherwise silently drift:
        bf16 grad-accum across microsteps drops small contributions;
        bf16 reduce on an fp32-compute run downcasts for no benefit.
        """
        precision = self.model_cfg.precision
        if precision.grad_acc_dtype == "bf16" and self.cfg.accumulate_grad_batches > 1:
            raise ValueError(
                f"grad_acc_dtype='bf16' is incompatible with accumulate_grad_batches="
                f"{self.cfg.accumulate_grad_batches} (>1): bf16 rounding silently drops "
                "small microstep contributions from the accumulator. Use 'fp32' for grad_acc."
            )
        if self.cfg.dtype == "fp32" and precision.reduce_dtype == "bf16":
            raise ValueError(
                "reduce_dtype='bf16' is incompatible with compute dtype='fp32' — the hook "
                "downcasts fp32 grads to bf16 before allreduce, losing precision without "
                "any bandwidth benefit in an fp32 run."
            )

        comm_hook_name = (
            register_reduce_hook(self.ddp_model, precision.reduce_dtype)
            if self.dist.is_distributed
            else "default (fp32 allreduce)"
        )
        rank_zero_info(
            "Precision policy: compute=%s, grad_acc=%s, reduce=%s, loss=%s; comm_hook=%s",
            self.cfg.dtype,
            precision.grad_acc_dtype,
            precision.reduce_dtype,
            precision.loss_dtype,
            comm_hook_name,
        )

    def _load_pretrained(self, path: str) -> None:
        """Thin wrapper around :func:`load_pretrained_weights`."""
        load_pretrained_weights(self.model, path, strict=self.cfg.load_strict, device=self.dist.device)

    def _init_trackers(self) -> None:
        """Build a :class:`TrackerHub` from the training config.

        Non-main ranks get an empty hub so ``log()`` is a safe no-op on
        every rank. On rank 0 the hub always contains a ``StdoutTracker``
        plus a ``WandBTracker`` when ``wandb_enabled`` + a non-empty
        ``wandb_project`` are set.
        """
        trackers: list[Tracker] = [StdoutTracker()]
        if self.dist.is_main and self.cfg.wandb_enabled:
            if not self.cfg.wandb_project:
                rank_zero_warn("wandb_enabled=True but wandb_project is empty; skipping wandb init")
            else:
                run_name = self.cfg.wandb_run_name or Path(self.cfg.output_dir).name
                trackers.append(
                    WandBTracker(
                        project=self.cfg.wandb_project,
                        entity=self.cfg.wandb_entity,
                        run_name=run_name,
                    )
                )
        self.trackers = TrackerHub(trackers)
        self.trackers.start(dataclasses.asdict(self.cfg))

    # ------------------------------------------------------------------
    # Data setup
    # ------------------------------------------------------------------

    def _init_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        """Thin wrapper around :func:`build_dataloaders`."""
        return build_dataloaders(
            cfg=self.cfg,
            model_cfg=self.model_cfg,
            dist=self.dist,
            records=self.records,
            eval_records=self.eval_records,
            distillation_records=self.distillation_records,
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _training_step(
        self, features: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, int]]:
        """Thin wrapper around :func:`training_step` — keeps the call sites tidy.

        Returns ``(total_loss, loss_dict, loss_counts)``. ``loss_counts`` is the
        number of batch elements that actually emitted each key, so the meters
        can weight the cross-rank average by contribution instead of treating a
        rank that reported one sample the same as one that reported two.
        """
        # MUST pass the DDP-wrapped model so that DDP's ``_pre_forward`` runs
        # and primes the reducer for the upcoming backward — otherwise the
        # bucket AllReduce is never scheduled and ranks train independently.
        total_loss, loss_dict, loss_counts, num_recycles, fwd_wall, loss_wall = training_step(
            self.ddp_model,
            features,
            cfg=self.cfg,
            model_cfg=self.model_cfg,
            global_step=self.state.global_step,
            accumulator=self.accumulator,
            scaler=self.scaler,
            device=self.dist.device,
        )
        self.state.last_num_recycles = num_recycles
        # Tier-4c phase breakdown (0.0 unless slow_step_log_threshold_s > 0).
        self._last_fwd_wall = fwd_wall
        self._last_loss_wall = loss_wall
        return total_loss, loss_dict, loss_counts

    def _chunked_diffusion_active(self) -> bool:
        return is_chunked_diffusion_active(self.cfg)

    def _scaled_backward(self, loss: torch.Tensor) -> None:
        scaled_backward(self.scaler, loss, self.cfg.accumulate_grad_batches)

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def _optimizer_step(self, loss: torch.Tensor, record_ids: list[str | None] | None = None) -> bool:
        """Accumulate the microstep's NaN flag, backward, and — on sync — step.

        Delegates the NaN collective + unscale + clip + step contract
        to :class:`WrappedOptimizer`; this shell handles the
        no_sync/sync-step DDP gating and EMA update. ``record_ids`` are
        the dataset ids of the samples in this microstep's batch, used
        for NaN-skip diagnostics.
        """
        self.wrapped_optimizer.add_microstep_loss(loss, record_ids=record_ids)

        is_sync_step = self.accumulator.is_sync_step(self.state.global_step)
        with self.accumulator.step_context(is_sync_step=is_sync_step):
            self._scaled_backward(loss)
        self.state.global_step += 1

        if not is_sync_step:
            return False

        result = self.wrapped_optimizer.step_or_skip(self.state)
        if result.advanced:
            self.callbacks.on_step_end(self, result)
        else:
            self.callbacks.on_skip_step(self, result)
        return result.advanced

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _run_eval_and_track_best(self, eval_dl: DataLoader, output_dir: Path) -> None:
        """Reseed eval sampler, run one eval pass, log metrics, save best checkpoint."""
        if hasattr(eval_dl.sampler, "set_epoch"):
            eval_dl.sampler.set_epoch(self.state.effective_step)
        eval_metrics = run_eval(
            self.model,
            self.ema,
            eval_dl,
            self.cfg,
            self.model_cfg,
            optimizer=self.optimizer,
            device=self.dist.device,
            log_microsteps=self.cfg.log_microsteps,
        )
        eval_total = eval_metrics.get("eval/total", float("inf"))
        save_best = eval_total < self.state.best_eval_loss
        if self.dist.is_main:
            logger.info(f"Eval at step {self.state.effective_step}: {eval_metrics}")
            # Inject ``train/step`` so wandb's ``step_metric="train/step"``
            # places eval points on the correct x-axis. Without it, an eval
            # fired before any train-step log call (e.g. eval_before on
            # resume) is charted at the wandb internal step (=0 for a fresh
            # run) instead of ``effective_step``.
            self.trackers.log(
                {**eval_metrics, "train/step": self.state.effective_step},
                step=self.state.effective_step,
            )
        # Broadcast the save decision so every rank participates in the
        # RNG-gather inside ``_save_checkpoint``. ``eval_total`` may be
        # rank-0-only when meters aren't all-reduced, so don't trust it
        # on non-main ranks for the comparison.
        if self.dist.is_distributed:
            flag = torch.tensor(int(save_best), device=self.dist.device)
            torch_dist.broadcast(flag, src=0)
            save_best = bool(flag.item())
        if save_best:
            self.state.best_eval_loss = eval_total
            self._save_checkpoint(output_dir / "checkpoint_best.pt")
            if self.dist.is_main:
                logger.info(f"New best eval loss: {eval_total:.4f}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: Path) -> None:
        """Save checkpoint via :class:`CheckpointManager`.

        All ranks call this so the cross-rank RNG-state gather can run;
        only rank 0 actually writes the file. Callers that previously
        guarded with ``self.dist.is_main`` no longer need to (the gate
        moved into ``CheckpointManager.save``).
        """
        self.ckpt_manager.save(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema=self.ema,
            scaler=self.scaler,
            state=self.state,
            model_cfg=self.model_cfg,
            train_cfg=self.cfg,
            dist_info=self.dist,
            train_sampler=self._train_sampler,
        )

    @staticmethod
    def _find_latest_checkpoint(output_dir: Path) -> Path | None:
        return CheckpointManager.find_latest(output_dir)

    def _load_checkpoint(self, path: Path) -> None:
        """Load checkpoint via :class:`CheckpointManager`.

        Computes ``steps_per_epoch`` and ``samples_per_step_per_rank``
        so old checkpoints (no ``epoch`` / ``sampler`` keys) can derive
        a sampler-epoch + within-epoch position from ``effective_step``
        — the sampler RNG keys on epoch, so we want to land in the
        right basin and skip records already trained on, even when
        resuming from a pre-sampler-persistence ckpt.
        """
        cfg = self.cfg
        effective_batch_size = cfg.batch_size_per_rank * self.dist.world_size * cfg.accumulate_grad_batches
        steps_per_epoch = max(1, cfg.epoch_size // max(1, effective_batch_size)) if cfg.epoch_size > 0 else 0
        samples_per_step_per_rank = cfg.batch_size_per_rank * cfg.accumulate_grad_batches
        self.ckpt_manager.load(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema=self.ema,
            scaler=self.scaler,
            state=self.state,
            train_cfg=self.cfg,
            dist_info=self.dist,
            steps_per_epoch=steps_per_epoch,
            train_sampler=self._train_sampler,
            samples_per_step_per_rank=samples_per_step_per_rank,
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run(self, resume_from: str | Path | None = None) -> None:
        """Run the full training loop.

        Args:
            resume_from: Optional checkpoint path to resume from.
        """
        self._run_inner(resume_from)

    def _run_inner(self, resume_from: str | Path | None) -> None:
        self._init_distributed()
        self._seed_everything()
        self._init_memory_policy()

        self._init_model()
        self._init_trackers()

        train_dl, eval_dl = self._init_dataloaders()
        # Stash the train sampler so the checkpoint manager can persist
        # its within-epoch position (only one sampler per run; capturing
        # it once here avoids re-deriving from ``train_dl.sampler`` at
        # every save site).
        self._train_sampler = train_dl.sampler
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save resolved config (all CLI/YAML overrides applied) for reproducibility.
        # Layout mirrors configs/*.yaml so the dump can be re-fed via --config.
        if self.dist.is_main:
            config_out = output_dir / "config.yaml"
            with open(config_out, "w") as f:
                yaml.safe_dump(
                    serialize_run_config(self.cfg, self.model_cfg),
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                )
            logger.info(f"Resolved config saved to {config_out}")
            if self.provenance is not None:
                prov_out = output_dir / "provenance.yaml"
                with open(prov_out, "w") as f:
                    yaml.safe_dump(_yaml_safe(self.provenance), f, default_flow_style=False, sort_keys=False)
                logger.info(f"Run provenance saved to {prov_out}")

        # Auto-resume: find latest checkpoint in output_dir if none specified
        resumed = False
        if resume_from is not None:
            self._load_checkpoint(Path(resume_from))
            resumed = True
        elif self.cfg.auto_resume:
            latest = self._find_latest_checkpoint(output_dir)
            if latest is not None:
                self._load_checkpoint(latest)
                resumed = True

        # Resume LR override: optimizer.load_state_dict and
        # scheduler.load_state_dict both restore the lr/initial_lr/base_lrs
        # baked into the checkpoint, which silently shadows any --config
        # change. When the configured lr differs from the saved peak, force
        # the new value into both so the schedule honors the override.
        # SF-AdamW's accumulator state (z buffer, lr_max, weight_sum) is
        # preserved — only the per-step `lr`/`initial_lr` is bumped.
        if resumed and self.optimizer.param_groups:
            saved_lr = float(self.optimizer.param_groups[0].get("initial_lr", self.optimizer.param_groups[0]["lr"]))
            target_lr = float(self.cfg.lr)
            if abs(saved_lr - target_lr) > 1e-12:
                rank_zero_info(
                    f"[resume] LR override: {saved_lr:.3e} → {target_lr:.3e} "
                    "(forcing optimizer.param_groups['lr','initial_lr'] + scheduler.base_lrs; "
                    "SF accumulator state preserved)"
                )
                for g in self.optimizer.param_groups:
                    g["lr"] = target_lr
                    g["initial_lr"] = target_lr
                base_lrs = getattr(self.scheduler, "base_lrs", None)
                if base_lrs is not None:
                    self.scheduler.base_lrs = [target_lr] * len(base_lrs)

        # Resume decoupling_parameter override: same "yaml wins" pattern as
        # the LR override above. Justified because decoupling_parameter (C) is
        # purely a hyperparameter; nothing in optimizer.state[p] is tied to its
        # value. The weight_sum/lr_max accumulator is C-independent — C only
        # rescales the final averaging coefficient (Algorithm 2 line 11) — so
        # switching a saved run's C mid-trajectory (incl. vanilla → refined) is
        # safe.
        if resumed and self.optimizer.param_groups:
            # Migrate the legacy group-key from checkpoints saved before
            # average_concentration → decoupling_parameter was renamed, so
            # step() (reads group["decoupling_parameter"]) can't KeyError.
            for g in self.optimizer.param_groups:
                if "decoupling_parameter" not in g and "average_concentration" in g:
                    g["decoupling_parameter"] = g.pop("average_concentration")
            target_c = self.cfg.decoupling_parameter
            saved_c = self.optimizer.param_groups[0].get("decoupling_parameter", None)
            if saved_c != target_c:
                rank_zero_info(
                    f"[resume] decoupling_parameter override: {saved_c} → {target_c} "
                    "(yaml wins; no internal accumulator entangled with this field)"
                )
                for g in self.optimizer.param_groups:
                    g["decoupling_parameter"] = target_c

        # Resume scheduler-schedule override: same "yaml wins" pattern as the
        # LR override above. Every field in ``_SCHEDULER_CFG_FIELDS`` is stored
        # on the scheduler instance, hence saved into ``scheduler.state_dict()``
        # and restored by ``load_state_dict`` — so a --config change to
        # ``start_decay_after`` / ``decay_every`` / ``decay_factor`` is silently
        # ignored on resume without this. These are pure hyperparameters; no
        # optimizer or scheduler accumulator is tied to their value, so the yaml
        # wins and every override is reported to the log.
        #
        # ``last_epoch`` / ``_step_count`` / ``base_lrs`` are deliberately left
        # alone: the step counter must stay aligned with global_step, and
        # base_lrs is owned by the LR override above. ``wsds_decay_windows`` is
        # also not forced — rebuilding the sorted/validated window list belongs
        # in the constructor — so the audit below still reports it as shadowed.
        if resumed and self.scheduler is not None:
            sched_overrides: list[str] = []
            for sched_attr, cfg_attr in _scheduler_cfg_fields(self.scheduler):
                if not hasattr(self.scheduler, sched_attr) or not hasattr(self.cfg, cfg_attr):
                    continue
                saved_val = getattr(self.scheduler, sched_attr)
                target_val = getattr(self.cfg, cfg_attr)
                if saved_val != target_val:
                    setattr(self.scheduler, sched_attr, target_val)
                    sched_overrides.append(f"{sched_attr}: {saved_val} → {target_val}")
            if sched_overrides:
                rank_zero_info(
                    "[resume] scheduler schedule override (yaml wins over checkpoint): " + ", ".join(sched_overrides)
                )

        # Resume EMA shadow re-seed: if the source checkpoint has no saved
        # EMA shadow (e.g. an EMA-on-x arm starting from a vanilla-SF
        # checkpoint), the freshly constructed shadow is at random init
        # while the model holds the resumed weights. Re-clone the shadow
        # from the (now-loaded) model so the EMA starts equal to the
        # resumed weights and decays naturally from there.
        if resumed and self.ema is not None and resume_from is not None:
            try:
                raw = torch.load(str(Path(resume_from)), map_location="cpu", weights_only=False)
                has_ema = isinstance(raw, dict) and "ema" in raw
            except Exception:
                has_ema = True  # don't second-guess the load if introspection fails
            if not has_ema:
                self.ema.reseed_from(self.model)
                rank_zero_info("[resume] EMA shadow re-seeded from loaded weights (source checkpoint has no 'ema' key)")

        # Resume audit (warn-only): catalogue every yaml-controlled field
        # that's silently shadowed by checkpoint state on resume. lr is
        # already force-overridden above; everything else here is reported
        # to the user so they can decide whether the saved value should win
        # (intentional) or the yaml should win (typo / forgot to update).
        # No state is mutated by this block.
        if resumed:
            mismatches: list[str] = []

            # weight_decay (per param-group). build_optimizer constructs
            # param_groups as [decay (cfg.weight_decay), no_decay (0.0)] in
            # that fixed order; param_groups have no `name` key.
            decay_targets = (float(self.cfg.weight_decay), 0.0)
            for gi, g in enumerate(self.optimizer.param_groups):
                saved_wd = float(g.get("weight_decay", 0.0))
                target_wd = decay_targets[gi] if gi < len(decay_targets) else 0.0
                if abs(saved_wd - target_wd) > 1e-12:
                    label = ("decay", "no_decay")[gi] if gi < 2 else f"group_{gi}"
                    mismatches.append(f"  weight_decay[{label}]: saved={saved_wd:.4g} != cfg={target_wd:.4g}")

            # Optimizer betas / eps (stored in param_groups; restored verbatim
            # by optimizer.load_state_dict). Saved as list; cfg keeps tuple.
            if self.optimizer.param_groups:
                saved_betas = tuple(self.optimizer.param_groups[0].get("betas", ()))
                target_betas = tuple(self.cfg.betas)
                if saved_betas and saved_betas != target_betas:
                    mismatches.append(f"  betas: saved={saved_betas} != cfg={target_betas}")
                saved_eps = float(self.optimizer.param_groups[0].get("eps", 0.0))
                target_eps = float(self.cfg.eps)
                if abs(saved_eps - target_eps) > 1e-20:
                    mismatches.append(f"  eps: saved={saved_eps:.3e} != cfg={target_eps:.3e}")

            # EMA decay (only when EMA is enabled and was loaded).
            if self.ema is not None:
                saved_ema = float(getattr(self.ema, "decay", self.cfg.ema_decay))
                target_ema = float(self.cfg.ema_decay)
                if abs(saved_ema - target_ema) > 1e-12:
                    mismatches.append(f"  ema_decay: saved={saved_ema} != cfg={target_ema}")

            # Scheduler instance attributes. LambdaLR.state_dict carries these
            # into the checkpoint and load_state_dict restores them verbatim,
            # so they are force-overridden from the yaml above; anything still
            # mismatched here (currently only the WSDS window list) genuinely
            # kept the saved value.
            sched = self.scheduler
            sched_pairs: list[tuple[str, str]] = []
            for sched_attr, cfg_attr in _scheduler_cfg_fields(sched):
                if hasattr(sched, sched_attr) and hasattr(self.cfg, cfg_attr):
                    sched_pairs.append((sched_attr, cfg_attr))
            for sched_attr, cfg_attr in sched_pairs:
                saved_val = getattr(sched, sched_attr)
                target_val = getattr(self.cfg, cfg_attr)
                if saved_val != target_val:
                    mismatches.append(f"  scheduler.{sched_attr}: saved={saved_val} != cfg={target_val}")
            # WSDS-only: windows are list-of-tuples internally vs list-of-list in cfg.
            if hasattr(sched, "windows"):
                saved_w = [tuple(p) for p in getattr(sched, "windows", [])]
                target_w = [tuple(p) for p in getattr(self.cfg, "wsds_decay_windows", [])]
                if saved_w != target_w:
                    mismatches.append(f"  wsds_decay_windows: saved={saved_w} != cfg={target_w}")

            if mismatches:
                rank_zero_warn(
                    "[resume] yaml fields silently shadowed by checkpoint state — "
                    "the values below come from the saved run, NOT your --config. "
                    "Fix the yaml or accept the saved value:\n" + "\n".join(mismatches)
                )
            else:
                rank_zero_info("[resume] yaml-vs-checkpoint state audit: all shadowed fields match")

        cfg = self.cfg
        tokens_per_rank = cfg.max_tokens * cfg.batch_size_per_rank
        effective_batch_size = cfg.batch_size_per_rank * self.dist.world_size * cfg.accumulate_grad_batches
        tokens_per_opt_step = tokens_per_rank * self.dist.world_size * cfg.accumulate_grad_batches
        rank_zero_info(f"Starting training: max_steps={cfg.max_steps}")
        rank_zero_info(f"  confidence_only={cfg.train_confidence_only}")
        if cfg.train_confidence_only:
            rank_zero_info(
                "  confidence-only: diffusion training denoise skipped (no diffusion loss); "
                f"mini-rollout kept ({cfg.mini_rollout_steps} steps x {cfg.mini_rollout_samples} samples)"
            )
        rank_zero_info(f"  lr={cfg.lr}, warmup={cfg.warmup_steps}, scheduler={cfg.finetune_scheduler}")
        rank_zero_info(
            f"  crop: max_tokens={cfg.max_tokens}, max_atoms={cfg.max_atoms}, "
            f"max_msa_seqs={cfg.max_msa_seqs}, max_recycles={cfg.max_recycles}"
        )
        rank_zero_info(
            f"  batch: world_size={self.dist.world_size}, batch_per_rank={cfg.batch_size_per_rank}, "
            f"accumulate={cfg.accumulate_grad_batches} → "
            f"effective_batch_size={effective_batch_size} (tokens_per_opt_step={tokens_per_opt_step})"
        )
        rank_zero_info(
            f"  diffusion: batch_size={cfg.diffusion_batch_size}, chunk_size={cfg.diffusion_chunk_size}, "
            f"mini_rollout_steps={cfg.mini_rollout_steps}, mini_rollout_samples={cfg.mini_rollout_samples}"
        )
        rank_zero_info(f"  dtype={cfg.dtype}, grad_clip={cfg.grad_clip_norm}, ema_decay={cfg.ema_decay}")
        if (
            cfg.eval_snapshot == "average"
            and getattr(self.optimizer, "eval_mode", None) is None
            and self.ema is not None
        ):
            rank_zero_info(
                "  eval_snapshot='average' on a non-Schedule-Free optimizer with EMA active — "
                "AdamW has no averaged iterate; evaluating on the EMA shadow (the deployment weights)."
            )

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Advance through ``state.epoch`` (carries over a resumed value
        # when present in the checkpoint, or is derived from
        # ``effective_step // steps_per_epoch`` for old checkpoints).
        meters = DistributedAverageMeters(self.dist.device)
        t0 = time.monotonic()
        t_last_ckpt = t0
        # Per-effective-step GPU time via CUDA events (window sum → dt mean /
        # throughput; per-effective-step min/max → burst tail). Event-timed so
        # the reported dt no longer relies on an incidental CPU sync to be
        # accurate; elapsed is read once at the log boundary. Excludes eval /
        # checkpoint / dataloader wait (bracketed per microstep, after the
        # fetch). CPU-monotonic fallback when CUDA is unavailable.
        step_time_meter = StepTimeMeter(self.dist.device)
        # Tier-4: dataloader-wait accumulator — wall-clock blocked on the next
        # batch (``next(iter)``), i.e. the gap between the end of one loop body
        # (after eval/checkpoint) and the start of the next microstep. Surfaces
        # prefetch starvation: when this is non-trivial the dataloader (not
        # compute) is the per-step bottleneck. Flushed at each summary log.
        microstep_data_wait = 0.0
        t_iter_end: float | None = None
        # Tier-4d: per-microstep CUDA caching-allocator activity (cheap, no
        # sync — reads cached counters). Non-zero ``retries`` / ``device_alloc``
        # at a slow step ⇒ allocator fragmentation forced a free-and-retry /
        # cudaMalloc (a device stall), i.e. the spike is allocator-bound, not a
        # GPU throttle. Captured at microstep start, delta logged in SLOW_STEP.
        slow_step_on = cfg.slow_step_log_threshold_s > 0 and torch.cuda.is_available()
        _retries0 = 0
        _devalloc0 = 0
        # Tier-4c per-phase walls (set each microstep by ``_training_step`` /
        # the opt-step timer; 0.0 unless ``slow_step_log_threshold_s > 0``).
        self._last_fwd_wall = 0.0
        self._last_loss_wall = 0.0

        # Grad-norm window accumulators. ``last_grad_norm`` is a single
        # opt-step sample, so a bare read at the log boundary drops the
        # intermediate optimizer steps whenever ``log_interval > 1``. Mirror
        # the loss meters: accumulate every advanced step and report the
        # window mean (plus max, which the mean would smooth spikes out of).
        # On-GPU f64 accumulation keeps this collective- and sync-free until
        # the ``.item()`` at the (already-synchronized) log boundary.
        gn_sum = torch.zeros((), device=self.dist.device, dtype=torch.float64)
        gn_max = torch.zeros((), device=self.dist.device, dtype=torch.float64)
        gn_count = 0
        # Post-all-reduce global grad norm — only populated in per-rank-clip
        # mode (``last_grad_norm_global`` is None otherwise). Tracks the norm
        # of the averaged, already-clipped gradient the optimizer steps on.
        gn_global_sum = torch.zeros((), device=self.dist.device, dtype=torch.float64)
        gn_global_count = 0

        # SIGUSR2 → request immediate checkpoint (preemption safety)
        # SIGTERM/SIGINT/DF3_STOP_FILE → checkpoint + graceful exit (SLURM walltime, kill, Ctrl+C)
        shutdown = ShutdownCoordinator(self.dist)
        ckpt_request = CheckpointRequestHandler()

        # Single-track stability probe (file-only logging, no WandB).
        # Attached only on rank 0 — the per-block stats are local to each
        # rank's forward and we don't need a cross-rank reduction.
        self._single_track_probe = None
        if cfg.single_track_probe and self.dist.is_main:
            from deepfold3.training.single_track_probe import SingleTrackProbe

            probe_path = output_dir / "single_track_probe.jsonl"
            self._single_track_probe = SingleTrackProbe(self.model, probe_path)
            rank_zero_info(f"Single-track probe enabled → {probe_path} (every {cfg.single_track_probe_interval} steps)")

        # Per-PF-layer activation RMS sentinel. Attached on every rank (the
        # flush does a packed all_reduce so cross-rank averaging is correct).
        # Enabled only around the training forward; eval / mini-rollout
        # forwards run with ``meter.enabled = False``.
        self._act_meter = None
        if cfg.log_activation_rms:
            from deepfold3.training.activation_meter import attach_activation_meter

            self._act_meter = attach_activation_meter(self.model, self.dist.device)
            rank_zero_info(
                f"Activation-RMS meter attached ({self._act_meter.num_points} points × 2 tracks, "
                f"flush every {cfg.log_interval} steps)"
            )
        # MHR-TriMul router read-out rides the same window as the activation
        # meter. ``router/entropy`` at 1.0 means the router never left uniform,
        # i.e. it is inert — the one question the combined r17 arm can answer
        # from inside the run rather than from a second run.
        from deepfold3.model.ops import collect_router_stats, set_router_stats_enabled

        self._collect_router_stats = collect_router_stats
        self._set_router_stats_enabled = set_router_stats_enabled

        if cfg.eval_before and eval_dl is not None:
            self._run_eval_and_track_best(eval_dl, output_dir)

        while self.state.effective_step < self.cfg.max_steps:
            sampler = train_dl.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.state.epoch)

            for features in train_dl:
                if self.state.effective_step >= self.cfg.max_steps:
                    break

                # With an infinite sampler this ``for`` never exhausts, so the
                # iterator is created once and workers prefetch across epoch
                # reseeds — no per-epoch iterator recreation / prefetch-drain
                # data_wait spike. The while-level ``epoch += 1`` below is then
                # unreachable, so mirror the sampler's self-advancing epoch here
                # for logging/checkpoint. (Finite/DistributedSampler fallback:
                # this is a no-op and ``epoch += 1`` still drives the counter.)
                self.state.epoch = getattr(sampler, "epoch", self.state.epoch)

                t_microstep_start = time.monotonic()
                # Tier-4: time blocked waiting for this batch (prefetch drained).
                if t_iter_end is not None:
                    microstep_data_wait += t_microstep_start - t_iter_end
                # Tier-4d: snapshot allocator counters before the microstep.
                if slow_step_on:
                    _ms = torch.cuda.memory_stats(self.dist.device)
                    _retries0 = _ms.get("num_alloc_retries", 0)
                    _devalloc0 = _ms.get("num_device_alloc", 0)
                # Non-tensor diagnostic: pop before the forward pass so the
                # model / ``prepare_features`` never see a string. Threaded
                # into the optimizer so NaN-skip warnings can name the
                # offending record(s).
                record_ids = features.pop("_record_id", None)
                # Single-track probe gate. Fire on the LAST microstep of the
                # accumulation window that builds toward a target effective
                # step matching the probe interval. Captures per-block stats
                # during this microstep's forward pass.
                probe = self._single_track_probe
                if probe is not None:
                    # Clear any stale flag from a skipped opt step (NaN/Inf)
                    # so non-probe microsteps don't accidentally accumulate.
                    probe.discard_step()
                    last_microstep_in_window = ((self.state.global_step + 1) % self.cfg.accumulate_grad_batches) == 0
                    target_step = self.state.effective_step + (1 if last_microstep_in_window else 0)
                    should_probe = (
                        last_microstep_in_window
                        and target_step > 0
                        and target_step % self.cfg.single_track_probe_interval == 0
                    )
                    if should_probe:
                        probe.begin_step(target_step)
                # No except-and-continue on OOM/exception: ``continue`` skips
                # the per-microstep shutdown_flag all_reduce + the sync-boundary
                # NaN all_gather, which desynchronizes collective participation
                # across ranks → peers hang on a 600s NCCL watchdog. Fail hard
                # so torchrun reaps the whole process group; the surviving
                # traceback (logged via the rank-tagged handler in train.py)
                # identifies the offending rank.
                # Activation-RMS meter is enabled only around the training
                # forward — eval / mini-rollout forwards run uninstrumented
                # so they don't contaminate train-window averages.
                if self._act_meter is not None:
                    self._act_meter.enabled = True
                    self._set_router_stats_enabled(self.model, True)
                # GPU-time bracket start (records on the current CUDA stream,
                # after the dataloader fetch so inter-microstep wait is out).
                step_time_meter.start()
                with torch.autocast(
                    device_type="cuda",
                    dtype=self.cfg.torch_dtype,
                    enabled=(self.cfg.dtype != "fp32"),
                ):
                    loss, loss_dict, loss_counts = self._training_step(features)
                if self._act_meter is not None:
                    self._act_meter.enabled = False
                    self._set_router_stats_enabled(self.model, False)

                # NaN/Inf detection is deferred to the opt-step boundary inside
                # ``_optimizer_step`` — a single all-reduce per window replaces
                # one-per-microstep. Backward runs unconditionally here; any
                # NaN grads are discarded at the boundary before the opt step.
                _t_bwd = time.monotonic()
                on_opt_step_boundary = self._optimizer_step(loss, record_ids=record_ids)
                # Tier-4c: aux backward + gradient all_reduce + opt step (GPU +
                # collective). With phase timing on, fwd/loss are synced in
                # training_step so this is the remaining bwd+collective phase.
                bwd_wall = time.monotonic() - _t_bwd
                # Close the GPU-time bracket (forward → optimizer step). The opt
                # step already CPU-synced on the gradient all-reduce (NaN
                # ``.item()`` consensus), so the collective wait is in-bracket.
                step_time_meter.stop(on_opt_step_boundary)

                # Accumulate grad norm for window mean/max logging. An advanced
                # opt step guarantees a finite ``last_grad_norm`` (NaN/Inf steps
                # skip and leave ``on_opt_step_boundary`` False), so no finite
                # guard is needed. GPU ops → no per-step CPU sync.
                if on_opt_step_boundary and self.state.last_grad_norm is not None:
                    # Prefer the cross-rank pre-clip summary. ``last_grad_norm``
                    # is this rank's own norm and only rank 0's is logged, so
                    # using it would hide a spike on any other rank. Falls back
                    # to the local value single-process, where they coincide.
                    _mean_src = self.state.last_grad_norm_rank_mean
                    _max_src = self.state.last_grad_norm_rank_max
                    gn_sum += (
                        (_mean_src if _mean_src is not None else self.state.last_grad_norm).detach().to(torch.float64)
                    )
                    _gn_max_step = (
                        (_max_src if _max_src is not None else self.state.last_grad_norm).detach().to(torch.float64)
                    )
                    gn_max = torch.maximum(gn_max, _gn_max_step)
                    gn_count += 1
                    if self.state.last_grad_norm_global is not None:
                        gn_global_sum += self.state.last_grad_norm_global.detach().to(torch.float64)
                        gn_global_count += 1

                # Flush probe row (rank-0 only); pass per-microstep loss
                # (total + components from ``loss_dict``) so the probe JSONL
                # is self-contained — plot scripts don't need to align with
                # the slower ``log_interval`` stdout cadence.
                if probe is not None and probe.enabled:
                    probe_extra = {"loss": float(loss.detach().item())}
                    for k, v in loss_dict.items():
                        if isinstance(v, torch.Tensor):
                            probe_extra[k] = float(v.detach().item())
                    probe.end_step(extra=probe_extra)

                # Accumulate logging metrics locally (no collective).
                # Inject token count so it rides the same all-reduce as losses.
                seq_mask = features.get("seq_mask")
                n_tokens = (
                    seq_mask.sum()
                    if seq_mask is not None
                    else torch.tensor(cfg.max_tokens * cfg.batch_size_per_rank, device=self.dist.device)
                )
                loss_dict["_tokens"] = n_tokens
                # Per-key n: a key emitted by only one of two batch elements
                # must not be averaged as if both had reported it.
                meters.update(loss_dict, n=loss_counts)

                # CPU wall-clock of the microstep — kept only for the per-step
                # SLOW_STEP threshold + log_microsteps below, which need an
                # immediate per-step value. The REPORTED dt / min / max are
                # event-timed via ``step_time_meter``; this is async-approximate
                # but adequate for a coarse per-step gate.
                dt_micro = time.monotonic() - t_microstep_start

                # Tier-4b: per-rank slow-STEP attribution (GPU-side, not loader).
                # Logs on EVERY rank (no collective) so a data-tied straggler
                # names its OWN record(s). At a spike step every rank is over
                # threshold (the all_reduce stalls them all), so the straggler
                # is identified by inspecting the logged records — the
                # pathological one stands out. Grep per-rank logs:
                # ``grep 'SLOW_STEP' logs/*/attempt_*/*/stdout.log``.
                if cfg.slow_step_log_threshold_s > 0 and dt_micro >= cfg.slow_step_log_threshold_s:
                    # Phase split (synced in training_step): fwd = forward+diff
                    # bwd GPU; loss = chain-perm CPU + aux-loss GPU (the CPU
                    # section between fwd and bwd); bwd = aux backward + grad
                    # all_reduce + opt step. A spiking ``loss`` on one rank ⇒
                    # CPU straggler (chain-perm on that record); all-uniform
                    # ``bwd`` ⇒ collective/network; spiking ``fwd`` ⇒ compute or
                    # (if retries/allocs > 0) allocator-fragmentation stall.
                    if slow_step_on:
                        _ms = torch.cuda.memory_stats(self.dist.device)
                        retries_d = _ms.get("num_alloc_retries", 0) - _retries0
                        devalloc_d = _ms.get("num_device_alloc", 0) - _devalloc0
                    else:
                        retries_d = devalloc_d = -1
                    logger.warning(
                        "SLOW_STEP rank=%d node=%d effstep=%d dt_micro=%.1fs "
                        "fwd=%.1fs loss=%.1fs bwd=%.1fs retries=%d allocs=%d records=%s",
                        self.dist.rank,
                        self.dist.rank // 8,
                        self.state.effective_step,
                        dt_micro,
                        self._last_fwd_wall,
                        self._last_loss_wall,
                        bwd_wall,
                        retries_d,
                        devalloc_d,
                        record_ids if record_ids is not None else "?",
                    )

                if cfg.log_microsteps:
                    samples_per_microstep = self.dist.world_size * cfg.batch_size_per_rank
                    samples_per_sec = samples_per_microstep / max(dt_micro, 1e-9)
                    # Position within the current accumulation window: 1..accumulate
                    # (the accumulate-th microstep is the sync step that fires
                    # optimizer.step and increments effective_step).  ``target`` is
                    # the effective_step this microstep is building toward — on the
                    # sync step it equals the just-incremented effective_step;
                    # otherwise it's effective_step+1.
                    in_step = ((self.state.global_step - 1) % cfg.accumulate_grad_batches) + 1
                    target = self.state.effective_step if on_opt_step_boundary else self.state.effective_step + 1
                    rank_zero_info(
                        f"micro: target={target} "
                        f"global={self.state.global_step} "
                        f"in_step={in_step}/{cfg.accumulate_grad_batches} "
                        f"recycles={self.state.last_num_recycles} "
                        f"loss={loss.item():.4f} "
                        f"dt_micro={dt_micro:.2f}s samples/s={samples_per_sec:.3f}"
                    )

                # Release forward-pass outputs and input features immediately.
                # Without this, local vars hold references to large GPU tensors
                # (denoised [S,N,M,3], atom_gt_positions [N,M,3], etc.) until
                # Python rebinds them at the next iteration — wasting GPU memory
                # during the dataloader fetch and gradient accumulation micro-steps.
                del loss, loss_dict, features

                # --- Logging ---
                # Gate fires after clip_grad_norm populates ``_last_grad_norm``.
                # Collective: ``meters.sync`` issues ``all_gather_object`` +
                # ``all_reduce``, so the gate MUST be identical on all ranks
                # (no per-rank conditions like ``meters.count > 0``).
                if on_opt_step_boundary and self.state.effective_step % self.cfg.log_interval == 0:
                    avg, sums = meters.sync()
                    meters.reset()

                    # Window grad norm: mean matches the loss meters, max keeps
                    # spike visibility a mean would hide. Computed on all ranks
                    # for symmetry (two ``.item()`` syncs per window, at the
                    # meters.sync collective boundary — negligible); only rank 0
                    # logs. Empty window (all steps skipped) → NaN.
                    if gn_count > 0:
                        grad_norm_mean = (gn_sum / gn_count).item()
                        grad_norm_window_max = gn_max.item()
                    else:
                        grad_norm_mean = float("nan")
                        grad_norm_window_max = float("nan")
                    grad_norm_global_mean = (
                        (gn_global_sum / gn_global_count).item() if gn_global_count > 0 else float("nan")
                    )
                    gn_sum.zero_()
                    gn_max.zero_()
                    gn_count = 0
                    gn_global_sum.zero_()
                    gn_global_count = 0

                    # Event-timed step time: window sum (→ dt mean / tok-s) plus
                    # per-effective-step min/max (→ burst tail). ``flush`` issues
                    # one device sync here; the meters all-reduce above already
                    # crossed a collective boundary so it's ~free. All ranks
                    # flush in lockstep (the gate is identical across ranks).
                    compute_time_window, dt_window_min, dt_window_max = step_time_meter.flush()

                    # Per-PF-layer activation RMS — packed all_reduce on every
                    # rank in lockstep with the meters sync above. Gated by
                    # ``act_rms_log_interval`` so the meter can accumulate
                    # multiple log windows before each ρ snapshot is emitted
                    # (reduces WandB cost ~10× without losing the slow-dynamic
                    # depth profile). The collective gate must remain identical
                    # across ranks — uses ``effective_step`` only.
                    act_rms = {}
                    if (
                        self._act_meter is not None
                        and self.cfg.act_rms_log_interval > 0
                        and self.state.effective_step % (self.cfg.log_interval * self.cfg.act_rms_log_interval) == 0
                    ):
                        act_rms = self._act_meter.flush_to_dict()
                        self._act_meter.reset()
                        act_rms.update(self._collect_router_stats(self.model))

                    # data_wait flush (compute_time_window came from the event
                    # meter above). Reset on all ranks so the flag can flip.
                    data_wait_window = microstep_data_wait
                    microstep_data_wait = 0.0
                    # data_wait is rank-local, but the DDP step clock is set by
                    # the SLOWEST rank's fetch (a straggler stalls peers inside
                    # the gradient all-reduce, where it shows up as ``dt``, not
                    # ``data_wait``, on the fast ranks). Reduce with MAX across
                    # ranks so the reported value reflects the worst rank. The
                    # gate above is identical on all ranks → collective-safe.
                    if self.dist.is_distributed:
                        _dw = torch.tensor(data_wait_window, device=self.dist.device)
                        torch_dist.all_reduce(_dw, op=torch_dist.ReduceOp.MAX)
                        data_wait_window = _dw.item()

                    if self.dist.is_main:
                        grad_norm = grad_norm_mean
                        elapsed = time.monotonic() - t0
                        # ``dt`` is per-effective-step compute time (forward+backward+step).
                        # Eval / checkpoint / logging time is intentionally excluded so the
                        # displayed rate/ETA reflect pure training throughput.
                        dt = compute_time_window / max(self.cfg.log_interval, 1)
                        # Per-effective-step dataloader wait, max over ranks (same
                        # denominator as ``dt``). Near 0 ⇒ prefetch fully hides the
                        # fetch; large relative to ``dt`` ⇒ prefetch-starved on the
                        # slowest rank.
                        data_wait = data_wait_window / max(self.cfg.log_interval, 1)
                        # Schedule-Free folds warmup + bias-correction into ``scheduled_lr``;
                        # ``lr`` stays at the configured peak. Prefer the scheduled value so
                        # the log/WandB curve reflects what's actually applied.
                        g = self.optimizer.param_groups[0]
                        lr = g.get("scheduled_lr", g["lr"])

                        # Throughput (global token sum from all ranks via all-reduce).
                        # Same compute-time denominator as ``dt`` for a consistent picture.
                        tok_per_sec = sums.pop("_tokens", 0) / max(compute_time_window, 1e-9)
                        avg.pop("_tokens", None)

                        # GPU memory
                        peak_mem_gb = torch.cuda.max_memory_allocated() / (1 << 30) if torch.cuda.is_available() else 0

                        # ETA
                        remaining = cfg.max_steps - self.state.effective_step
                        eta_sec = remaining * dt
                        eta_h, eta_rem = divmod(int(eta_sec), 3600)
                        eta_str = f"{eta_h}:{eta_rem // 60:02d}" if eta_h > 0 else f"{int(eta_sec // 60)}m"

                        parts = [f"step={self.state.effective_step}"]
                        parts.append(f"loss={avg.get('total', 0):.4f}")
                        for k, v in sorted(avg.items()):
                            if k != "total":
                                parts.append(f"{k}={v:.4f}")
                        parts.append(f"grad_norm={grad_norm:.3f}")
                        parts.append(f"gn_max={grad_norm_window_max:.3f}")
                        # Per-rank-clip mode only: norm after the manual
                        # all-reduce of the clipped per-rank grads.
                        if math.isfinite(grad_norm_global_mean):
                            parts.append(f"gn_global={grad_norm_global_mean:.3f}")
                        parts.append(f"lr={lr:.2e}")
                        if tok_per_sec >= 1000:
                            parts.append(f"tok/s={tok_per_sec / 1000:.1f}k")
                        else:
                            parts.append(f"tok/s={tok_per_sec:.0f}")
                        parts.append(f"mem={peak_mem_gb:.1f}G")
                        parts.append(f"eta={eta_str}")
                        parts.append(f"dt={dt:.1f}s")
                        parts.append(f"dt_min={dt_window_min:.1f}s")
                        parts.append(f"dt_max={dt_window_max:.1f}s")
                        parts.append(f"data_wait={data_wait:.3f}s")
                        parts.append(f"t={elapsed:.0f}s")
                        logger.info(" | ".join(parts))

                        payload = build_step_payload(
                            avg=avg,
                            effective_step=self.state.effective_step,
                            lr=lr,
                            epoch=self.state.epoch,
                            grad_norm=grad_norm,
                            grad_norm_max=grad_norm_window_max,
                            grad_norm_global=grad_norm_global_mean,
                            dt=dt,
                            dt_min=dt_window_min,
                            dt_max=dt_window_max,
                            tok_per_sec=tok_per_sec,
                            peak_mem_gb=peak_mem_gb,
                            data_wait=data_wait,
                        )
                        # Per-PF-layer activation RMS — already prefixed
                        # with ``act_rms/``; pass through verbatim so the
                        # WandB workspace's saved panel grouping picks it up.
                        payload.update(act_rms)
                        self.trackers.log(payload, step=self.state.effective_step)

                # --- Checkpoint (before eval so weights are saved even if eval crashes) ---
                # ``effective_step`` is identical on all ranks, so the
                # interval check decides locally without a broadcast.
                # The time-based and signal-based triggers are rank-0
                # local; we broadcast a single bool so save fires
                # synchronously on every rank (RNG-gather requirement).
                need_ckpt = on_opt_step_boundary and (self.state.effective_step % self.cfg.checkpoint_interval == 0)
                if on_opt_step_boundary and self.dist.is_main and not need_ckpt:
                    if (
                        cfg.checkpoint_every_hours > 0
                        and (time.monotonic() - t_last_ckpt) / 3600 >= cfg.checkpoint_every_hours
                    ):
                        need_ckpt = True
                    elif ckpt_request.consume():
                        need_ckpt = True
                        logger.info("Signal-triggered checkpoint requested")
                if self.dist.is_distributed and on_opt_step_boundary:
                    flag = torch.tensor(int(need_ckpt), device=self.dist.device)
                    torch_dist.broadcast(flag, src=0)
                    need_ckpt = bool(flag.item())

                if need_ckpt:
                    self._save_checkpoint(output_dir / f"checkpoint_{self.state.effective_step}.pt")
                    t_last_ckpt = time.monotonic()

                # --- Evaluation ---
                # Gate on ``on_opt_step_boundary`` so eval fires exactly once
                # per effective step — without the gate, ``effective_step``
                # stays constant across ``accumulate_grad_batches`` microsteps
                # and the check would trigger on every one of them.
                if (
                    on_opt_step_boundary
                    and eval_dl is not None
                    and self.state.effective_step % self.cfg.eval_interval == 0
                ):
                    self._run_eval_and_track_best(eval_dl, output_dir)

                # --- Shutdown ---
                # Propagate on every microstep (cheap 1-elem all_reduce) so
                # all ranks bail together, not just at opt-step boundaries.
                shutdown_now = shutdown.tick()

                if shutdown_now:
                    # Mid-window: drop partial grads and rewind global_step
                    # to the last completed boundary so resume starts a fresh
                    # accumulation window. scheduler / scaler / EMA stay as-is
                    # (they only advance on sync steps).
                    partial = 0
                    if not on_opt_step_boundary:
                        partial = self.state.global_step - self.state.effective_step * cfg.accumulate_grad_batches
                        self.optimizer.zero_grad(set_to_none=True)
                        self.state.global_step = self.state.effective_step * cfg.accumulate_grad_batches
                    if self.dist.is_main:
                        if partial:
                            logger.info(
                                f"Shutdown mid-window at step {self.state.effective_step}: discarded "
                                f"{partial}/{cfg.accumulate_grad_batches} partial microsteps"
                            )
                        else:
                            logger.info(f"Shutdown at step {self.state.effective_step}")
                    # All ranks must call save so the RNG-gather collective
                    # runs on every rank (rank-0 write gate is inside).
                    self._save_checkpoint(output_dir / f"checkpoint_{self.state.effective_step}.pt")
                    break

                # Tier-4: end of this iteration's work. The next iteration's
                # ``t_microstep_start`` minus this marks pure dataloader wait
                # (eval / checkpoint / logging wall-clock above is excluded).
                t_iter_end = time.monotonic()

            if shutdown.requested:
                break
            self.state.epoch += 1

        # Final checkpoint — call on every rank so the cross-rank RNG
        # gather inside ``CheckpointManager.save`` runs on all ranks
        # (the rank-0 write gate is inside the manager).
        if not shutdown.requested:
            self._save_checkpoint(output_dir / "checkpoint_final.pt")
        # The final checkpoint's fsync / publish runs on a background thread;
        # drain it before teardown so it reaches storage (and any failure is
        # raised here rather than lost on exit).
        self.ckpt_manager.wait_for_io()
        if self.dist.is_main:
            logger.info(
                f"Training {'interrupted' if shutdown.requested else 'complete'} at step {self.state.effective_step}"
            )

        if self.trackers is not None:
            self.trackers.finish()

        # Collective teardown — see ``destroy_process_group_safely`` for the
        # ordering rationale (must live outside ``try/finally``).
        destroy_process_group_safely()
