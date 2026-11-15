"""Checkpoint save/load/rotate + pretrained-weight loading.

On-disk format adds (over the pre-refactor blob): ``epoch``,
``rng_states`` (one entry per DDP rank; gathered on save, scattered
on load), ``scaler``, and ``sampler``. Older checkpoints missing
these keys load through fallbacks: epoch derived from
``effective_step``; sampler ``sample_idx`` fast-forwarded from
``(effective_step % steps_per_epoch) * samples_per_step_per_rank``;
RNGs left at the seeds set by :func:`seed_everything`; scaler at
default. Each fallback emits a rank-zero warning so the divergence
from a step-exact resume is visible in the log.
"""

from __future__ import annotations

import logging
import os
import random
import re
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from deepfold3.data.io_cache import drop_from_page_cache
from deepfold3.model.config import ModelConfig, TrainingConfig
from deepfold3.training.distributed import DistInfo
from deepfold3.training.ema import EMA
from deepfold3.training.rank_zero import rank_zero_info, rank_zero_warn
from deepfold3.training.schedule_free import AdamWScheduleFree, optimizer_eval_mode
from deepfold3.training.scheduler import AF3LRScheduler
from deepfold3.training.state import TrainState

logger = logging.getLogger(__name__)


def _evict_after_read(path: str | Path) -> None:
    """Drop a just-read checkpoint from the page cache.

    A checkpoint is read once per process and never again, but every rank on a
    node reads its own copy — 8× the blob size of cache per node, at the moment
    the trainer is about to fork dataloader workers. The pages are already
    clean after a read, so no ``fsync`` is needed here.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        drop_from_page_cache(fd)
    finally:
        os.close(fd)


def load_pretrained_weights(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool,
    device: torch.device,
) -> None:
    """Load pretrained weights from ``.safetensors`` or ``.pt``.

    Mirrors the previous ``Trainer._load_pretrained`` behavior: a
    ``.pt`` file may contain either a bare state_dict or a checkpoint
    dict with a ``"model"`` key.
    """
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(p), device=str(device))
        _evict_after_read(p)
        model.load_state_dict(state, strict=strict)
        rank_zero_info(f"Loaded pretrained weights from {p}")
    elif p.suffix == ".pt":
        ckpt = torch.load(str(p), map_location=device, weights_only=False)
        _evict_after_read(p)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=strict)
        rank_zero_info(f"Loaded pretrained weights from {p}")


def _rng_state_local() -> dict:
    """Snapshot the four RNG sources we re-seed in :func:`seed_everything`."""
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_state,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def _gather_rng_states(dist_info: DistInfo) -> list[dict]:
    """Collect each rank's RNG state into a list indexed by rank.

    Saved on rank 0 only, but every rank's state is captured so a later
    load scatters the right slot back to each rank. On single-rank runs
    skips the all-gather entirely.
    """
    local = _rng_state_local()
    if not dist_info.is_distributed:
        return [local]
    gathered: list[dict | None] = [None] * dist_info.world_size
    dist.all_gather_object(gathered, local)
    return gathered  # type: ignore[return-value]


def _restore_rng_state_local(state: dict) -> None:
    # scatter_object_list / all_gather_object pickle-roundtrip strips the
    # ByteTensor subclass, leaving a plain uint8 tensor. set_rng_state
    # rejects that with "RNG state must be a torch.ByteTensor"; cast back.
    torch.set_rng_state(state["torch_cpu"].cpu().byte())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_state.cpu().byte())
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def _scatter_rng_states(dist_info: DistInfo, all_states: list[dict] | None) -> dict | None:
    """Send each rank's slot from ``all_states`` (held on rank 0) back to it.

    ``all_states is None`` means "old checkpoint without RNG capture" —
    propagated to every rank as ``None``.
    """
    if not dist_info.is_distributed:
        return all_states[0] if all_states else None
    payload: list = [None]
    if dist_info.is_main:
        if all_states is None:
            payload = [None] * dist_info.world_size
        elif len(all_states) != dist_info.world_size:
            # World size shrank/grew between save and resume — broadcast
            # rank 0's slot to everyone as a degraded fallback rather than
            # silently mis-mapping.
            payload = [all_states[0]] * dist_info.world_size
        else:
            payload = list(all_states)
    else:
        payload = [None] * dist_info.world_size
    out: list = [None]
    dist.scatter_object_list(out, payload if dist_info.is_main else None, src=0)
    return out[0]


class CheckpointManager:
    """Save + load + rotate + find-latest for the training checkpoint blob.

    A thin value-free utility — every public method takes the pieces
    it needs explicitly. Holds no state so the trainer can instantiate
    one and call methods directly.
    """

    def __init__(
        self,
        *,
        max_checkpoints: int = 0,
        strict: bool,
        device: torch.device,
        staging_dir: str | Path | None = None,
    ) -> None:
        self.max_checkpoints = max_checkpoints
        self.strict = strict
        self.device = device
        self.staging_dir = Path(staging_dir) if staging_dir else None
        # ``max_workers=1`` is the no-overlap guarantee: a second save's I/O
        # queues behind the first instead of racing it for staging disk.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt-io")
        self._io_future: Future | None = None

    # ----- post-write I/O, off the training thread -----

    def wait_for_io(self) -> None:
        """Block until this save's background file I/O has finished.

        Called before the next save and at shutdown, so the process never
        exits mid-publish. Re-raises a background failure on the caller.
        """
        if self._io_future is not None:
            pending, self._io_future = self._io_future, None
            pending.result()

    @staticmethod
    def _flush_and_evict(path: Path) -> None:
        """Force the checkpoint to storage, then drop it from the page cache.

        Nothing on this node reads a checkpoint again, so those pages are pure
        cgroup pressure — several GB of it per save, on rank 0's node only,
        which is the shape of a single-node stall.

        Both halves have to happen here rather than inline after
        ``torch.save``. ``DONTNEED`` only drops *clean* pages, so it needs the
        ``fsync`` first; and an inline ``fsync`` would serialize the multi-GB
        shared-storage write into the training step while every other rank
        waits at the next collective. By the time this runs the blob is fully
        written, so the thread needs only the path — no race with training.
        """
        with open(path, "rb") as f:
            os.fsync(f.fileno())
            drop_from_page_cache(f.fileno())

    def _publish(self, staged: Path, final: Path) -> None:
        """Copy ``staged`` → ``final``, then rotate. Runs off the training thread.

        The copy goes to a ``.tmp`` sibling and is renamed into place, so a
        crash mid-publish leaves no half-written ``checkpoint_*.pt`` for
        ``find_latest`` to pick up on resume. ``copyfile`` dispatches to
        ``sendfile`` on Linux, so the bytes never pass through this process.
        """
        tmp = final.with_suffix(final.suffix + ".tmp")
        shutil.copyfile(staged, tmp)
        self._flush_and_evict(tmp)
        os.replace(tmp, final)
        # Releases the staged pages too, so no DONTNEED is needed on the source.
        staged.unlink(missing_ok=True)
        logger.info(f"Published checkpoint to {final}")
        if self.max_checkpoints > 0:
            self._rotate(final.parent)

    # ----- save path -----

    def save(
        self,
        path: Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: AF3LRScheduler,
        ema: EMA | None,
        scaler: torch.amp.GradScaler | None,
        state: TrainState,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        dist_info: DistInfo,
        train_sampler: object | None = None,
    ) -> None:
        """Save the checkpoint blob + rotate old files.

        Keys match the pre-refactor format exactly so existing
        checkpoints remain loadable by this module.

        For Schedule-Free optimizers, ``optimizer_eval_mode`` flips live
        params from ``y`` (gradient point) to ``x`` (inference target)
        for the duration of the save so ``model.state_dict()`` captures
        the inference weights — matching the published library's
        contract ("the optimizer should be placed in eval mode when
        storing checkpoints"). Restored to ``y`` on context exit. No-op
        for plain AdamW.
        """
        # All ranks participate in the gather; only rank 0 actually writes.
        rng_states = _gather_rng_states(dist_info)
        if not dist_info.is_main:
            return

        self.wait_for_io()

        # With a staging dir the blocking write goes to node-local disk and the
        # much slower shared-storage write moves to a background thread. Every
        # other rank is parked in the next collective for the blocking part, so
        # that is the section that costs the whole world its step time.
        if self.staging_dir is not None:
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            write_path = self.staging_dir / path.name
        else:
            write_path = path

        with optimizer_eval_mode(optimizer):
            # snapshot_kind: descriptor for what's in ``blob["model"]`` to aid
            # inference-time selection. Schedule-Free flips into eval mode
            # above, so model.state_dict() captures ``x`` (the SF averaged
            # iterate); plain AdamW captures the live params. EMA shadow,
            # when present, is always EMA of this snapshot — inference
            # composes by overlay (see scripts/_common.py:select_snapshot).
            snapshot_kind = "x" if isinstance(optimizer, AdamWScheduleFree) else "live"
            # Flat optimizer-index → parameter-name map. ``optimizer.state_dict()``
            # keys its per-parameter state (including Schedule-Free's ``z``) by a
            # flat index that runs in *param-group* order — ``build_optimizer``
            # emits a decay group followed by a no-decay group, so the index
            # order is NOT ``state_dict()`` order (which additionally interleaves
            # buffers). Recording the mapping here lets ``select_snapshot`` fetch
            # ``z`` without replaying the grouping heuristics. See
            # ``scripts/_common.py:select_snapshot``.
            _param_name = {id(p): n for n, p in model.named_parameters()}
            optimizer_param_names = [
                _param_name[id(p)] for group in optimizer.param_groups for p in group["params"] if id(p) in _param_name
            ]
            blob: dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "optimizer_param_names": optimizer_param_names,
                "scheduler": scheduler.state_dict(),
                "global_step": state.global_step,
                "effective_step": state.effective_step,
                "epoch": state.epoch,
                "best_eval_loss": state.best_eval_loss,
                "rng_states": rng_states,
                "model_cfg": model_cfg,
                "train_cfg": train_cfg,
                "snapshot_kind": snapshot_kind,
            }
            if ema is not None:
                blob["ema"] = ema.state_dict()
            if scaler is not None and scaler.is_enabled():
                blob["scaler"] = scaler.state_dict()
            # Sampler state: per-rank, captured at save time (rank 0
            # records its own position; other ranks may be at a slightly
            # different index due to DataLoader prefetch, but they all
            # work from the same shard so resume re-shards uniformly).
            if train_sampler is not None and hasattr(train_sampler, "state_dict"):
                blob["sampler"] = train_sampler.state_dict()

            torch.save(blob, write_path)

        # Outside ``optimizer_eval_mode``: the blob is on disk, so params can
        # go back to ``y`` before the (slow) publish starts.
        if write_path != path:
            logger.info(f"Staged checkpoint to {write_path}; publishing to {path} in background")
            self._io_future = self._pool.submit(self._publish, write_path, path)
        else:
            logger.info(f"Saved checkpoint to {path}")
            self._io_future = self._pool.submit(self._flush_and_evict, path)
            if self.max_checkpoints > 0:
                self._rotate(path.parent)

    def _rotate(self, output_dir: Path) -> None:
        """Keep only the N most recent ``checkpoint_*.pt`` files.

        ``checkpoint_final.pt`` and ``checkpoint_best.pt`` are never
        rotated out.
        """
        ckpts = sorted(output_dir.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime)
        ckpts = [c for c in ckpts if "final" not in c.stem and "best" not in c.stem]
        while len(ckpts) > self.max_checkpoints:
            old = ckpts.pop(0)
            old.unlink()
            logger.info(f"Removed old checkpoint: {old}")

    # ----- load path -----

    def load(
        self,
        path: Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: AF3LRScheduler,
        ema: EMA | None,
        scaler: torch.amp.GradScaler | None,
        state: TrainState,
        train_cfg: TrainingConfig,
        dist_info: DistInfo,
        steps_per_epoch: int | None = None,
        train_sampler: object | None = None,
        samples_per_step_per_rank: int = 1,
        apply_skip_flags: bool = True,
    ) -> None:
        """Resume training state from ``path``, mutating the live objects in place.

        Handles DDP ``module.`` prefix stripping automatically.
        Partial-load gates (``skip_load_optimizer`` / ``skip_load_scheduler`` /
        ``skip_load_step``) follow the training config when
        ``apply_skip_flags`` is true. The trainer passes ``False`` when it
        auto-resumes from the run's own output directory: those flags start a
        new stage fresh, and applying them to the run's own checkpoint would
        restart every chain link at step 0 with a new optimizer and warmup.

        If the checkpoint includes per-rank ``rng_states``, each rank's
        slot is scattered back to it; otherwise RNGs are left at the
        seed values set by :func:`seed_everything`. Same for ``scaler``
        and ``epoch``: missing keys fall back to defaults / derivation
        from ``effective_step``.

        ``samples_per_step_per_rank`` (= ``batch_size_per_rank ×
        accumulate_grad_batches``) is used only when the checkpoint has
        no ``sampler`` key: the manager fast-forwards
        :class:`WeightedClusterSampler`'s ``sample_idx`` to
        ``(effective_step % steps_per_epoch) × samples_per_step_per_rank``
        so resume skips records already trained — at the cost of a
        differing RNG trajectory (dropout / EDM noise / shuffle).
        Each missing-key fallback logs a rank-zero warning.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        _evict_after_read(path)

        state_dict = ckpt["model"]
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=self.strict)

        skip_optimizer = apply_skip_flags and train_cfg.skip_load_optimizer
        skip_scheduler = apply_skip_flags and train_cfg.skip_load_scheduler
        skip_step = apply_skip_flags and train_cfg.skip_load_step

        if "optimizer" in ckpt and not skip_optimizer:
            # G5 invariant: SF saves with ``train_mode=False`` so the live
            # params loaded from ``ckpt["model"]`` are ``x``. The follow-up
            # ``optimizer.train()`` then lerps ``x → y``. If the saved
            # state thought it was in train mode, the lerp would be the
            # wrong direction → silent param drift.
            opt_state = ckpt["optimizer"]
            if hasattr(optimizer, "eval_mode"):
                groups = opt_state.get("param_groups", [])
                if groups and groups[0].get("train_mode", False):
                    raise RuntimeError(
                        f"Schedule-Free checkpoint at {path} was saved with train_mode=True; "
                        "expected False (saver should wrap with optimizer_eval_mode)."
                    )
            optimizer.load_state_dict(opt_state)
        # Schedule-Free was saved with ``train_mode=False`` (live param =
        # x) so ``load_state_dict`` restored the same. Lerp ``p ← y`` from
        # ``(x, z)`` so training resumes at the same gradient point as
        # just before the save. ``eval_mode`` is the SF marker — plain
        # AdamW lacks both ``.train()`` and ``.eval_mode``. Calling
        # ``train()`` while already in train mode is a no-op (idempotent),
        # so a checkpoint pre-dating the eval-on-save behavior is handled.
        if hasattr(optimizer, "eval_mode"):
            optimizer.train()
        if "scheduler" in ckpt and not skip_scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "ema" in ckpt and ema is not None:
            ema.load_state_dict(ckpt["ema"])
        if scaler is not None and scaler.is_enabled():
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            else:
                rank_zero_warn(
                    f"Checkpoint {path} has no 'scaler' state; using default GradScaler "
                    "(scale will be re-estimated, may cause skipped steps early on)."
                )

        if not skip_step:
            state.global_step = ckpt.get("global_step", 0)
            state.effective_step = ckpt.get("effective_step", 0)
            if "epoch" in ckpt:
                state.epoch = ckpt["epoch"]
            elif steps_per_epoch and steps_per_epoch > 0:
                # Old checkpoint without ``epoch`` key: derive from
                # opt-step count so the sampler RNG resumes near where
                # it would have been (off by at most one boundary).
                state.epoch = state.effective_step // steps_per_epoch
                rank_zero_info(
                    f"Checkpoint {path} has no 'epoch' key; derived "
                    f"epoch={state.epoch} = effective_step={state.effective_step} "
                    f"// steps_per_epoch={steps_per_epoch}."
                )
        state.best_eval_loss = ckpt.get("best_eval_loss", float("inf"))

        # Scatter per-rank RNG states (rank 0 holds the list; everyone
        # else receives ``None`` and skips). Missing in old ckpts → rank
        # 0 broadcasts ``None`` and seed_everything's state stands.
        if "rng_states" not in ckpt:
            rank_zero_warn(
                f"Checkpoint {path} has no 'rng_states'; per-rank RNG (CPU/CUDA "
                "torch, python, numpy) stays at seed_everything values. DataLoader "
                "shuffle / dropout / EDM noise trajectory will diverge from save point."
            )
        rng_all = ckpt.get("rng_states") if dist_info.is_main else None
        my_rng = _scatter_rng_states(dist_info, rng_all)
        if my_rng is not None:
            _restore_rng_state_local(my_rng)

        # Sampler state: same epoch + within-epoch index across ranks
        # (each rank yields its own shard of the same RNG-derived index
        # list, so identical sample_idx is the right resume target).
        if train_sampler is not None and "sampler" in ckpt and hasattr(train_sampler, "load_state_dict"):
            train_sampler.load_state_dict(ckpt["sampler"])
        elif (
            train_sampler is not None
            and steps_per_epoch
            and steps_per_epoch > 0
            and hasattr(train_sampler, "set_epoch")
            and hasattr(train_sampler, "sample_idx")
        ):
            # Old checkpoint: derive within-epoch position from
            # effective_step. Set ``epoch`` first so the trainer's
            # subsequent ``set_epoch(state.epoch)`` is a no-op and
            # doesn't reset ``sample_idx`` back to 0. The fast-forward
            # skips records already trained on; the RNG trajectory
            # (shuffle/dropout/noise) still diverges from the original
            # run, so this is "approximate resume" — louder warning.
            step_in_epoch = state.effective_step % steps_per_epoch
            sample_idx_est = step_in_epoch * samples_per_step_per_rank
            train_sampler.epoch = state.epoch
            train_sampler.sample_idx = sample_idx_est
            rank_zero_warn(
                f"Checkpoint {path} has no 'sampler' state; fast-forwarding "
                f"{type(train_sampler).__name__} to epoch={state.epoch}, "
                f"sample_idx={sample_idx_est} "
                f"(= ({state.effective_step} % {steps_per_epoch}) × "
                f"{samples_per_step_per_rank}). RNG trajectory will differ "
                "from the original run."
            )

        sampler_idx = getattr(train_sampler, "sample_idx", None) if train_sampler is not None else None
        suffix = f", epoch={state.epoch}"
        if sampler_idx is not None:
            suffix += f", sample_idx={sampler_idx}"
        rank_zero_info(f"Resumed from {path} at step {state.effective_step}{suffix}")

    # ----- discovery -----

    @staticmethod
    def find_latest(output_dir: Path) -> Path | None:
        """Find the ``checkpoint_*.pt`` file with the highest step number."""
        best_step, best_path = -1, None
        for p in output_dir.glob("checkpoint_*.pt"):
            m = re.search(r"checkpoint_(\d+)\.pt$", p.name)
            if m:
                step = int(m.group(1))
                if step > best_step:
                    best_step, best_path = step, p
        return best_path
