"""Learning rate schedulers for DeepFold3.

Four schedulers:

1. ``AF3LRScheduler``: linear warmup → plateau → step decay (initial training).
2. ``CosineAnnealingScheduler``: linear warmup → cosine decay (fine-tuning).
3. ``WSDSScheduler``: warmup → stable plateau → inverse-proportional decay
   windows that resume back at ``max_lr`` (continual pretraining; Hägele et al.
   2024, *Warmup-Stable-Decay-Simplified*).
4. ``PiecewiseLRScheduler``: arbitrary list of ``[start, end, start_ratio, end_ratio]``
   segments. Each segment uses inverse-proportional interpolation between its
   endpoints (degenerates to constant when ``start_ratio == end_ratio``).
   Strictly more general than ``WSDSScheduler``: supports custom peak/valley
   per segment, instantaneous ratio jumps between segments, and a stay-at-end
   tail behavior. Used for ad-hoc experimental schedules.

All parameters are configurable via ``TrainingConfig``.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from deepfold3.model.config import TrainingConfig


class AF3LRScheduler(LambdaLR):
    """AF3-style learning rate scheduler.

    Three-phase schedule:

    1. **Warmup** (steps ``[0, warmup_steps)``): linear ramp ``0 -> max_lr``.
    2. **Plateau** (steps ``[warmup_steps, start_decay_after)``): constant ``max_lr``.
    3. **Decay** (steps ``>= start_decay_after``):
       ``max_lr * decay_factor ** (steps_since // decay_every)``.

    Args:
        optimizer: Optimizer to schedule.
        cfg: Training config with LR schedule parameters.
    """

    def __init__(self, optimizer: Optimizer, cfg: TrainingConfig):
        self.max_lr = cfg.lr
        self.warmup_steps = cfg.warmup_steps
        self.start_decay_after = cfg.start_decay_after
        self.decay_every = cfg.decay_every
        self.decay_factor = cfg.decay_factor

        super().__init__(optimizer, self._lr_lambda)

    def _lr_lambda(self, step: int) -> float:
        """Compute LR multiplier for the given step.

        Args:
            step: Current optimizer step.

        Returns:
            Multiplier in ``[0, 1]`` to apply to the base LR.
        """
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)

        if step < self.start_decay_after:
            return 1.0

        steps_since = step - self.start_decay_after
        n_decays = steps_since // max(1, self.decay_every)
        return self.decay_factor**n_decays


#: Decay shapes of :class:`CosineAnnealingScheduler`, as ``progress -> factor``
#: on ``[0, 1] -> [1, 0]``. ``1-sqrt`` is the cooldown Hägele et al. 2024
#: (arXiv:2405.18392) found best for warmup-stable-decay runs.
DECAY_SHAPES = {
    "cosine": lambda p: (1.0 + math.cos(math.pi * p)) / 2.0,
    "linear": lambda p: 1.0 - p,
    "1-sqrt": lambda p: 1.0 - math.sqrt(p),
}


class CosineAnnealingScheduler(LambdaLR):
    """Linear warmup, optional constant phase, then a decay to ``min_ratio``.

    1. **Warmup** (steps ``[0, warmup_steps)``): linear ramp ``0 -> max_lr``.
    2. **Stable** (steps ``[warmup_steps, decay_start)``): constant ``max_lr``.
       Empty when ``finetune_decay_start <= warmup_steps`` (the default 0).
    3. **Decay** (steps ``[decay_start, max_steps]``):
       ``max_lr * (min_ratio + (1 - min_ratio) * shape(progress))``, held at
       ``min_ratio`` after ``max_steps``. ``shape`` is ``finetune_scheduler``:
       ``cosine``, ``linear`` or ``1-sqrt`` (see :data:`DECAY_SHAPES`).

    Args:
        optimizer: Optimizer to schedule.
        cfg: Training config — reads ``warmup_steps``, ``finetune_decay_start``,
            ``finetune_max_steps``, ``finetune_min_lr_ratio``, ``finetune_scheduler``.
    """

    def __init__(self, optimizer: Optimizer, cfg: TrainingConfig):
        self.shape = cfg.finetune_scheduler if cfg.finetune_scheduler in DECAY_SHAPES else "cosine"
        self.warmup_steps = cfg.warmup_steps
        self.decay_start = cfg.finetune_decay_start
        self.max_steps = cfg.effective_finetune_max_steps
        self.min_ratio = cfg.effective_finetune_min_lr_ratio
        start = max(self.warmup_steps, self.decay_start)
        if self.max_steps <= start:
            raise ValueError(f"finetune_max_steps ({self.max_steps}) must be > the decay start ({start})")
        if not 0.0 <= self.min_ratio <= 1.0:
            raise ValueError(f"finetune_min_lr_ratio must be in [0, 1], got {self.min_ratio}")

        super().__init__(optimizer, self._lr_lambda)

    def _lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)

        start = max(self.warmup_steps, self.decay_start)
        progress = min(1.0, max(0.0, (step - start) / max(1, self.max_steps - start)))
        return self.min_ratio + (1.0 - self.min_ratio) * DECAY_SHAPES[self.shape](progress)


class WSDSScheduler(LambdaLR):
    """Warmup-Stable-Decay-Simplified (WSD-S, Hägele et al. 2024).

    Three regimes per step ``t``, with ``W = warmup_steps``,
    ``M = max_lr``, ``m = max_lr * min_ratio``, and decay windows
    ``{(D_i, T_i)}`` (sorted, non-overlapping):

    1. **Warmup** (``t < W``): linear ``0 → M``.
    2. **Decay** (``D_i ≤ t < T_i``): inverse-proportional
       ``1/lr(t) = (s/L)·(1/m) + (1 − s/L)·(1/M)`` where
       ``s = t − D_i`` and ``L = T_i − D_i``. At ``s = 0`` ``lr = M``;
       at ``s = L`` ``lr = m``.
    3. **Stable** (everywhere else): constant ``M``. After a decay window
       ends the schedule jumps straight back to ``M`` — no rollback.

    Args:
        optimizer: Optimizer to schedule.
        cfg: Training config — reads ``warmup_steps``,
            ``wsds_decay_windows``, ``wsds_min_lr_ratio``.
    """

    def __init__(self, optimizer: Optimizer, cfg: TrainingConfig):
        self.warmup_steps = cfg.warmup_steps
        windows = sorted([(int(s), int(e)) for s, e in cfg.wsds_decay_windows])
        for s, e in windows:
            if e <= s:
                raise ValueError(f"wsds_decay_windows: end ({e}) must be > start ({s})")
        for (s1, e1), (s2, e2) in zip(windows[:-1], windows[1:], strict=False):
            if s2 < e1:
                raise ValueError(f"wsds_decay_windows overlap: [{s1}, {e1}) and [{s2}, {e2})")
        self.windows = windows
        self.min_ratio = float(cfg.wsds_min_lr_ratio)
        if not 0.0 <= self.min_ratio <= 1.0:
            raise ValueError(f"wsds_min_lr_ratio must be in [0, 1], got {self.min_ratio}")

        super().__init__(optimizer, self._lr_lambda)

    def _lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)

        for start, end in self.windows:
            if start <= step < end:
                s = step - start
                length = end - start
                u = s / length  # fractional progress through the decay
                # Clamp ratio to avoid div-by-zero at exactly min_ratio == 0.
                m = max(self.min_ratio, 1e-8)
                inv_mult = u / m + (1.0 - u)
                return 1.0 / inv_mult

        return 1.0


def build_scheduler(optimizer: Optimizer, cfg: TrainingConfig) -> LambdaLR:
    """Build the appropriate scheduler based on config.

    Args:
        optimizer: Optimizer to schedule.
        cfg: Training config.

    Returns:
        AF3LRScheduler, CosineAnnealingScheduler, or WSDSScheduler.
        For ``optimizer_type='schedule_free'`` returns an identity
        :class:`LambdaLR` so the existing ``scheduler.step()`` call sites
        and checkpoint state_dict format keep working unchanged. SF
        manages its own warmup via ``warmup_steps`` and has no decay.
    """
    if cfg.optimizer_type == "schedule_free":
        # ``cfg.finetune_scheduler`` is intentionally ignored — SF has no
        # external schedule. Identity multiplier keeps ``param_groups[0]['lr']``
        # equal to the configured ``cfg.lr`` for the existing logging path.
        return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    name = cfg.finetune_scheduler
    if name in DECAY_SHAPES:
        return CosineAnnealingScheduler(optimizer, cfg)
    if name == "wsds":
        return WSDSScheduler(optimizer, cfg)
    if name == "af3":
        return AF3LRScheduler(optimizer, cfg)
    raise ValueError(f"finetune_scheduler must be 'af3', 'cosine', 'linear', '1-sqrt' or 'wsds'; got {name!r}")
