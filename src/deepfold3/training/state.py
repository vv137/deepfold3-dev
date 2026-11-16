"""Per-rank training state carried through the outer loop.

A single :class:`TrainState` dataclass replaces the scatter of
``self.global_step`` / ``self.effective_step`` / ``self._best_eval_loss``
attributes that used to live directly on :class:`Trainer`. Separating
state from the orchestrator lets pure-function helpers in
``training/step.py``, ``training/optim.py``, etc. mutate it explicitly
without reaching into a god-object.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class TrainState:
    """Run-local state that advances across optimizer steps.

    Attributes are mutated in place by the trainer and its helpers;
    this is intentional — the state is a one-of-one per run.

    Attributes:
        global_step: Microstep counter (includes accumulation steps).
        effective_step: Optimizer step counter (advances once per
            ``accumulate_grad_batches`` microsteps when no NaN skip).
        epoch: Current epoch index.
        best_eval_loss: Lowest ``eval/total`` seen so far (for best-ckpt
            tracking).
        consecutive_nan_skips: How many opt steps in a row were skipped
            due to non-finite loss or grads. Reset to 0 on a successful
            step. ``cfg.max_consecutive_nan_skips`` bounds this.
        last_grad_norm: Most recent grad norm used for the clip/skip
            decision (for logging). In per-rank-clip mode this is the
            per-rank pre-clip norm; otherwise the global pre-clip norm.
        last_grad_norm_global: Post-all-reduce global grad norm, set only in
            per-rank-clip mode (``None`` otherwise). This is the norm of the
            averaged, already-clipped gradient the optimizer actually steps
            on — bounded by ``grad_clip_norm``.
        last_grad_norm_rank_mean / last_grad_norm_rank_max: PRE-clip norms
            reduced across ranks, from the all-gather the NaN consensus
            already performs (no extra collective). ``last_grad_norm`` is
            rank-local and only rank 0's copy is ever logged, so a large but
            finite blow-up on any other rank was invisible: it gets clipped to
            ``grad_clip_norm`` and ``last_grad_norm_global`` is measured after
            that clip, so neither existing metric can reveal it.
        last_num_recycles: Number of recycles sampled for the last
            microstep (for microstep logging).
    """

    global_step: int = 0
    effective_step: int = 0
    epoch: int = 0
    best_eval_loss: float = field(default_factory=lambda: float("inf"))
    consecutive_nan_skips: int = 0
    last_grad_norm: torch.Tensor | None = None
    last_grad_norm_global: torch.Tensor | None = None
    last_grad_norm_rank_mean: torch.Tensor | None = None
    last_grad_norm_rank_max: torch.Tensor | None = None
    # Norm of the gradient the optimizer applies to ``diffusion_head.null_pair``
    # (clipped, rank-averaged; ``cond_drop_fill: learned``); None without it.
    last_null_pair_grad_norm: torch.Tensor | None = None
    last_num_recycles: int = 0


def seed_everything(seed: int, rank: int, *, deterministic: bool = False) -> None:
    """Set seeds for reproducibility, offset by rank.

    Offsetting by rank gives each rank an independent RNG stream while
    keeping the run as a whole deterministic with respect to the
    user-supplied seed.

    When ``deterministic=True``, also force torch/cuDNN to pick
    deterministic algorithms. This trades throughput for bit-exact
    reproducibility.
    """
    effective = seed + rank
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    if deterministic:
        # cuBLAS workspace must be configured before the first CUDA op;
        # skip silently if the user already set it.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
