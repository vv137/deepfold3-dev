"""Signal-driven shutdown + immediate-checkpoint coordination.

Two tiny helpers:

* :class:`ShutdownCoordinator` owns the SIGTERM/SIGINT handlers, the
  ``DF3_STOP_FILE`` poll on rank 0, and a per-microstep collective
  (``MAX`` all_reduce) that propagates one rank's shutdown request to
  every peer.
* :class:`CheckpointRequestHandler` owns the SIGUSR2 handler for
  "please checkpoint now" signals (used by SLURM preemption hooks and
  manual ``kill -USR2``).

Kept in a dedicated module so the collective contract (MAX over a
1-element int tensor, every microstep) lives in one place.
"""

from __future__ import annotations

import logging
import os
import platform
import signal

import torch
import torch.distributed as dist

from deepfold3.training.distributed import DistInfo

logger = logging.getLogger(__name__)


class ShutdownCoordinator:
    """SIGTERM/SIGINT handler + per-microstep collective fan-out.

    Rationale for ``MAX`` over ``OR``: the shutdown flag is int-valued
    (0/1) rather than a bitmask, so ``MAX`` gives the same semantics as
    ``OR`` while matching the underlying NCCL op surface. The collective
    runs EVERY microstep (cheap 1-elem all_reduce on a preallocated
    buffer) so all ranks bail together rather than waiting for the next
    opt-step boundary.

    ``DF3_STOP_FILE``, when set, names a file whose appearance requests the
    same shutdown. Rank 0 polls it in :meth:`tick`. A batch script cannot use
    SIGTERM for this under srun: signalling srun cancels the step, and
    signalling the step also reaches DataLoader workers and the wandb service.
    """

    def __init__(self, dist_info: DistInfo) -> None:
        self.dist = dist_info
        self.requested = False
        self._stop_file = os.environ.get("DF3_STOP_FILE")
        self._flag = torch.zeros(1, dtype=torch.int32, device=dist_info.device)
        if platform.system() != "Windows":
            signal.signal(signal.SIGTERM, lambda *_: self._set())
            signal.signal(signal.SIGINT, lambda *_: self._set())

    def _set(self) -> None:
        self.requested = True

    def tick(self) -> bool:
        """Run the per-microstep shutdown collective and return the post-reduce flag.

        Must be called by every rank in lockstep on every microstep.
        """
        if self._stop_file and self.dist.is_main and not self.requested and os.path.exists(self._stop_file):
            logger.info(f"Stop file {self._stop_file} found, shutting down")
            self.requested = True
        if self.dist.is_distributed:
            self._flag.fill_(int(self.requested))
            dist.all_reduce(self._flag, op=dist.ReduceOp.MAX)
            self.requested = bool(self._flag.item())
        return self.requested


class CheckpointRequestHandler:
    """SIGUSR2 handler for "please checkpoint ASAP" requests.

    Unlike shutdown, a checkpoint request is rank-0-only — the
    collective save happens on rank 0 from its normal ckpt cadence,
    this handler just raises a flag that the loop polls and clears.
    """

    def __init__(self) -> None:
        self._requested = False
        if platform.system() != "Windows":
            signal.signal(signal.SIGUSR2, lambda *_: self._set())

    def _set(self) -> None:
        self._requested = True

    def consume(self) -> bool:
        """Return + clear the flag."""
        out = self._requested
        self._requested = False
        return out
