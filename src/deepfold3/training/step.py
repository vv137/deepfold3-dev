"""Per-microstep forward + loss computation (pre-backward).

Moved out of ``trainer.py`` so the inner loop can be read without
paging through ~100 lines of forward-pass + per-batch loss reduction.
All functions are pure — they take explicit state (``global_step``,
``scaler``, ``accumulator``) rather than reaching into a trainer
instance.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

import torch
from torch.amp import GradScaler

from deepfold3.model.config import ModelConfig, TrainingConfig
from deepfold3.model.model import DeepFold3
from deepfold3.model.precision import fp32_guard
from deepfold3.training.accumulator import AccumulationLoop
from deepfold3.training.collectives import slice_batch_dict
from deepfold3.training.loss import training_loss
from deepfold3.training.optim import is_chunked_diffusion_active


def prepare_features(
    features: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move features to ``device``, keeping the leading batch dim.

    Keeping ``[B, ...]`` consistent makes every downstream module take
    the batched code path regardless of ``batch_size_per_rank``.
    """
    return {k: v.to(device, non_blocking=True) for k, v in features.items()}


def scaled_backward(scaler: GradScaler, loss: torch.Tensor, accumulate: int) -> None:
    """``scaler.scale(loss / accumulate).backward()``.

    Shared between the chunked-diffusion per-chunk callback and the
    outer aux backward in ``_optimizer_step``.
    """
    scaler.scale(loss / accumulate).backward()


def aggregate_sample_losses(
    per_sample: list[dict[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Mean each reported loss term over the samples that actually emitted it.

    Not every batch element emits every key: a distillation sample returns
    early before the confidence block, and ``tm_true`` / ``bond`` / ``fape``
    are feature- or alpha-gated. Dividing every key by the batch size
    under-reported those by ``n_present / batch_size`` — at
    ``batch_size_per_rank: 2`` with a ~50% distillation mix that halved the
    logged pLDDT / PAE / PDE / exp_resolved.

    Args:
        per_sample: One loss dict per batch element.

    Returns:
        ``(means, counts)`` — the per-key mean and the number of samples that
        contributed to it. ``counts`` is passed on to the meters so the
        cross-rank average is weighted by contribution too.
    """
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    for ld in per_sample:
        for k, v in ld.items():
            sums[k] = sums[k] + v if k in sums else v
            counts[k] = counts.get(k, 0) + 1
    return {k: v / counts[k] for k, v in sums.items()}, counts


def pick_num_recycles(cfg: TrainingConfig, global_step: int) -> int:
    """Sample ``num_recycles`` for this microstep.

    Two modes:

    * ``uniform_recycle=True`` (default): all DDP ranks pick the SAME
      ``num_recycles`` via a fresh ``random.Random`` seeded from
      ``(cfg.seed, global_step)`` — both identical on every rank. Avoids
      the straggler effect where one rank does 3 no_grad warmup passes
      and another does 0, blocking DDP all_reduce on the slowest rank
      (~4× wasted compute at ``max_recycles=3``).
    * ``uniform_recycle=False``: legacy per-rank Python random — each
      rank picks independently from its own seeded RNG. Trains on more
      diverse recycle counts per global batch but pays the straggler
      cost.
    """
    if cfg.uniform_recycle:
        return random.Random(cfg.seed + global_step).randint(0, cfg.max_recycles)
    return random.randint(0, cfg.max_recycles)


def training_step(
    model: DeepFold3,
    features: dict[str, torch.Tensor],
    *,
    cfg: TrainingConfig,
    model_cfg: ModelConfig,
    global_step: int,
    accumulator: AccumulationLoop,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, int], int, float, float]:
    """Run one forward pass + compute losses for all batch elements.

    Returns ``(total_loss, loss_dict, loss_counts, num_recycles, fwd_wall,
    loss_wall)``, where ``loss_counts[k]`` is how many batch elements emitted
    key ``k`` — see :func:`aggregate_sample_losses`. The chunked
    diffusion path back-props each chunk internally via
    ``scaled_backward``; those per-chunk backwards run under
    ``accumulator.chunk_context`` so DDP doesn't all_reduce per chunk
    — the final aux backward fires the single per-microstep all_reduce.
    """
    features = prepare_features(features, device)

    num_recycles = pick_num_recycles(cfg, global_step)

    chunked = is_chunked_diffusion_active(cfg)
    backward_fn: Callable[[torch.Tensor], None] | None
    loss_cfg_arg: TrainingConfig | None
    if chunked:

        def backward_fn(loss: torch.Tensor) -> None:
            scaled_backward(scaler, loss, cfg.accumulate_grad_batches)

        loss_cfg_arg = cfg
        chunk_size_arg = cfg.diffusion_chunk_size
    else:
        backward_fn = None
        loss_cfg_arg = None
        chunk_size_arg = 0

    # Tier-4c phase timing. Gated on the slow-step diagnostic. The sync after
    # the forward (and after the loss) is what makes the fwd vs loss-CPU split
    # clean: without it ``fwd`` is only the async launch and the forward-GPU
    # drain leaks into the loss section (the chain-perm ``.cpu()`` blocks on
    # it). The syncs are near-free — the loss's chain-perm already drains the
    # forward, and nothing overlaps the loss-CPU section on the GPU anyway.
    _timing = cfg.slow_step_log_threshold_s > 0 and torch.cuda.is_available()
    t0 = time.monotonic() if _timing else 0.0

    with accumulator.chunk_context(chunked_diffusion=chunked):
        # ``model`` is the DDP-wrapped ``_DDPTrainingForward`` (see
        # ``wrap_ddp``), whose ``forward`` routes to
        # ``DeepFold3.forward_training``. Calling it via ``__call__`` is
        # what triggers DDP's ``_pre_forward`` → reducer priming; calling
        # ``forward_training`` directly bypasses DDP and silently drops
        # gradient sync.
        pred = model(
            features,
            num_recycles=num_recycles,
            diffusion_batch_size=cfg.diffusion_batch_size,
            mini_rollout_steps=cfg.mini_rollout_steps,
            mini_rollout_samples=cfg.mini_rollout_samples,
            seed=global_step,
            fused_ce=cfg.use_fused_ce_loss,
            diffusion_chunk_size=chunk_size_arg,
            backward_fn=backward_fn,
            loss_cfg=loss_cfg_arg,
            compute_confidence=cfg.alpha_confidence > 0,
            msa_precropped=cfg.msa_precrop,
            chain_perm_gate_k=(cfg.chain_perm_gate_k if cfg.chain_perm_stability_gate else 1),
            permute_gt_before_noise=cfg.permute_gt_before_noise,
            compute_diffusion=not cfg.train_confidence_only,
        )
    if _timing:
        torch.cuda.synchronize(device)
        t1 = time.monotonic()

    # ``training_loss`` assumes unbatched ``[N, ...]`` features — slice
    # per batch element and average (even for B=1, since the batch dim
    # is kept). ``fp32_loss`` runs the combined loss with autocast
    # disabled so softmax / CE always see fp32 logits.
    loss_fn = fp32_guard(model_cfg.precision.fp32_loss)(training_loss)
    batch_size = cfg.batch_size_per_rank
    losses_list: list[torch.Tensor] = []
    per_sample_losses: list[dict[str, torch.Tensor]] = []
    for b in range(batch_size):
        pred_b = slice_batch_dict(pred, b, batch_size)
        features_b = slice_batch_dict(features, b, batch_size)
        loss_b, ld_b = loss_fn(pred_b, features_b, cfg)
        losses_list.append(loss_b)
        per_sample_losses.append(ld_b)
    # ``total_loss`` keeps the plain batch mean — a sample that contributes 0
    # to some term is still a real sample of the objective. Only the REPORTED
    # per-term values get a per-key denominator.
    total_loss = torch.stack(losses_list).mean()
    loss_dict, loss_dict_counts = aggregate_sample_losses(per_sample_losses)
    # One draw per forward (not per batch element), so each counts once.
    for k, v in pred.get("__cond_drop__", {}).items():
        loss_dict[f"cond_drop/{k}"] = v
        loss_dict_counts[f"cond_drop/{k}"] = 1

    if _timing:
        torch.cuda.synchronize(device)
        fwd_wall = t1 - t0  # forward + chunked-diffusion backward (GPU)
        loss_wall = time.monotonic() - t1  # loss section: chain-perm CPU + aux-loss GPU
    else:
        fwd_wall = loss_wall = 0.0
    return total_loss, loss_dict, loss_dict_counts, num_recycles, fwd_wall, loss_wall


__all__ = [
    "aggregate_sample_losses",
    "pick_num_recycles",
    "prepare_features",
    "scaled_backward",
    "training_step",
]
