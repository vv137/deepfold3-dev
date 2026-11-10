"""Optimizer, scheduler, DDP wrapping, and the NaN-hardened step contract.

:class:`WrappedOptimizer` owns the full per-microstep NaN handling and
the sync-boundary collective. It is the one place where the
``all_gather_into_tensor`` across ranks runs; dropping the call on any
rank desynchronizes collective participation and hangs NCCL (see
``memory/project_try_except_continue_deadlock.md``), so the call MUST
run unconditionally on every rank when ``is_sync_step`` is true.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from deepfold3.model.config import TrainingConfig
from deepfold3.training.collective_guard import collective_guard
from deepfold3.training.distributed import DistInfo
from deepfold3.training.rank_zero import rank_zero_critical, rank_zero_info, rank_zero_warn
from deepfold3.training.schedule_free import AdamWScheduleFree
from deepfold3.training.state import TrainState

logger = logging.getLogger(__name__)

# torch 2.13 deprecated ``all_gather_into_tensor`` in favour of the identically
# signed ``all_gather_single`` (emits a FutureWarning on every call otherwise).
# Prefer the new name when present, fall back for older torch — same semantics.
_all_gather_into_tensor = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor


def _clip_grad_norm_fp64(parameters, max_norm: float) -> torch.Tensor:
    """Deterministic-precision analogue of ``torch.nn.utils.clip_grad_norm_``.

    Accumulates per-param L2² in fp64, then takes sqrt and clips in place.
    Headroom beyond fp32 prevents the sum-of-squares from overflowing on
    one rank while staying finite on another (torch fp32 reduction order
    is device-dependent), which otherwise triggers spurious whole-window
    skips at the collective consensus check in :meth:`step_or_skip`.

    Returns an fp32 scalar matching ``clip_grad_norm_``'s contract
    (including non-finite pass-through when any grad is NaN/Inf).
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.zeros((), dtype=torch.float32, device="cpu")
    # vector_norm(g, 2) in fp32, square in fp64 — fp64 sum has ~1e308
    # headroom so realistic grads never overflow the reduction.
    sq = torch.stack([torch.linalg.vector_norm(g, 2).to(torch.float64).pow_(2) for g in grads])
    total_fp64 = sq.sum().sqrt()
    total = total_fp64.to(torch.float32)
    if torch.isfinite(total) and max_norm != float("inf"):
        coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
        # Cast coef once per param-dtype group to avoid per-grad .to() churn.
        by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
        for g in grads:
            by_dtype.setdefault(g.dtype, []).append(g)
        for dtype, gs in by_dtype.items():
            torch._foreach_mul_(gs, coef.to(dtype))
    return total


def _all_reduce_grads_(parameters, world_size: int) -> torch.Tensor:
    """Average per-rank gradients across ranks; return the post-reduce global norm.

    Used only in per-rank grad-clip mode: the whole backward ran under
    ``no_sync`` so DDP's reducer never fired, and each rank has already
    clipped its LOCAL gradient. The clipped grads are averaged across ranks
    with a single coalesced ``AVG`` all-reduce (the mean folds into the
    collective — no separate divide pass), so the optimizer steps on the mean
    of the per-rank clipped directions. The global L2 norm of that averaged
    gradient is measured from the contiguous ``flat`` buffer in one fused fp64
    reduction and returned, so the caller needs no extra param sweep to log
    it.

    The flat buffer spans EVERY trainable param in a fixed order (missing
    grad → zeros), NOT just params that received a grad this step. That set is
    data-dependent: a distillation sample skips the confidence/PAE loss so its
    confidence params get no gradient, and because the backward ran under
    ``no_sync`` DDP's reducer never zero-filled them — so they stay ``None``.
    Filtering ``p.grad is not None`` would then size the buffer differently on
    a rank that drew a distillation sample vs a PDB sample, and the collective
    would DEADLOCK (mismatched counts). ``requires_grad`` is static and
    identical on every rank, so a full-layout buffer is always the same size.
    Grads are fp32 (fp32 master weights), so a single flat-buffer reduction is
    exact and needs no dtype grouping. In-place: ``.grad`` holds the averaged
    value on return (created for params that had none locally, matching DDP's
    zero-fill-then-reduce semantics).
    """
    params = [p for p in parameters if p.requires_grad]
    if not params:
        return torch.zeros((), dtype=torch.float32, device="cpu")
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.AVG)  # SUM ÷ world_size, folded into the collective
    global_norm = torch.linalg.vector_norm(flat, 2, dtype=torch.float64).to(torch.float32)
    for p, synced in zip(params, torch._utils._unflatten_dense_tensors(flat, grads), strict=True):
        if p.grad is None:
            p.grad = synced.clone()  # unused locally, but nonzero on another rank → must step
        else:
            p.grad.copy_(synced)
    return global_norm


def is_chunked_diffusion_active(cfg: TrainingConfig) -> bool:
    """Return True when diffusion is run in fixed-size chunks inside each microstep.

    Kept as a pure function of the config so ``wrap_ddp`` can decide
    whether to force-disable ``static_graph`` without instantiating the
    trainer.
    """
    return cfg.diffusion_chunk_size > 0 and cfg.diffusion_batch_size % cfg.diffusion_chunk_size == 0


def freeze_for_confidence(model: nn.Module) -> None:
    """Freeze all parameters except the confidence head (stage-3 training)."""
    for name, param in model.named_parameters():
        if "confidence" not in name:
            param.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank_zero_info(f"Confidence-only training: {trainable / 1e6:.1f}M trainable parameters")


def freeze_modules(model: nn.Module, names: list[str]) -> None:
    """Freeze the named top-level submodules (e.g. ``evoformer``) of ``model``.

    Raises:
        ValueError: if a name is not a direct child module of ``model``.
    """
    children = dict(model.named_children())
    unknown = [n for n in names if n not in children]
    if unknown:
        raise ValueError(f"freeze_modules: no top-level module {unknown}; have {sorted(children)}")
    for n in names:
        children[n].requires_grad_(False)
    frozen = sum(p.numel() for n in names for p in children[n].parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank_zero_info(f"Frozen {names}: {frozen / 1e6:.1f}M params frozen, {trainable / 1e6:.1f}M trainable")


# Parameter-name substrings that mark a weight as "gate" or "embedding" — both
# excluded from weight decay alongside LN/bias (1D) params. Substring match on
# the qualified parameter name (so ``ModuleList`` indices like
# ``attn_cond_scale.0.weight`` and ``cond_scale.weight`` both match).
#
# Gates: sigmoid-modulated multiplicative paths. Decaying their weights pulls
# the gate toward 0.5 (always-on), erasing the conditional behaviour and
# hurting expressivity. Covers ``to_g`` (attention output gate), ``out_gate``
# / ``.gate.`` (TriangleMul gates), and the AdaLN-zero ``cond_scale``,
# ``cond_bias``, ``zero_cond`` projections (incl. atom-transformer's
# ``attn_cond_scale``, ``k_cond_scale``, ``ffn_cond_scale``, ``*_cond_bias``
# ``ModuleList`` variants).
#
# Embeddings: first projection from raw features to channels (no LN before
# them — they ARE the input lifting). Decaying these shrinks the input
# representation toward zero. Covers per-atom and template ``embed_*``
# linears, ``{single,msa}_activations``, ``{left,right}_single``,
# ``position_activations``, ``bond_embedding``, ``extra_msa_target_feat``,
# ``prev_embedding``, ``prev_single_embedding``.
_NO_DECAY_NAME_SUBSTRINGS = (
    # gates / AdaLN-zero scale·shift·gate (must end with ``.weight`` to skip
    # biases, which are already 1D-no-decay)
    ".to_g.",
    ".out_gate.",
    ".gate.",
    "cond_scale.",
    "cond_bias.",
    "zero_cond.",
    # embeddings — input feature → channel projections
    ".embed_",
    "single_activations.",
    "msa_activations.",
    "extra_msa_target_feat.",
    "left_single.",
    "right_single.",
    "position_activations.",
    "bond_embedding.",
    "prev_embedding.",
    "prev_single_embedding.",
)


def _is_no_decay_by_name(name: str) -> bool:
    if not name.endswith(".weight"):
        return False
    return any(s in name for s in _NO_DECAY_NAME_SUBSTRINGS)


def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    """Build AdamW with decoupled weight decay.

    Protenix / nanoGPT convention extended for AF3 architecture:
        * 1D params (LayerNorm gamma/beta, biases) → no-decay (collapses
          scale/shift terms otherwise).
        * 2D weights with ``min(shape) == 1`` (``LinearNoBias(1, C)`` etc.,
          per-channel scale factors) → no-decay.
        * Gate weights (``to_g``, ``out_gate``, ``.gate.``, AdaLN-zero
          ``cond_scale`` / ``cond_bias`` / ``zero_cond``) → no-decay.
          Decay would pull sigmoids toward 0.5 (always-on) and kill the
          conditional / learned-mask behaviour.
        * Embedding weights (input-feature → channel lift, e.g.
          ``single_activations``, ``msa_activations``, ``embed_*`` in
          per-atom and template stacks, ``prev_embedding``) → no-decay.
          These have no preceding LN, so decay shrinks the input
          representation directly.
        * All remaining 2D+ weights (matmul weights inside attention
          / FFN / triangle multiplication / OPM out_w) → decay.

    See :data:`_NO_DECAY_NAME_SUBSTRINGS` for the full pattern list.
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    # Per-multiplier strong-decay buckets: multiplier (float) → list of params
    strong_buckets: dict[float, list[nn.Parameter]] = {}
    strong_decay_map = dict(cfg.strong_decay or {})  # substring → multiplier
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or min(param.shape) == 1:
            no_decay_params.append(param)
            continue
        if _is_no_decay_by_name(name):
            no_decay_params.append(param)
            continue
        # Strong-decay first-match wins
        matched = None
        for sub, mult in strong_decay_map.items():
            if sub in name:
                matched = mult
                break
        if matched is not None:
            strong_buckets.setdefault(matched, []).append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    for mult, ps in strong_buckets.items():
        param_groups.append({"params": ps, "weight_decay": cfg.weight_decay * mult})

    n_decay = sum(p.numel() for p in decay_params) / 1e6
    n_nodecay = sum(p.numel() for p in no_decay_params) / 1e6
    n_strong = {m: sum(p.numel() for p in ps) / 1e6 for m, ps in strong_buckets.items()}

    if cfg.optimizer_type == "schedule_free":
        # SF manages its own warmup-then-flat schedule; ``finetune_scheduler``
        # is bypassed via an identity ``LambdaLR`` in :func:`build_scheduler`.
        optimizer = AdamWScheduleFree(
            param_groups,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            warmup_steps=cfg.warmup_steps,
            decoupling_parameter=cfg.decoupling_parameter,
        )
        # SF.step() raises if not in train mode. Eval/checkpoint save flip it
        # via ``optimizer_eval_mode`` and restore on exit.
        optimizer.train()
        # SF + EMA(x): the trainer wraps ``ema.update(model)`` in
        # ``optimizer_eval_mode(optimizer)`` so the shadow tracks ``x`` (the
        # SF inference target), not ``y`` (the gradient point). Disable
        # explicitly via ``ema_decay: 0`` (configs/schedule_free.yaml does
        # this) when relying on SF's own implicit averaging.
        #
        # Through the River (arxiv 2507.09846 §6) finds that EMA on top of
        # vanilla SF gives little benefit (because SF's x already provides
        # implicit weight averaging with a wide window). With refined SF
        # (``decoupling_parameter`` set), x's window is tighter, so external
        # EMA can add complementary smoothing — a useful ablation knob.
        if cfg.ema_decay > 0:
            rank_zero_info(
                f"AdamWScheduleFree + EMA(x): ema_decay={cfg.ema_decay}. "
                "EMACallback wraps ema.update() in optimizer_eval_mode() so "
                "the shadow tracks x, not y. Per-step cost: 2 extra "
                "param-sized lerps."
            )
        strong_summary = "".join(f", {m:g}× wd: {p:.2f}M" for m, p in sorted(n_strong.items()))
        rank_zero_info(
            f"AdamWScheduleFree: warmup_steps={cfg.warmup_steps}, "
            f"weight_decay={cfg.weight_decay}, "
            f"decoupling_parameter={cfg.decoupling_parameter} "
            f"({n_decay:.1f}M decayed, {n_nodecay:.2f}M no-decay{strong_summary})"
        )
        return optimizer

    if cfg.optimizer_type != "adamw":
        raise ValueError(f"optimizer_type must be 'adamw' or 'schedule_free', got {cfg.optimizer_type!r}")

    # Fused AdamW on CUDA: single kernel launch per param group,
    # ~1.3-1.5x faster than the foreach implementation at our scale.
    use_fused_adam = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        fused=use_fused_adam,
    )
    strong_summary = "".join(f", {m:g}× wd: {p:.2f}M" for m, p in sorted(n_strong.items()))
    rank_zero_info(
        f"AdamW: fused={use_fused_adam}, weight_decay={cfg.weight_decay} "
        f"({n_decay:.1f}M decayed, {n_nodecay:.2f}M no-decay{strong_summary})"
    )
    return optimizer


class _DDPTrainingForward(nn.Module):
    """Expose ``DeepFold3.forward_training()`` as the wrapped module's ``forward``.

    DDP's gradient reducer is primed inside ``ddp_model.__call__`` via
    ``_pre_forward``/``prepare_for_backward`` — if that path never runs,
    no bucket AllReduce is scheduled for the upcoming backward and each
    rank trains on its own local gradients (silent DDP-no-sync bug).

    DF3's training entry point is a bespoke method ``forward_training``
    distinct from the inference ``forward``, so we put a tiny shim whose
    ``forward`` routes to it. Parameters auto-register via ``self.df3 =
    df3``, so ``ddp_model.parameters()`` returns the underlying DF3
    params unchanged — checkpoint save/load and EMA still see the raw
    DeepFold3 via ``self.model`` on the trainer.
    """

    def __init__(self, df3: nn.Module) -> None:
        super().__init__()
        self.df3 = df3

    def forward(
        self,
        features: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        return self.df3.forward_training(features, **kwargs)


def wrap_ddp(model: nn.Module, dist: DistInfo, cfg: TrainingConfig) -> tuple[nn.Module, bool]:
    """Wrap the model in :class:`DistributedDataParallel` (if distributed).

    Returns ``(wrapped_model, use_static_graph)``. The static-graph
    decision matrix:

    * ``static_graph=True`` is safe in the unchunked path — recycles
      1..N-1 run inside ``torch.no_grad()`` (model.py:396-409), so only
      the last recycle is in the autograd graph and the graph topology
      is identical iteration-to-iteration regardless of
      ``num_recycles``. Gradient accumulation uses ``no_sync()`` on
      non-final microsteps, so only the final backward per optimizer
      step triggers DDP sync. Confidence-loss resolution gating is a
      tensor multiply (loss.py) rather than a Python branch, so the set
      of grad-receiving parameters is identical every step.
    * ``static_graph=True`` is INCOMPATIBLE with ``model.no_sync()``.
      PyTorch raises if the two are combined. ``no_sync`` is used in
      two places: (a) chunked-diffusion to defer DDP all_reduce until
      after the outer aux backward, and (b) gradient-accumulation
      microsteps to defer sync until the final microstep. We
      auto-disable ``static_graph`` when either is active.
    """
    chunked_diffusion = is_chunked_diffusion_active(cfg)
    # Per-rank grad clip defers ALL gradient all-reduce to a manual reduce
    # after clipping, so even the sync-step backward runs under no_sync.
    needs_no_sync = chunked_diffusion or cfg.accumulate_grad_batches > 1 or cfg.per_rank_grad_clip
    use_static_graph = cfg.static_graph and not needs_no_sync

    if not dist.is_distributed:
        # Still wrap so the training call path is uniform, even though
        # there's no gradient sync to schedule.
        return _DDPTrainingForward(model), use_static_graph

    if cfg.static_graph and not use_static_graph:
        reasons = []
        if chunked_diffusion:
            reasons.append(f"diffusion_chunk_size={cfg.diffusion_chunk_size}")
        if cfg.accumulate_grad_batches > 1:
            reasons.append(f"accumulate_grad_batches={cfg.accumulate_grad_batches}")
        if cfg.per_rank_grad_clip:
            reasons.append("per_rank_grad_clip=True")
        rank_zero_info(
            "static_graph auto-disabled (%s): model.no_sync() is required, which conflicts with static_graph.",
            ", ".join(reasons),
        )

    if cfg.per_rank_grad_clip:
        # Invariant surfaced at startup: per-rank clip is correct ONLY while
        # DDP's automatic all-reduce is suppressed on every backward (which
        # ``AccumulationLoop.step_context`` guarantees via always-on
        # no_sync). If a future change let a backward run without no_sync,
        # ``.grad`` would already be all-reduced and the per-rank clip would
        # silently degrade to a global clip.
        rank_zero_info(
            "per-rank grad clip ON: DDP auto all-reduce suppressed (no_sync on every "
            "backward); each rank clips its LOCAL grad to grad_clip_norm=%.3g, then a "
            "single manual all-reduce averages the clipped grads. static_graph and the "
            "reduce_dtype comm hook are unused in this mode; the reduce runs after the "
            "backward (no overlap).",
            cfg.grad_clip_norm,
        )

    ddp_model = DDP(
        _DDPTrainingForward(model),
        device_ids=[dist.local_rank],
        find_unused_parameters=cfg.find_unused_parameters,
        static_graph=use_static_graph,
        bucket_cap_mb=cfg.bucket_cap_mb,
        gradient_as_bucket_view=cfg.gradient_as_bucket_view,
    )
    return ddp_model, use_static_graph


def register_reduce_hook(ddp_model: nn.Module, reduce_dtype: str) -> str:
    """Install the DDP allreduce comm hook matching ``reduce_dtype``.

    Returns a human-readable hook name for logging. No-op (returns
    ``"default (fp32 allreduce)"``) when ``reduce_dtype == "fp32"`` —
    DDP's stock allreduce already preserves fp32. ``"bf16"`` registers
    PyTorch's ``bf16_compress_hook`` which halves allreduce bandwidth
    at the cost of one bf16 cast per bucket.
    """
    if reduce_dtype == "bf16":
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_hooks

        ddp_model.register_comm_hook(state=None, hook=ddp_hooks.bf16_compress_hook)
        return "bf16_compress"
    return "default (fp32 allreduce)"


@dataclass
class StepResult:
    """Outcome of one optimizer-step attempt."""

    advanced: bool
    was_nan: bool
    reason: Literal["loss", "grad"] | None
    grad_norm: torch.Tensor | None
    nan_ranks: list[int]


class WrappedOptimizer:
    """Optimizer wrapper that enforces the NaN/scaler/clip/step contract.

    The trainer calls :meth:`add_microstep_loss` after each microstep
    backward (to OR the local NaN flag into the on-device accumulator)
    and :meth:`step_or_skip` at the sync boundary (to run the single
    cross-rank collective, unscale + clip, and either step or skip).

    Precision contract (DeepSpeed-inspired; enforced at construction +
    lazily after the first step):

    * **Master weights are fp32.** Asserted at ``__init__`` on every
      trainable parameter. AdamW's moment state is fp32-or-nothing — a
      bf16 param would force bf16 moments, whose rounding error swallows
      weight updates on most steps.
    * **Optimizer state is fp32.** Adam's ``exp_avg`` / ``exp_avg_sq``
      are asserted once on the first successful step (lazy — the tensors
      don't exist until ``.step()`` runs). A checkpoint load that
      downgrades either dtype will fail here with a clear message.
    * **Gradients arrive in the compute dtype** (bf16/fp16 under
      autocast, fp32 otherwise). Unscale (if ``fp16``) → clip via
      :func:`_clip_grad_norm_fp64` → cross-rank NaN consensus all run
      before ``optimizer.step``. Grad-accumulation dtype is
      ``precision.grad_acc_dtype``; with ``accumulate_grad_batches > 1``
      only ``fp32`` is safe (validated upstream in the trainer).
    * **Reductions (all-reduce + grad-norm)** stay in fp32 regardless
      of ``precision.reduce_dtype``. The comm-hook downcast (when
      ``reduce_dtype=='bf16'``) is PyTorch-internal and invisible to
      this class.

    Invariants preserved verbatim from the pre-refactor trainer:

    * ``_nan_accum`` is materialised lazily on the first call, shape
      ``[1]`` so ``all_gather_into_tensor`` can concatenate across
      ranks into ``[world_size]``.
    * The NaN collective runs EXACTLY ONCE per opt step on every rank.
    * Two NaN checks: loss-side (from the accumulator) + grad-side
      (after ``clip_grad_norm_`` — catches finite-loss / NaN-grad
      paths like log/sqrt/div).
    * ``cfg.max_consecutive_nan_skips`` raises after N skips in a row.
    * The scheduler is stepped ONLY on a successful opt step (skips
      on NaN leave the LR unchanged).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler: GradScaler,
        model: nn.Module,
        dist_info: DistInfo,
        cfg: TrainingConfig,
    ) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.model = model
        self.dist = dist_info
        self.cfg = cfg
        self._nan_accum: torch.Tensor | None = None
        self._nan_gather_buf: torch.Tensor | None = None
        self._grad_norm_gather_buf: torch.Tensor | None = None
        # Record ids of every sample that contributed to the current
        # accumulation window on THIS rank. Reset in ``step_or_skip`` so
        # each window starts empty; logged on NaN/Inf skip to identify
        # offending examples.
        self._window_record_ids: list[str] = []
        self._state_dtype_checked = False

        self._assert_master_weight_dtype()

    def _assert_master_weight_dtype(self) -> None:
        """Every trainable param must be fp32 (master-weight invariant).

        A bf16 parameter would force AdamW to store moments in bf16,
        whose rounding error swallows the (lr * moment) update on most
        steps. Autocast keeps the *forward* in bf16; the parameter
        itself must stay fp32.
        """
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.dtype != torch.float32:
                    raise ValueError(
                        f"WrappedOptimizer requires fp32 master weights, got dtype={param.dtype} "
                        f"for a parameter with shape {tuple(param.shape)}. Autocast controls the "
                        "forward-pass dtype; parameters themselves must stay fp32."
                    )

    def _assert_state_dtype(self) -> None:
        """Assert Adam moment state is fp32. Runs once, after the first step."""
        if self._state_dtype_checked:
            return
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                state = self.optimizer.state.get(param)
                if not state:
                    continue
                for key in ("exp_avg", "exp_avg_sq"):
                    tensor = state.get(key)
                    if tensor is not None and tensor.dtype != torch.float32:
                        raise ValueError(
                            f"AdamW state['{key}'] must be fp32, got {tensor.dtype}. This usually "
                            "means a checkpoint was loaded from a bf16-optimizer run; upcast the "
                            "moments in the checkpoint before resuming."
                        )
        self._state_dtype_checked = True

    # ----- per-microstep -----

    def add_microstep_loss(self, loss: torch.Tensor, record_ids: list[str | None] | None = None) -> None:
        """OR the local NaN/Inf flag into the on-device accumulator.

        No collective, no CPU sync — this is called on every microstep.
        ``record_ids`` (one per sample in the microstep batch) are
        appended to the window's diagnostic list.
        """
        if self._nan_accum is None:
            self._nan_accum = torch.zeros(1, device=loss.device)
        self._nan_accum.add_((torch.isnan(loss) | torch.isinf(loss)).float().view(1))
        if record_ids:
            for rid in record_ids:
                if rid is not None:
                    self._window_record_ids.append(rid)

    # ----- sync boundary -----

    def _unscale_clip_and_measure(self, state: TrainState) -> torch.Tensor:
        """Unscale → clip → measure, in a single ordered block.

        ORDER INVARIANT — do not reorder:

        1. ``scaler.unscale_`` divides grads by the loss scale AND sets the
           scaler's internal per-device ``found_inf`` flag.
        2. ``clip_grad_norm_`` sees the unscaled grads and returns a
           non-finite total whenever any grad is non-finite (bf16 overflow,
           NaN/Inf from log/sqrt/div, etc).
        3. The returned ``grad_norm`` is the single source of truth for
           the collective skip decision in :meth:`step_or_skip`.

        Encapsulating the sequence here prevents a future refactor from
        reordering the statements and reintroducing the drift bug: if
        ``scaler.step`` runs before our ``isfinite(grad_norm)`` check
        clears it, it may silently skip on some ranks (per-device
        ``found_inf``) while advancing on others → param drift without
        any observable log line.

        Per-rank clip (``cfg.per_rank_grad_clip``): the whole backward ran
        under ``no_sync`` (DDP never all-reduced), so the clip above acts on
        this rank's LOCAL gradient. We then average the already-clipped
        per-rank grads with a single manual all-reduce and measure the
        post-reduce global norm. The returned ``grad_norm`` stays the LOCAL
        pre-clip norm, so the downstream finite/outlier consensus catches a
        per-rank blow-up before the reduce would spread it across ranks.
        """
        self.scaler.unscale_(self.optimizer)

        clip = self.cfg.grad_clip_norm if self.cfg.grad_clip_norm > 0 else float("inf")
        if self.cfg.grad_norm_fp64:
            grad_norm = _clip_grad_norm_fp64(self.model.parameters(), clip).detach()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip).detach()
        state.last_grad_norm = grad_norm

        if self.cfg.per_rank_grad_clip and self.dist.is_distributed:
            # Manual all-reduce (DDP's reducer is suppressed under no_sync). The
            # returned norm is measured from the averaged flat buffer, so no
            # extra param sweep is needed. Outside per-rank mode the field
            # stays at its ``None`` default (never read by the trainer).
            state.last_grad_norm_global = _all_reduce_grads_(self.model.parameters(), self.dist.world_size).detach()
        return grad_norm

    def step_or_skip(self, state: TrainState) -> StepResult:
        """Run the collective NaN check, then either step or skip.

        MUST be called by EVERY rank when ``is_sync_step`` is true.
        Dropping the call on any rank desyncs the
        ``all_gather_into_tensor`` below and hangs NCCL. The outer
        ``collective_guard`` attaches the region name to any watchdog
        failure so hangs point here instead of at an anonymous
        all_gather deep in the call chain.
        """
        assert self._nan_accum is not None, "add_microstep_loss must run at least once before step_or_skip"

        with collective_guard("step_or_skip", self.dist):
            # Sync boundary: single collective for NaN check across ranks +
            # all microsteps in this accumulation window. ``all_gather`` so
            # the warning can name the offending rank(s).
            if self.dist.is_distributed:
                if self._nan_gather_buf is None:
                    self._nan_gather_buf = torch.zeros(self.dist.world_size, device=self._nan_accum.device)
                _all_gather_into_tensor(self._nan_gather_buf, self._nan_accum)
                nan_flags = self._nan_gather_buf
            else:
                nan_flags = self._nan_accum
            nan_detected = bool(nan_flags.any().item())  # one CPU sync per opt step
            self._nan_accum.zero_()

            attempted_step = state.effective_step + 1

            if nan_detected:
                # Any microstep on any rank produced NaN/Inf — discard the
                # whole accumulation window. NaN would have propagated via
                # DDP all_reduce on the sync-step backward.
                self.optimizer.zero_grad(set_to_none=True)
                bad_ranks = (nan_flags > 0).nonzero(as_tuple=False).flatten().tolist()
                rank_zero_warn(
                    f"Step {attempted_step}: NaN/Inf loss in accumulation window (ranks={bad_ranks}), skipping"
                )
                # Per-rank: only the rank whose data triggered the NaN knows
                # the offending record ids. Log locally from there; no string
                # broadcast needed (NCCL doesn't handle variadic strings well).
                if self.dist.rank in bad_ranks and self._window_record_ids:
                    logger.warning(
                        f"Step {attempted_step}: rank {self.dist.rank} NaN/Inf loss window "
                        f"record_ids={self._window_record_ids}"
                    )
                self._window_record_ids.clear()
                self._account_skip(state, attempted_step, reason="loss")
                return StepResult(advanced=False, was_nan=True, reason="loss", grad_norm=None, nan_ranks=bad_ranks)

            grad_norm = self._unscale_clip_and_measure(state)

            # clip_grad_norm_ returns a NaN/Inf total when ANY param grad
            # is non-finite, so this single check covers the whole model.
            # Without it, a finite loss with non-finite grads (log/sqrt/div
            # paths) silently corrupts weights via optimizer.step.
            #
            # Collective consensus: even with DDP-synced grads, bf16 rounding
            # in ``clip_grad_norm_``'s sum-of-squared-norms can overflow on
            # one rank but stay finite on another. A per-rank ``isfinite``
            # check then returns different ``advanced`` values → ranks
            # disagree on ``on_opt_step_boundary`` → the log/ckpt/eval gates
            # in ``trainer._run_inner`` diverge → NCCL hang. All-gather the
            # scalar so every rank sees the same vector, then reduce with
            # ``any()`` for identical advance/skip decisions.
            if self.dist.is_distributed:
                if self._grad_norm_gather_buf is None:
                    self._grad_norm_gather_buf = torch.zeros(self.dist.world_size, device=grad_norm.device)
                _all_gather_into_tensor(self._grad_norm_gather_buf, grad_norm.view(1))
                grad_norms_per_rank = self._grad_norm_gather_buf
            else:
                grad_norms_per_rank = grad_norm.view(1)
            grad_finite_per_rank = torch.isfinite(grad_norms_per_rank)
            # Cross-rank PRE-clip summary, free from the gather above. Without
            # it the only logged grad norm is rank 0's local one, so a large
            # but finite spike on any other rank is unobservable: it is clipped
            # to ``grad_clip_norm`` and ``last_grad_norm_global`` is measured
            # after that clip. Computed from the finite entries only, so a NaN
            # on one rank does not wipe out the summary for the others (the
            # skip path below reports those separately).
            _finite_norms = grad_norms_per_rank[grad_finite_per_rank]
            if _finite_norms.numel() > 0:
                state.last_grad_norm_rank_mean = _finite_norms.mean().detach()
                state.last_grad_norm_rank_max = _finite_norms.max().detach()
            if not bool(grad_finite_per_rank.all().item()):
                return self._skip_for_bad_grad(
                    state,
                    attempted_step,
                    grad_norm,
                    grad_norms_per_rank,
                    (~grad_finite_per_rank).nonzero(as_tuple=False).flatten().tolist(),
                    warn_prefix="NaN/Inf in grads",
                    record_prefix="NaN/Inf grad window",
                )

            # Per-rank outlier guard: clipping a pre-clip norm ~1e8 to
            # ``grad_clip_norm`` yields a near-random direction (scale factor
            # ~1e-7) that contributes nothing meaningful and accumulates
            # magnitude drift across many steps. Drop the whole step instead.
            skip_threshold = self.cfg.grad_norm_skip_threshold
            if skip_threshold > 0:
                outlier_per_rank = grad_norms_per_rank > skip_threshold
                if bool(outlier_per_rank.any().item()):
                    bad_ranks = outlier_per_rank.nonzero(as_tuple=False).flatten().tolist()
                    return self._skip_for_bad_grad(
                        state,
                        attempted_step,
                        grad_norm,
                        grad_norms_per_rank,
                        bad_ranks,
                        warn_prefix=f"grad outlier exceeded skip_threshold={skip_threshold:.3e}",
                        record_prefix="grad outlier window",
                    )

            # Surface grad-norm spikes that stay below the NaN/Inf skip
            # trigger — gives per-rank record_ids for examples driving the
            # spike without waiting for a hard blow-up.
            log_threshold = self.cfg.grad_norm_log_threshold
            if log_threshold > 0 and self._window_record_ids:
                grad_norm_val = grad_norm.item()
                if grad_norm_val > log_threshold:
                    logger.warning(
                        f"Step {attempted_step}: rank {self.dist.rank} grad_norm={grad_norm_val:.3e} "
                        f"exceeded log_threshold={log_threshold:.3e} window record_ids={self._window_record_ids}"
                    )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            # Lazy — Adam moments only exist after the first successful
            # ``.step()``. Raises on bf16 moment state (see
            # ``_assert_state_dtype``).
            self._assert_state_dtype()

            self._window_record_ids.clear()
            state.effective_step += 1
            state.consecutive_nan_skips = 0
            return StepResult(advanced=True, was_nan=False, reason=None, grad_norm=grad_norm, nan_ranks=[])

    def _skip_for_bad_grad(
        self,
        state: TrainState,
        attempted_step: int,
        grad_norm: torch.Tensor,
        grad_norms_per_rank: torch.Tensor,
        bad_ranks: list[int],
        *,
        warn_prefix: str,
        record_prefix: str,
    ) -> StepResult:
        self.optimizer.zero_grad(set_to_none=True)
        per_rank = ", ".join(f"rank{r}={grad_norms_per_rank[r].item():.3e}" for r in bad_ranks)
        rank_zero_warn(f"Step {attempted_step}: {warn_prefix} (ranks={bad_ranks}, {per_rank}), skipping")
        if self.dist.rank in bad_ranks and self._window_record_ids:
            logger.warning(
                f"Step {attempted_step}: rank {self.dist.rank} {record_prefix} record_ids={self._window_record_ids}"
            )
        self._window_record_ids.clear()
        self._account_skip(state, attempted_step, reason="grad")
        return StepResult(advanced=False, was_nan=True, reason="grad", grad_norm=grad_norm, nan_ranks=bad_ranks)

    def _account_skip(self, state: TrainState, attempted_step: int, *, reason: Literal["loss", "grad"]) -> None:
        """Bump the consecutive-skip counter and raise if it exceeds the limit.

        Every rank raises the same exception when the limit is hit —
        ``RuntimeError`` propagates cleanly through torchrun so all
        ranks die together rather than deadlocking on the next
        collective.
        """
        state.consecutive_nan_skips += 1
        limit = self.cfg.max_consecutive_nan_skips
        if limit > 0 and state.consecutive_nan_skips >= limit:
            msg = (
                f"Aborting: {state.consecutive_nan_skips} consecutive non-finite "
                f"{reason} skips at step {attempted_step} (limit={limit})."
            )
            rank_zero_critical(msg)
            raise RuntimeError(msg)
