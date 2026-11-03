"""Model configuration dataclasses for DeepFold3."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

from deepfold3.data import const


def _backfill_dataclass(obj, default) -> None:
    """Recursively add fields present on a fresh ``default`` but absent on
    ``obj`` (config schema evolved after ``obj`` was pickled).

    Old checkpoints embed a pickled config from an earlier schema; when a new
    field is later added to the dataclass (e.g. ``ModelConfig.masked_msa``),
    unpickling restores the old ``__dict__`` without it and any
    ``cfg.new_field`` access raises ``AttributeError``. Calling this from
    ``__setstate__`` migrates every load path (inference, trainer resume,
    convert, analysis scripts) automatically.
    """
    if not dataclasses.is_dataclass(obj):
        return
    for f in dataclasses.fields(default):
        dval = getattr(default, f.name)
        if not hasattr(obj, f.name):
            setattr(obj, f.name, dval)
        elif dataclasses.is_dataclass(dval):
            _backfill_dataclass(getattr(obj, f.name), dval)


@dataclass
class AtomTransformerConfig:
    num_blocks: int = 3
    num_intermediate_factor: int = 2
    num_head: int = 4
    key_dim: int = 128
    value_dim: int = 128


@dataclass
class AtomCrossAttConfig:
    per_token_channels: int = 768  # diffusion transformer channel dim
    per_atom_channels: int = 128
    per_atom_pair_channels: int = 16
    atom_transformer: AtomTransformerConfig = field(default_factory=AtomTransformerConfig)
    queries_subset_size: int = 32  # Q: atoms per query subset (non-overlapping)
    keys_subset_size: int = 128  # K: atoms per key window (overlapping, centered on Q)
    num_element_classes: int = 128  # element one-hot vocabulary size
    num_atom_name_chars: int = 64  # atom name character vocabulary size
    atom_name_length: int = 4  # characters per atom name
    # ``ref_element`` indexing convention for the element one-hot. False (default):
    # one-hot the raw atomic number Z (H=1, C=6, ...) — index 0 is a dead channel,
    # self-consistent for from-scratch training. True: one-hot the zero-indexed
    # periodic-table position (Z-1: H=0, C=5), the AF3/Protenix convention —
    # required when loading AF3-converted ``embed_ref_element`` weights.
    ref_element_zero_index: bool = False


@dataclass
class MSAAttentionConfig:
    num_head: int = 8


@dataclass
class GridAttentionConfig:
    num_head: int = 4


@dataclass
class EvoformerConfig:
    num_layer: int = 4
    msa_channel: int = 64
    seq_channel: int = 384
    pair_channel: int = 128
    max_relative_idx: int = 32
    max_relative_chain: int = 2
    msa_attention: MSAAttentionConfig = field(default_factory=MSAAttentionConfig)
    outer_product_num_channel: int = 32
    pair_attention: GridAttentionConfig = field(default_factory=GridAttentionConfig)
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25
    # Residual Output Scaling (ROS) on every F sub-module of this stack:
    # ``x ← x + s · F(x)`` where ``s`` is determined per-block. Two
    # mutually exclusive policies:
    #   residual_output_scaling : per-layer ``s = 1/sqrt(ℓ)``  (Sun-et-al.-flavoured)
    #   uniform_residual_scale  : depth-uniform ``s = 1/sqrt(num_layer)``
    residual_output_scaling: bool = False
    uniform_residual_scale: bool = False
    # ── Pair-track architecture (r17: MHR-TriMul + triangle-attention removal;
    # see export/mhr-trimul/README.md) ────────────────────────────────────────
    # ``use_triangle_attention`` deletes the two GridSelfAttention branches from
    # every block of this stack. That is a real subtraction, not a
    # simplification: triangle attention is the only pair operator whose
    # selection over the intermediate node ``k`` is conditioned on the *target*
    # pair ``(i, j)`` (it is the query) and the only one whose mixing over ``k``
    # is softmax-normalized. Triangle multiplication weights ``k`` by functions
    # of ``(i, k)`` and ``(j, k)`` alone. MHR-TriMul does not restore this — its
    # router selects which relation *subspace* explains a pair, not which
    # intermediate node does.
    #
    # ``trimul_*`` configure MHR-TriMul (``TriangleMul`` docstring for the
    # per-field meaning). ``trimul_num_heads=0`` ≡ AF3.
    use_triangle_attention: bool = True
    trimul_num_heads: int = 0
    trimul_head_norm: Literal["channel", "head_rms", "head_ln", "post_channel"] = "channel"
    trimul_router: bool = False
    # Router nonlinearity. ``softmax`` gives g = H*softmax_h(R(x)), so mean_h g
    # = 1 and the router can only reallocate — heads compete for a fixed budget
    # and, with no temperature or regularizer, collapse to one head is the
    # unregularized fixed point (measured in r17: effective heads fall 6.2 ->
    # 2.4 from L0-11 to L36-47, and the collapsed modules route *less*).
    # ``sigmoid`` gives g = 2*sigmoid(R(x)): independent per head, bounded in
    # (0, 2), so complementary relation subspaces can co-activate and there is
    # no winner-take-all attractor to balance away. Its cost is that mean_h g
    # may drift from 1, so the branch magnitude drifts and out_gate absorbs it.
    # Both are exactly g == 1 at zero-init. Default softmax = the r17 behaviour.
    trimul_router_kind: Literal["softmax", "sigmoid"] = "softmax"


@dataclass
class PairFormerConfig:
    num_layer: int = 48
    pair_channel: int = 128
    seq_channel: int = 384
    pair_attention: GridAttentionConfig = field(default_factory=GridAttentionConfig)
    single_attention_num_head: int = 16
    single_key_dim: int = 384
    single_value_dim: int = 384
    pair_dropout: float = 0.25
    # Residual Output Scaling (ROS) on every F sub-module of this stack
    # (see :class:`EvoformerConfig` for the two policies).
    residual_output_scaling: bool = False
    uniform_residual_scale: bool = False
    # Per-head learnable PairBias gate inside SingleAttention (log-param,
    # τ_h = exp(a_h), init τ₀ = pair_bias_gate_init_tau). Calibrates
    # PairBias vs ``q·kᵀ`` in the attention softmax. Shape [num_heads] per
    # attention; auto-routed to the no-decay AdamW group via the 1-D rule.
    pair_bias_gate: bool = False
    pair_bias_gate_init_tau: float = 0.25
    # ``single_transition`` SwiGLU output projection (linear_out / linear2)
    # init: ``None`` keeps the AF3 zeros init; a float ``gain`` replaces it
    # with a scaled row-orthogonal init ``W = gain · Q``. Breaks the
    # zero-wake-up bias that drives the SingleAttention/SingleTransition
    # output to align with the residual stream. ``0.01`` is small enough
    # that initial residual updates are tiny (output stddev ≈ gain).
    single_transition_linear_out_init_gain: float | None = None
    # Branch-specific scaling on the ``single_transition`` / ``pair_transition``
    # residual updates. Composes multiplicatively with the block-wide ROS:
    #
    #   s_branch = s_ros · gain · (1/sqrt(layer_idx+1) if per_layer_extra else 1)
    #
    # ``gain`` ``None`` ≡ 1.0. ``per_layer_extra`` adds a fresh
    # ``1/sqrt(ℓ+1)`` factor — under per-layer ROS this composes with
    # ``s_ros = 1/sqrt(ℓ+1)`` to give an effective ``1/(ℓ+1)`` schedule.
    # Used for the trunk transition aligned-gain stability ablation
    # (``docs/single-track-stability-experiments.md`` §11).
    single_transition_residual_gain: float | None = None
    single_transition_residual_per_layer_extra: bool = False
    pair_transition_residual_gain: float | None = None
    pair_transition_residual_per_layer_extra: bool = False
    # Function ``s(·)`` applied to every PairFormer residual branch output
    # before the per-block ``γ`` multiplier — extensibility hook for the
    # branch-output magnitude control (see
    # :class:`deepfold3.model.branch_scaler.BranchScalerKind`).
    #
    # * ``"identity"`` — raw branch (AF3 baseline)
    # * ``"rmsnorm"``  — non-learnable RMSNorm; each branch contributes
    #   exactly ``γ_k · unit-RMS`` regardless of internal scale drift
    #
    # Wraps the 7 branches: tri_mul_in, tri_mul_out, pair_attn_row,
    # pair_attn_col, pair_transition, single_attn, single_transition.
    # ``single_to_pair`` is **not** a residual branch (its output is consumed
    # as ``pair_logits`` for SingleAttention's PairBias) and is left alone.
    #
    # Trunk PairFormer scope only — Evoformer MSA stack, template
    # PairFormer, and confidence-head PairFormer are unaffected (they each
    # read their own ``PairFormerConfig`` instance for this flag).
    #
    # When non-identity, validates in ``__post_init__`` that the per-branch
    # residual gain knobs are at default — magnitude-fixing scalers make
    # them redundant (branch is already unit-RMS so an extra constant gain
    # just rescales the block-wide ``γ``).
    branch_scaler: Literal["identity", "rmsnorm"] = "identity"
    # ── Pair-track architecture (r17: MHR-TriMul + triangle-attention removal;
    # see export/mhr-trimul/README.md) ────────────────────────────────────────
    # ``use_triangle_attention`` deletes the two GridSelfAttention branches from
    # every block of this stack. That is a real subtraction, not a
    # simplification: triangle attention is the only pair operator whose
    # selection over the intermediate node ``k`` is conditioned on the *target*
    # pair ``(i, j)`` (it is the query) and the only one whose mixing over ``k``
    # is softmax-normalized. Triangle multiplication weights ``k`` by functions
    # of ``(i, k)`` and ``(j, k)`` alone. MHR-TriMul does not restore this — its
    # router selects which relation *subspace* explains a pair, not which
    # intermediate node does.
    #
    # ``trimul_*`` configure MHR-TriMul (``TriangleMul`` docstring for the
    # per-field meaning). ``trimul_num_heads=0`` ≡ AF3.
    use_triangle_attention: bool = True
    trimul_num_heads: int = 0
    trimul_head_norm: Literal["channel", "head_rms", "head_ln", "post_channel"] = "channel"
    trimul_router: bool = False
    # Router nonlinearity. ``softmax`` gives g = H*softmax_h(R(x)), so mean_h g
    # = 1 and the router can only reallocate — heads compete for a fixed budget
    # and, with no temperature or regularizer, collapse to one head is the
    # unregularized fixed point (measured in r17: effective heads fall 6.2 ->
    # 2.4 from L0-11 to L36-47, and the collapsed modules route *less*).
    # ``sigmoid`` gives g = 2*sigmoid(R(x)): independent per head, bounded in
    # (0, 2), so complementary relation subspaces can co-activate and there is
    # no winner-take-all attractor to balance away. Its cost is that mean_h g
    # may drift from 1, so the branch magnitude drifts and out_gate absorbs it.
    # Both are exactly g == 1 at zero-init. Default softmax = the r17 behaviour.
    trimul_router_kind: Literal["softmax", "sigmoid"] = "softmax"
    # ── Per-token track ───────────────────────────────────────────────────────
    # ``use_single_track`` deletes SingleAttention (with its ``single_to_pair``
    # PairBias projection) and the single transition from every block of this
    # stack, so the block updates the pair representation only and passes
    # ``single`` through untouched.
    #
    # This is cheaper than it looks and more consequential than it looks:
    #
    # * Cheap, because ``single`` is a pure *sink* in this block — it reads the
    #   pair representation through ``pair_logits`` and never writes back. No
    #   pair-track computation depends on it, so deleting it removes no input
    #   from anything upstream.
    # * Consequential, because with the updates gone the trunk's single output
    #   is ``single_activations(target_feat)`` — a *pointwise* function of the
    #   input features, with no token mixing anywhere. Its three consumers each
    #   land differently, and the difference is the whole design:
    #
    #     - the diffusion **atom** encoder
    #       (``AtomEncoder.compute_static(trunk_single=...)``) still needs it,
    #       and keeps it trained: it takes no ``target_feat`` of its own, so
    #       this is its only token-level identity channel;
    #     - the diffusion **conditioning** already has ``target_feat``, so
    #       concatenating this on top is a duplicate — see
    #       ``conditioning_use_trunk_single`` on :class:`DiffusionConfig`;
    #     - the confidence head reads it detached and rebuilds what it needs
    #       from trunk pair + the predicted distogram.
    #
    # At c_z=256, c_s=384, 48 layers this track is 120.6M parameters — 28% of
    # the whole model and 1.6x the trunk's entire pair track — so whether it
    # earns its slot is a first-order question, not a tidiness one.
    #
    # Trunk PairFormer scope only. The confidence-head PairFormer reads its own
    # config instance and keeps its single track (it rebuilds a per-token
    # summary from trunk pair + the predicted distogram, which is the whole
    # point of that stack).
    use_single_track: bool = True

    def __post_init__(self) -> None:
        if self.branch_scaler != "identity":
            conflicts = []
            if self.single_transition_residual_gain is not None:
                conflicts.append("single_transition_residual_gain")
            if self.pair_transition_residual_gain is not None:
                conflicts.append("pair_transition_residual_gain")
            if self.single_transition_residual_per_layer_extra:
                conflicts.append("single_transition_residual_per_layer_extra")
            if self.pair_transition_residual_per_layer_extra:
                conflicts.append("pair_transition_residual_per_layer_extra")
            if conflicts:
                raise ValueError(
                    f"PairFormerConfig.branch_scaler={self.branch_scaler!r} is incompatible "
                    f"with per-branch residual gain knobs: {conflicts}. Magnitude-fixing "
                    "scalers force each branch output to unit-RMS, so extra branch-specific "
                    "gains just rescale the block-wide γ — set them to default (None / False)."
                )


@dataclass
class MaskedMSAConfig:
    """Masked-MSA (BERT) learning head — a masked-language-model objective on
    the MSA + a single-query MSA cross-attention enrichment.

    When ``enabled``, a fraction of MSA entries on the gradient (final
    recycle) pass are corrupted BERT-style before the MSA stack sees them,
    and a head at the end of the MSA module predicts the original residue
    identity at every (row, token). The masking distribution is a categorical
    mixture (``uniform`` / ``profile`` / ``same`` components, remaining mass →
    the ``MSA_MASK`` token).

    The head reads the final MSA-stack representation ``m`` and the trunk
    single representation ``s``:

        h_i        = Attn(q = s_i, k/v = m[:, i])        # over MSA rows at token i
        logits_ri  = Linear(MLP([m_ri, s_i, h_i]))       # per (row r, token i)

    Weights are randomly initialized (AF3 ``linear`` init, not zeros) so the
    head learns from scratch when warm-started from a structure-only run.
    The loss weight lives in :attr:`TrainingConfig.alpha_masked_msa`.

    Attributes:
        enabled: Build + run the head and apply MSA masking when training.
        replace_fraction: Fraction of valid MSA entries to corrupt (BERT mask
            rate, 0.15).
        uniform_prob / profile_prob / same_prob: Categorical replacement
            mixture weights. Remaining mass (``1 - sum``) goes to the
            ``MSA_MASK`` token (defaults all 0.1 → 0.7 mask).
        num_heads / head_dim: Single-query cross-attention shape (q from
            single, k/v from the MSA column).
        hidden_dim: MLP hidden width over ``[m_ri, s_i, h_i]``.
        c_out: Output vocabulary (32 = full DF3 MSA vocab; target ∈ 0..30).
        out_bias: Add a bias to the output logit projection. Masked-MSA has a
            strong class prior (gap / UNK / common-residue frequencies); a
            learnable per-class bias captures it cleanly (zero-init). Set False
            for a strictly bias-free AF3-style head.
    """

    enabled: bool = False
    replace_fraction: float = 0.15
    uniform_prob: float = 0.1
    profile_prob: float = 0.1
    same_prob: float = 0.1
    num_heads: int = 4
    head_dim: int = 32
    hidden_dim: int = 128
    c_out: int = 32
    out_bias: bool = True

    def __post_init__(self) -> None:
        mask_prob = 1.0 - self.uniform_prob - self.profile_prob - self.same_prob
        if mask_prob < 0.0:
            raise ValueError(
                "MaskedMSAConfig: uniform_prob + profile_prob + same_prob must be <= 1 "
                f"(got {self.uniform_prob + self.profile_prob + self.same_prob})"
            )
        if not (0.0 <= self.replace_fraction <= 1.0):
            raise ValueError(f"MaskedMSAConfig.replace_fraction must be in [0, 1], got {self.replace_fraction}")


@dataclass
class DiffusionTransformerConfig:
    num_blocks: int = 24
    num_head: int = 16
    key_dim: int | None = None  # None → use num_channels (c_diff = 768)
    value_dim: int | None = None  # None → use num_channels
    num_intermediate_factor: int = 2
    super_block_size: int = 4
    # AdaLN ``cond_bias`` (attn + ffn) projection init: ``None`` keeps the
    # AF3 zeros init (initial AdaLN bias = 0). A float ``β`` overrides with
    # ``W_b ~ trunc_normal(0, (β/√fan_in)²)`` — output stddev ≈ β. Breaks
    # the zero-wake-up bias in the AdaLN bias path so the conditioning
    # contribution is non-zero from step 0 without amplification.
    # Recommended starting range: β ∈ [1e-3, 3e-3].
    cond_bias_init_beta: float | None = None


@dataclass
class DiffusionConfig:
    conditioning_pair_channel: int = 128
    conditioning_seq_channel: int = 384
    # Whether the diffusion conditioning concatenates the *trunk* single
    # representation onto ``target_feat`` before the LayerNorm + projection.
    #
    # AF3 concatenates both (``cat([s_trunk, target_feat])``, 384 + 447 = 831).
    # That is right when the trunk updates ``s``: the two carry different
    # things, one pair-derived and one raw input. It stops being right the
    # moment ``PairFormerConfig.use_single_track`` is off — then ``s_trunk`` is
    # a fixed pointwise function of ``target_feat``, so the concatenated half
    # carries no information the other half lacks. Keeping it would not merely
    # waste the parameters: the LayerNorm runs over the *concatenated* vector,
    # so a degenerate half also moves the normalization statistics of the live
    # half.
    #
    # Setting this False drops the trunk half, leaving ``LN(target_feat)`` ->
    # projection. Nothing is lost — ``s_trunk = W·target_feat``, and a linear
    # map can absorb ``W``.
    conditioning_use_trunk_single: bool = True
    fourier_dim: int = 256  # Fourier noise embedding dimension
    conditioning_num_intermediate_factor: int = 2  # transition block expansion factor
    # Sampling
    steps: int = 200
    num_samples: int = 5
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    # Inference sampler: drop empty atom windows and compute step-invariant
    # conditioning once (``DiffusionHead.hoist_static``). Same noise draws, 2.8x
    # faster at 566 tokens, lower peak. The output moves within the sampler's own
    # TF32 tolerance: see docs/optimizations.md. On for ``deepfold3 inference``.
    hoist_sampler: bool = False
    # Noise schedule
    sigma_data: float = 16.0
    smin: float = 0.0004
    smax: float = 160.0
    noise_p: float = 7.0
    # Training noise sampling (log-normal): t = sigma_data * exp(p_mean + p_std * z), z ~ N(0,1)
    p_mean: float = -1.2
    p_std: float = 1.5
    # Optional high-σ mixture on the training noise sampler. With probability
    # ``sigma_mix_alpha`` draw σ from a high-σ component instead of the base
    # log-normal σ = σ_data·exp(p_mean + p_std·z), to train the docking / global-
    # arrangement σ regime the base under-covers. The base (median σ≈4.8)
    # concentrates on local-refinement σ, so the high-σ decades — where inference
    # spends ~55% of its steps and the collapse-vs-separate outcome is decided
    # (steps 50–120, σ 18.6–490) — are under-trained.
    #   ``sigma_mix_kind="uniform"`` (simplest ablation): log-uniform over
    #     [σ_data·sigma_mix_smin, σ_data·sigma_mix_smax], i.e. bounded in the
    #     dimensionless ratio s = σ/σ_data via its OWN ``sigma_mix_smin`` /
    #     ``sigma_mix_smax`` (decoupled from the inference schedule's smin/smax).
    #     Defaults 4e-4 / 160 reproduce the old full-range behaviour; raise
    #     ``sigma_mix_smin`` to concentrate the mixture on the high-σ docking
    #     decades and skip the easy low-σ tail. α controls the mix, so α=1.0
    #     makes the sampler a pure log-uniform (flat in log σ, no base — the most
    #     extreme ablation). Enable ``clamp_train_sigma`` so the extreme-low-σ
    #     draws don't spike the EDM weight 1/σ².
    #   ``sigma_mix_kind="lognormal"``: a targeted smooth high-σ log-normal bump
    #     centred at σ = σ_data·exp(sigma_mix_p_mean) (≈90 at 1.73, mid docking
    #     window) with its own tighter ``sigma_mix_p_std`` — no discontinuity,
    #     and (unlike the full-range uniform) no mass wasted at the easy tails.
    # 0.0 = pure base log-normal (default, unchanged). α=0.2 docking σ coverage
    # 18%→~20% (uniform, full range) / ~33% (lognormal, targeted).
    sigma_mix_alpha: float = 0.0
    sigma_mix_kind: str = "uniform"
    # Bounds for the "uniform" mixture, as the dimensionless ratio s = σ/σ_data
    # (σ bounds are σ_data·s). Independent of the inference schedule smin/smax.
    sigma_mix_smin: float = 0.0004
    sigma_mix_smax: float = 160.0
    sigma_mix_p_mean: float = 1.73
    sigma_mix_p_std: float = 1.0
    # EDM diffusion-MSE weight w = (σ² + σ_d²)/(σ·σ_d)² → 1/σ² as σ→0, so a
    # rare small-σ draw in the log-normal's left tail can concentrate the
    # gradient in one sample and spike grad_norm by orders of magnitude.
    # Clamp σ from below during training sampling to cap w ≤ 1/σ_min² + 1/σ_d².
    # σ_min = 0.1 gives w ≲ 100; tighten/loosen via override.
    clamp_train_sigma: bool = False
    train_sigma_min: float = 0.1
    # Disable autocast around the diffusion sampler (Protenix ``skip_amp``
    # pattern, `configs_base.py:137-145`). When True, the method upcasts
    # all float tensor inputs to fp32 and runs with autocast disabled, so
    # the full denoising rollout executes in fp32. Warning: fused Triton
    # kernels in the diffusion head (``use_fused_ada_transition``, etc.)
    # assume bf16 inputs — disable those when enabling skip-amp.
    skip_amp_sample: bool = False  # inference Euler-Maruyama (sample())
    skip_amp_sample_training: bool = False  # training rollout (sample_training())
    # --- Trunk-conditioning dropout (OpenFold-3 ``use_conditioning_prob``) -----
    # Probability of KEEPING the trunk conditioning on a diffusion training
    # forward; ``1.0`` = always keep = current behaviour. OF3 ships 1.0 as the
    # code default and overrides to 0.8 (20% drop) in every training YAML
    # (`examples/training_yamls/initial_training.yml:45`).
    #
    # ``cond_drop`` picks what a drop zeroes: "pair" (default) zeroes only the
    # trunk pair, so the unconditional branch still sees the sequence, the
    # chain topology (``rel_feat``) and the trunk single — the guidance knob
    # below then scales the trunk's pair (contact / distance) signal alone.
    # "both" is OF3's drop of ``single`` and ``pair`` together.
    #
    # Dropped exactly as OF3 does it: the trunk inputs are multiplied by zero
    # *only* on the way into ``DiffusionConditioningBlock.compute_static``
    # (`diffusion_conditioning.py:241-249`). ``target_feat`` and ``rel_feat``
    # survive, the Fourier noise embedding survives, and — matching OF3 /
    # AF3 Algorithm 20 — the **un-zeroed** trunk single still reaches the atom
    # encoder (`diffusion_module.py:198-204`), so this is a *partial* drop, not
    # a true unconditional pass. Do not "fix" that without deciding to diverge.
    #
    # One draw per forward, shared by all ``diffusion_batch_size`` samples
    # (OF3 uses one ``random.random()`` per step). Zeroing an input still
    # reaches every parameter, so DDP ``static_graph=True`` is unaffected.
    #
    # OF3 has no inference-time guidance that consumes this; ``guidance_scale``
    # below does, and needs a checkpoint trained with ``use_conditioning_prob
    # < 1`` and the same ``cond_drop`` — otherwise the unconditional branch is
    # an input the model has never seen.
    use_conditioning_prob: float = 1.0
    cond_drop: Literal["pair", "both"] = "pair"
    # Fraction of drops that zero a random subset of INTER-CHAIN pair blocks
    # instead of the full ``cond_drop`` drop, so the model also learns
    # "these interfaces unconditioned, everything else conditioned" — what
    # ``guidance_drop: interface`` and a custom ``guidance_pair_keep`` sample
    # under. Per drawn subset: r ~ U(0, 1), each unordered chain pair {a, b}
    # dropped with probability r (both (a,b) and (b,a) blocks), at least one
    # pair always dropped, so r → 1 is the all-interfaces drop and small r a
    # single interface. Intra-chain blocks and the trunk single are never
    # touched. Crops where any batch element has < 2 chains have no interface
    # and take the full drop instead. ``0.0`` = off; the extra RNG draw only
    # happens when a drop has already fired, so the default changes nothing.
    cond_drop_interface_frac: float = 0.0
    # --- Classifier-free guidance (inference ``sample()`` only) ---------------
    # Per Euler step, D̃ = D_u + w·(D_c − D_u), with D_u the denoiser under the
    # ``cond_drop`` drop and the SAME noisy positions / augmentation. ``1.0`` =
    # no guidance and no extra forward (bit-identical to the unguided path);
    # ``0.0`` = the unconditional model. ``guidance_sigma_range = (lo, hi)``
    # applies it only where lo ≤ σ̂ ≤ hi (Kynkäänniemi et al. 2024): stereo-
    # chemistry is settled at small σ, where guidance tends to break it. Never
    # applied to the training mini-rollout, so the confidence head is trained
    # on unguided samples — rank guided samples with that in mind.
    guidance_scale: float = 1.0
    guidance_sigma_range: tuple[float, float] | None = None
    # What the unconditional branch drops: "full" = the ``cond_drop`` drop;
    # "interface" = every inter-chain pair block (intra-chain pair and trunk
    # single stay conditioned), i.e. guidance on the trunk's interface signal
    # only. Needs ``cond_drop_interface_frac > 0`` at training. Callers of
    # ``DiffusionHead.sample`` can instead pass ``guidance_pair_keep`` to
    # drop any chosen subset of chain pairs, or ``guidance_terms`` for a linear
    # combination of such terms (``deepfold3 inference --guidance``).
    guidance_drop: Literal["full", "interface"] = "full"
    # --- Masked / partial diffusion (motif scaffolding) -----------------------
    # NOT an OF3 feature — OF3 noises every atom unconditionally
    # (`projects/of3_all_atom/model.py:466-485`). Written from scratch here.
    #
    # With probability ``diffusion_mask_prob`` a diffusion sample keeps part of
    # the (augmented) ground truth CLEAN and noises only the rest, so the model
    # learns to place the noised part *relative to* a given anchor. Motivation
    # (`export/pose-decomposition/`): the inter-chain pose is committed by the
    # first denoise call and no amount of resampling escapes it, and uniform
    # noising cannot break that — any σ low enough to preserve good local
    # structure also preserves the wrong pose, while any σ high enough to erase
    # the pose erases everything and re-commits identically. Selective noising
    # is the only way to erase a pose while keeping a structure.
    #
    # Representation is RePaint-style **replacement at the single global σ**:
    # the clean tokens' coordinates are substituted into the noisy input and
    # the network is left to notice they are clean (they are, trivially: at high
    # σ they are the only atoms with sane bond lengths). Known limitation — the
    # EDM input scaling ``c_in = 1/√(σ²+σ_d²)`` is global, so at σ ≫ σ_data the
    # anchor arrives numerically ~300× smaller than the noised cloud. The
    # principled fix is a per-token σ, which needs the noise embedding and the
    # c_in/c_skip/c_out broadcasts to go per-token (OF3 cannot do it either:
    # its ``t`` is ``[B, S]``).
    #
    # The loss is NOT masked to the noised region. At high σ ``c_skip → 0``, so
    # reproducing the clean region is a real reconstruction task rather than a
    # free copy, and leaving the loss alone keeps this option from touching the
    # loss code at all.
    #
    # ``diffusion_mask_mode``:
    #   "chain" — keep a random non-empty PROPER subset of chains clean (always
    #             ≥1 clean and ≥1 noised). Targets docking directly. Falls back
    #             to "span" on a single-chain crop.
    #   "span"  — keep one contiguous token span clean (classic motif
    #             scaffolding).
    # The clean fraction is drawn uniformly from
    # ``[diffusion_mask_min_frac, diffusion_mask_max_frac]``.
    diffusion_mask_prob: float = 0.0
    diffusion_mask_mode: str = "chain"
    diffusion_mask_min_frac: float = 0.25
    diffusion_mask_max_frac: float = 0.75
    transformer: DiffusionTransformerConfig = field(default_factory=DiffusionTransformerConfig)
    atom_cross_att: AtomCrossAttConfig = field(default_factory=AtomCrossAttConfig)

    def __post_init__(self) -> None:
        if self.cond_drop not in ("pair", "both"):
            raise ValueError(f"DiffusionConfig.cond_drop must be 'pair' or 'both', got {self.cond_drop!r}")
        if self.guidance_drop not in ("full", "interface"):
            raise ValueError(f"DiffusionConfig.guidance_drop must be 'full' or 'interface', got {self.guidance_drop!r}")
        if not 0.0 <= self.cond_drop_interface_frac <= 1.0:
            raise ValueError(
                f"DiffusionConfig.cond_drop_interface_frac must be in [0, 1], got {self.cond_drop_interface_frac}"
            )
        if self.guidance_sigma_range is not None:
            self.guidance_sigma_range = tuple(self.guidance_sigma_range)
            lo, hi = self.guidance_sigma_range
            if not lo <= hi:
                raise ValueError(
                    f"DiffusionConfig.guidance_sigma_range needs lo <= hi, got {self.guidance_sigma_range}"
                )


@dataclass
class ConfidenceConfig:
    num_layer: int = 4
    pair_channel: int = 128
    seq_channel: int = 384
    num_plddt_bins: int = 50
    num_distogram_bins: int = 64
    max_error_bin: float = 31.0
    max_atoms_per_token: int = 24  # AF3 dense atom representation max
    pair_attention_num_head: int = 4
    single_attention_num_head: int = 16
    # Residual Output Scaling (see EvoformerConfig). L = this stack's num_layer (4).
    residual_output_scaling: bool = False
    uniform_residual_scale: bool = False
    # See PairFormerConfig.pair_bias_gate.
    pair_bias_gate: bool = False
    # Probability of KEEPING the trunk PAIR embedding in the confidence head
    # (OpenFold-3 ``use_confidence_emb_prob``, `head_modules.py:195-196`).
    # ``1.0`` = always keep = current behaviour; OF3's training YAMLs use 0.8.
    # Training-only, pair-only — the single repr, ``target_feat`` and the
    # predicted positions are never dropped. The distogram head reads the
    # un-zeroed pair (it is computed from ``embeddings["pair"]`` separately),
    # so the distogram loss is unaffected, matching OF3's ordering.
    use_pair_emb_prob: float = 1.0
    pair_bias_gate_init_tau: float = 0.25
    # See PairFormerConfig.single_transition_linear_out_init_gain.
    single_transition_linear_out_init_gain: float | None = None
    # Mirror of PairFormerConfig.{single,pair}_transition_residual_gain /
    # _per_layer_extra for the confidence-head PairFormer (L=4).
    # ``None``/``False`` ≡ no extra gain.
    single_transition_residual_gain: float | None = None
    single_transition_residual_per_layer_extra: bool = False
    pair_transition_residual_gain: float | None = None
    pair_transition_residual_per_layer_extra: bool = False
    # Mirror of PairFormerConfig.branch_scaler — function applied to every
    # confidence-head PairFormer residual branch output before the per-block
    # γ. Same kinds + validation rule.
    branch_scaler: Literal["identity", "rmsnorm"] = "identity"

    def __post_init__(self) -> None:
        if self.branch_scaler != "identity":
            conflicts = []
            if self.single_transition_residual_gain is not None:
                conflicts.append("single_transition_residual_gain")
            if self.pair_transition_residual_gain is not None:
                conflicts.append("pair_transition_residual_gain")
            if self.single_transition_residual_per_layer_extra:
                conflicts.append("single_transition_residual_per_layer_extra")
            if self.pair_transition_residual_per_layer_extra:
                conflicts.append("pair_transition_residual_per_layer_extra")
            if conflicts:
                raise ValueError(
                    f"ConfidenceConfig.branch_scaler={self.branch_scaler!r} is incompatible "
                    f"with per-branch residual gain knobs: {conflicts}. Magnitude-fixing "
                    "scalers force each branch output to unit-RMS, so extra branch-specific "
                    "gains just rescale the block-wide γ — set them to default (None / False)."
                )


@dataclass
class TemplatePairFormerConfig:
    """Config for the simplified PairFormer stack inside template embedding.

    Uses pair-only layers (no single track) with a smaller transition factor.
    """

    num_layer: int = 2
    num_channels: int = 64
    num_heads: int = 4
    num_intermediate_factor: int = 2
    pair_dropout: float = 0.25
    # Residual Output Scaling (see EvoformerConfig). L = this stack's num_layer (2).
    residual_output_scaling: bool = False
    uniform_residual_scale: bool = False


@dataclass
class TemplateEmbeddingConfig:
    """Config for template embedding (structural templates → pair embedding).

    Haiku scope: diffuser/evoformer/template_embedding/

    Attributes:
        num_channels: Hidden dimension for template pair features.
        pair_channel: Pair channel from the Evoformer (query embedding dim).
        dgram_num_bins: Number of distance histogram bins.
        dgram_min_bin: Left edge of the first distance bin (Angstroms).
        dgram_max_bin: Left edge of the last distance bin (Angstroms).
        aatype_num_classes: Number of amino acid type classes for one-hot encoding.
        pairformer: Config for the template PairFormer stack.
    """

    num_channels: int = 64
    pair_channel: int = 128
    dgram_num_bins: int = 39
    dgram_min_bin: float = 3.25
    dgram_max_bin: float = 50.75
    aatype_num_classes: int = 31
    pairformer: TemplatePairFormerConfig = field(default_factory=TemplatePairFormerConfig)


# ``--optim`` names. ``train-matched`` keeps what the checkpoint recorded.
OPTIM_PRESETS = ("none", "train-matched", "inference", "inference-cueq")


@dataclass
class OptimConfig:
    """Toggle individual kernel optimization backends.

    Flags default to False (pure PyTorch). A requested kernel whose backend is
    not installed refuses at model construction (:meth:`require_backends`). One
    that a condition gates off, such as a channel count or a CPU tensor, runs
    stock and is named by ``kernel_census``.

    Benchmark results (RTX 5090, bf16, N=384-768):
    Measured at C=128 with trunk triangle attention. Current configs use
    pair_channel=256 and use_triangle_attention=false, so the module counts below
    and GridSelfAttention's share of the runtime do not apply to them.
    ──────────────────────────────────────────────────────────────────
    ATTENTION (104 GridSelfAttention modules per forward):
    - use_fused_evo_attention: 2.9-3.3x faster (autotuned Triton flash-attn with pair bias)
    - use_fused_pair_bias: +30% on top of evo_attn (fused LN + tl.dot projection + permute)
      Supports training (autograd backward) + inference (fully fused).
    - Combined evo+pair_bias: 3.7-4.6x faster vs pytorch baseline
    - use_torch_compile: 1.3-2.3x for MSAAttention/SingleAttention (SDPA backend)
    Note: Triton EvoAttention is NOT used for SingleAttention (N_seq=1 overhead dominates).

    TRIANGLE MULTIPLY (96 TriangleMul modules per forward):
    - use_fused_triangle_mul: enables three fusion levels:
      Level 3 (dual GEMM): 1.24-1.32x fwd, 1.18-1.20x full
        Supports TWO_INPUTS, bias, mask fusion, TF32 precision control.
      Level 2 (center proj): fused LN+Linear for post-einsum norm_center+out_proj
      Level 1 (gate): fallback fused LN+packed GEMM+sigmoid postprocess

    SWIGLU (inside TransitionBlock / AdaTransitionBlock):
    - use_fused_swiglu: 1.4-1.6x fwd, 1.9-2.1x full at pair scale (M=N², C=128)
      Multi-row Triton kernel (128 rows/block) + in-place backward → 2-6x less memory.
      Neutral at single scale (M=N) where launch overhead dominates — still safe to enable
      because the memory saving is always beneficial.

    TRANSITION (context-sensitive — M size determines whether Triton wins):
    - use_fused_transition_pair: 1.2-1.3x for pair transitions (M=N², C=128) where M is large
    - use_fused_transition_single: SLOWER at M=N, C=384 — DO NOT enable (PyTorch 0.8x)
    - use_fused_ada_transition: SLOWER at M=N, C=768 — DO NOT enable (PyTorch 0.8x)
    - use_fused_ln_transpose: 1.7-2.3x fwd for LN+transpose (column attn, inference only)
    """

    use_fused_evo_attention: bool = False
    use_fused_pair_bias: bool = False
    use_fused_swiglu: bool = False  # standalone FusedSwiGLU — wins at M=N² (pair transitions)
    use_fused_transition_pair: bool = False  # pair transitions (M=N², C=128) — Triton wins
    use_fused_transition_single: bool = False  # single transitions (M=N, C=384) — PyTorch wins, DO NOT enable
    use_fused_ada_transition: bool = False  # diffusion AdaTransitionBlock (M=N, C=768) — PyTorch wins
    use_fused_triangle_mul: bool = False
    # cuEquivariance triangle multiply (optional ``cueq`` group). Takes precedence
    # over use_fused_triangle_mul; refuses at construction if not installed.
    use_cueq_triangle_mul: bool = False
    # Triangle multiply on channel-first planes with a fused backward; see
    # model/triton/fused_trimul.py. Calls the kernels cannot serve run the same layout
    # in plain torch. Takes precedence over use_fused_triangle_mul.
    use_channel_major_triangle_mul: bool = False
    use_fused_ln_transpose: bool = False
    # Evoformer MSA transition as one in-place forward-only kernel (no [..., 2*hidden]
    # tensor; see model/triton/msa_transition.py). Served only with grad disabled, so
    # keep it off in training presets. A call it cannot serve logs why and goes stock.
    use_fused_msa_transition_inference: bool = False
    # dropout_add fuses per-batch row/column-wise dropout with a residual add and
    # samples the mask on the fly via tl.rand (see model/triton/dropout_add.py).
    # Default True preserves current behavior (Triton always active on CUDA);
    # set False to force the unfused PyTorch fallback for A/B diagnostics.
    use_fused_dropout_add: bool = True
    use_torch_compile: bool = False
    # Tiled linear: chunk pair transitions along dim=-2, recompute during backward
    # Saves ~67% of pair transition activation memory at cost of ~2x FLOPs for those layers.
    # Also supports inference tiling for very long sequences (peak memory reduction).
    use_tiled_linear: bool = False  # tiled recomputation for pair-scale transitions
    tiled_linear_shards: int = 0  # 0 = auto heuristic (ceil(spatial/channels))
    tiled_linear_inference: bool = False  # also tile during inference (for long sequences)
    # Tiled outer product mean: chunk rows to avoid [N,N,32,32] intermediate.
    # Saves ~3.6 GB at N=768. 0 = disabled, >0 = chunk size (128 recommended).
    tiled_opm_chunk: int = 0
    # Inference only: evaluate the row-independent MSA ops (transition, row
    # attention, OPM projections) this many MSA rows at a time into one output.
    # Their intermediates dominate peak memory at deep MSAs, far above the MSA
    # itself, and a fixed row count keeps Triton autotune keys constant across
    # MSA depths. 0 = whole tensor. Ignored whenever grad is enabled.
    msa_row_chunk: int = 0

    # Gradient checkpointing on the 4-layer Evoformer (MSA) stack and the
    # 48-layer PairFormer trunk. Essential for training — without it, 48 layers
    # of [N,N,128] pair activations accumulate and peak memory easily exceeds
    # 100+ GB at N=384. Each block is individually recomputed in the backward
    # pass. Adds ~30% runtime overhead but drops activations from O(L) to O(1).
    # No effect on inference (only applied when grad is enabled).
    use_gradient_checkpointing: bool = False

    # Legacy fused-kernel parameter substrings. Current fused wrappers are
    # stateless (no owned parameters), but older checkpoints may contain
    # these keys; they are filtered out of state-dict loads and EMA shadow.
    FUSED_STATE_DICT_MARKERS: tuple[str, ...] = (
        "_fused_gate.",
        "_fused_center_proj.",
        "_fused_dual_gemm.",
        "_fused_pair_bias.",
        "_fused_ada_ln_linear.",
        "_fused_swiglu.",
        "_ln_linear.",
        "_ln_transpose.",
        "_swiglu.",
    )

    @classmethod
    def none(cls) -> OptimConfig:
        """Stock PyTorch at every dispatch site."""
        return cls(use_fused_dropout_add=False)

    @classmethod
    def inference(cls) -> OptimConfig:
        """Maximum inference speed and memory efficiency.

        Adds forward-only kernels and MSA chunking to the trained set, so its
        numerics differ from training within the error of each kernel.
        """
        return cls(
            use_fused_evo_attention=True,
            use_fused_pair_bias=True,
            use_fused_swiglu=True,
            use_channel_major_triangle_mul=True,
            use_fused_transition_pair=True,
            use_fused_msa_transition_inference=True,
            tiled_opm_chunk=128,
            msa_row_chunk=512,
        )

    @classmethod
    def inference_cueq(cls) -> OptimConfig:
        """``inference`` with cuEquivariance serving the triangle multiply.

        Needs the optional ``cueq`` dependency group; construction refuses without it.
        """
        cfg = cls.inference()
        cfg.use_channel_major_triangle_mul = False
        cfg.use_cueq_triangle_mul = True
        return cfg

    @classmethod
    def from_preset(cls, name: str, checkpoint: OptimConfig | None) -> OptimConfig:
        """Resolve a ``--optim`` name.

        Args:
            name: One of :data:`OPTIM_PRESETS`.
            checkpoint: The config the checkpoint recorded, or ``None`` if it has none.

        Returns:
            The config to build the model with.

        Raises:
            ValueError: ``train-matched`` was asked for and the checkpoint records no config.
        """
        if name == "train-matched":
            if checkpoint is None:
                raise ValueError("--optim train-matched needs a checkpoint that records its OptimConfig")
            return checkpoint
        return {"none": cls.none, "inference": cls.inference, "inference-cueq": cls.inference_cueq}[name]()

    def require_backends(self) -> None:
        """Refuse a requested kernel whose backend cannot be imported.

        ``use_fused_dropout_add`` is exempt: it defaults on, so it is not a request.

        Raises:
            NotActive: A requested Triton kernel has no Triton to run on.
        """
        levers = [
            f
            for f in (
                "use_fused_evo_attention",
                "use_fused_pair_bias",
                "use_fused_swiglu",
                "use_fused_transition_pair",
                "use_fused_transition_single",
                "use_fused_ada_transition",
                "use_fused_triangle_mul",
                "use_fused_ln_transpose",
                "use_fused_msa_transition_inference",
            )
            if getattr(self, f)
        ]
        if not levers:
            return
        from deepfold3.model.inference_guard import NotActive
        from deepfold3.model.triton import HAS_TRITON

        if not HAS_TRITON:
            raise NotActive(levers[0], f"Triton is not installed; also requested: {levers[1:]}")


@dataclass
class DatasetConfig:
    """Configuration for a training or evaluation dataset source.

    Attributes:
        name: Identifier for logging.
        data_dir: Path to data directory (structures/ + msa/).
        weight: Sampling probability relative to primary dataset (1.0).
        strategy: ``"weighted"`` (cluster-balanced) or ``"uniform"``.
        is_distillation: If True, disables confidence losses (PDB-only per AF3 §4.3).
            Diffusion and distogram losses still apply.
        format: ``"df3"`` (default), ``"of3"`` (OpenFold 3 preprocessed monomer
            tree — ``{data_dir}/{id}/structure.npz``), or ``"of3pdb"``
            (OpenFold 3's ``pdb_training_set``: ``data_dir`` is the set's root
            and its metadata comes from the columnar cache at
            ``of3_cache_dir``). ``of3pdb`` is the only format that carries
            AF3 §2.5.1 chain / interface clusters of its own, so pair it with
            ``strategy: weighted``.
        cutoff_date: ISO date filter. For training datasets: records released
            AFTER this date are excluded. For eval datasets: only records
            released AFTER this date are included.
        exclude_loss_mol_types: List of integer ``mol_type`` values whose tokens
            should be **dropped from all per-atom and per-pair losses** while
            still appearing in the model input. The dataset synthesises a
            ``MASK_LOSS_WEIGHT`` token-mask channel from the tokenized
            ``mol_type`` array (1.0 elsewhere, 0.0 on excluded mol_types).
            Default empty = no exclusion. Common use: PATCHR distillation
            sets ``[3]`` (``MOL_NONPOLYMER``) to drop inpainted ligand poses.
        crop_weights: Optional per-dataset crop-strategy weights, e.g.
            ``{"contiguous": 0.2, "spatial": 0.8, "interface": 0.0}``. Keys are
            any subset of ``contiguous`` / ``spatial`` / ``interface``; missing
            keys inherit the global ``data.*_crop_weight`` defaults. Lets a
            monomer distillation set skip (wasted) interface crops while a
            multi-chain PDB set keeps them (AF3 §2.7 samples the strategy
            per example). ``None`` = use the global weights.
        template_struct_space: If True, this dataset's template indices —
            ``monomer_templates/{msa_id}.json`` **and** ``templates/{id}.json``
            — are structure-space (0-based resolved-residue) rather than
            SEQRES-space, so the seqres→structure realignment is skipped at
            feature extraction. Default False (SEQRES-space).

            Get this wrong and templates silently become noise: the two frames
            differ by the number of unresolved N-terminal residues, which varies
            per template. Measured on the shipped indices by re-deriving each
            hit's sequence identity and comparing it to the identity the search
            recorded — under the SEQRES reading the medians missed by 74 / 26 /
            16 points (patchr / afdb_nvda / teddymer), under the structure
            reading by 0.5 / 1.9 / 1.4, with ~98% of hits within 10 points. All
            three shipped sets are structure-space.

            **Only applies to ``format: df3`` datasets.** OF3 records store the
            alignment as ``template.npz``'s ``idx_map``, which is 1-based SEQRES
            by construction; ``OF3Dataset`` takes no such argument, so setting
            this on a ``format: of3`` entry does nothing (see
            ``training/data.py:of3_kwarg_names``).

            ``monomer_template_struct_space`` is accepted as a deprecated alias.
        max_template_date: Per-dataset override of the global
            ``data.max_template_date`` hard cap on template release dates
            (ISO ``YYYY-MM-DD``). ``""`` (default) inherits the global value.
            Set this to AF3's distillation cap ``2018-04-30`` for AlphaFold-
            predicted distillation sets (MGnify / AFDB): their pseudo-labels
            were generated by AlphaFold 2, whose own template cutoff is
            2018-04-30, so a later template would feed the model structural
            information the label generator never had (AF3 §2.4).
        curator_weights: ``format: of3pdb`` only — path to an ``.npz`` holding
            ``pdb_id`` (``<U8``) and ``weight`` (``f8``) arrays, a per-entry
            MULTIPLIER on the AF3 §2.5.1 cluster marginal. The dataset draws
            record ``i`` with probability ∝ ``w_i^AF3 · m_i``, so ``m ≡ 1``
            reproduces the baseline exactly and any model-aware curation
            (difficulty, overconfidence, difficulty-balanced, band-pass) enters
            as a single file rather than a code path. Entries absent from the
            file keep ``m = 1``; the dataset logs the coverage. Empty = off.
        of3_cache_dir: ``format: of3pdb`` only — directory written by
            ``deepfold3 build-of3-cache`` holding the memory-mapped columnar
            conversion of OpenFold 3's ``dataset_caches/*.json``. Empty
            defaults to ``{data_dir}/df3_cache``. The raw 1.7 GB JSON is never
            read at train time; see ``data/prepare/of3_pdb_cache.py``.
        template_cache_dir: ``format: of3pdb`` only — directory of per-query
            template hit lists (``{aln_rep}.npz``). Upstream splits these by
            subset (``templates/train_template_cache`` /
            ``templates/val_template_cache``) while the structures, alignments
            and template geometry stay shared, so a **validation** split under
            the same ``data_dir`` needs this pointed at the val cache. Empty
            defaults to the train cache, falling back to the val one only when
            the train directory is absent — which for a full mirror means an
            eval split would silently find hits for a small minority of its
            chains (measured 155/833 vs 640/833) and be evaluated nearly
            template-free while training is not.
    """

    name: str = ""
    data_dir: str = ""
    weight: float = 1.0
    strategy: str = "uniform"
    is_distillation: bool = False
    format: str = "df3"
    cutoff_date: str = ""
    exclude_loss_mol_types: list[int] = field(default_factory=list)
    crop_weights: dict[str, float] | None = None
    template_struct_space: bool = False
    max_template_date: str = ""
    of3_cache_dir: str = ""
    template_cache_dir: str = ""
    curator_weights: str = ""


@dataclass
class TrainingConfig:
    """Training hyperparameters.

    All variable training settings are collected here; nothing is hardcoded
    in the trainer or loss modules.  Override via JSON config or CLI flags.

    Attributes:
        lr: Peak learning rate for Adam.
        weight_decay: Weight decay (applied to >= 2-D params only).
        betas: Adam beta coefficients.
        eps: Adam epsilon.
        warmup_steps: Linear LR warmup duration.
        start_decay_after: Step at which LR decay begins.
        decay_every: Steps between each LR decay step.
        decay_factor: Multiplicative LR decay factor.
        grad_clip_norm: Max L2 gradient norm (0 disables).
        accumulate_grad_batches: Gradient accumulation steps.
        ema_decay: EMA decay factor (negative disables EMA).
        max_tokens: Maximum tokens per crop.
        max_atoms: Maximum atoms per crop.
        max_msa_seqs: Maximum MSA sequences.
        max_recycles: Upper bound for random recycle count per batch.
        diffusion_batch_size: Number of noise samples per training step.
        mini_rollout_steps: Denoising steps for the confidence mini-rollout.
        mini_rollout_samples: Diffusion samples in the mini-rollout.
        alpha_diffusion: Diffusion loss weight.
        alpha_distogram: Distogram loss weight.
        alpha_confidence: Confidence loss weight.
        alpha_pae: PAE loss multiplier (0 in stage 1, 1 in finetuning).
        alpha_bond: Bond loss multiplier (0 in stage 1).
        lddt_cutoffs: Smooth LDDT distance error thresholds (Angstroms).
        lddt_inclusion_radius: Smooth LDDT pair inclusion radius (Angstroms).
        min_resolution: Minimum resolution for confidence loss gating (Angstroms).
        max_resolution: Maximum resolution for confidence loss gating (Angstroms).
        dtype: Mixed-precision dtype (``"bf16"``, ``"fp16"``, or ``"fp32"``).
        find_unused_parameters: DDP ``find_unused_parameters`` flag.
        static_graph: DDP ``static_graph`` flag.
        max_steps: Total effective training steps.
        seed: Base random seed (offset by rank for DDP).
        deterministic: If True, force deterministic algorithms
            (``torch.use_deterministic_algorithms(True)``,
            ``cudnn.deterministic = True``, ``cudnn.benchmark = False``)
            for bit-exact reproducibility. Incurs a significant perf cost.
        checkpoint_interval: Save a checkpoint every N effective steps.
        eval_interval: Run evaluation every N effective steps.
        log_interval: Log metrics every N effective steps.
        output_dir: Output directory for checkpoints and logs.
        data_dir: Root directory containing training data.
        params_path: Path to pretrained weights (safetensors or .pt).
    """

    # Optimizer (Adam)
    lr: float = 0.0018
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    # "adamw" = torch.optim.AdamW + ``finetune_scheduler`` LambdaLR.
    # "schedule_free" = AdamWScheduleFree (Defazio et al. 2024); ``finetune_scheduler``
    # is ignored — SF manages its own warmup-then-flat schedule via ``warmup_steps``.
    optimizer_type: str = "adamw"

    # LR schedule (AF3-style: warmup, plateau, step decay)
    warmup_steps: int = 1000
    start_decay_after: int = 50000
    decay_every: int = 50000
    decay_factor: float = 0.95

    # Gradient
    grad_clip_norm: float = 10.0
    accumulate_grad_batches: int = 1
    # Raise RuntimeError after this many consecutive opt-step skips from
    # non-finite loss/grads. 0 or negative disables the check.
    max_consecutive_nan_skips: int = 5
    # Compute the grad-norm reduction in fp64 (upcast per-param L2² before
    # summing). Removes spurious non-finite grad-norms from fp32 reduction
    # overflow that differs across ranks by summation order. Default off
    # to preserve bit-exact parity with the historical ``clip_grad_norm_``
    # path; enable when training at high grad magnitudes.
    grad_norm_fp64: bool = False
    # When the observed (pre-clip) grad-norm exceeds this threshold, each
    # rank logs the record_ids that contributed to its accumulation window.
    # 0 disables. Useful for identifying examples that drive grad spikes
    # but stay below the NaN/Inf skip trigger.
    grad_norm_log_threshold: float = 0.0
    # If ANY rank's pre-clip grad-norm exceeds this value, skip the whole
    # optimizer step (per-rank outlier guard). Clipping to ``grad_clip_norm``
    # from a huge pre-clip norm produces a near-random direction that
    # degrades update quality and feeds magnitude drift; dropping the step
    # is cleaner than applying garbage. 0 disables.
    grad_norm_skip_threshold: float = 0.0
    # Per-rank gradient clipping (OF3-style). When True the ENTIRE backward
    # runs under ``no_sync`` (DDP's automatic gradient all-reduce is
    # suppressed), each rank clips its own LOCAL gradient to
    # ``grad_clip_norm``, and the clipped per-rank grads are then averaged
    # with a single manual all-reduce. This bounds any one rank's
    # contribution BEFORE averaging, so a pathological sample cannot dominate
    # the batch-mean gradient (standard global clipping clips the
    # already-averaged gradient, after the bad sample has been mixed in).
    #
    # Cost: the manual all-reduce runs after the full backward, so it no
    # longer overlaps the backward pass, and DDP's ``reduce_dtype`` comm hook
    # is bypassed (reduction is fp32). ``static_graph`` is forced off (it is
    # incompatible with the always-on ``no_sync``). With
    # ``batch_size_per_rank > 1`` the clip is per-rank over the within-rank
    # sample mean, not literally per-sample. Default off; see
    # ``docs/training.md`` § Gradient Management.
    per_rank_grad_clip: bool = False

    # EMA
    ema_decay: float = 0.999
    # Eval snapshot — applies to run_eval forward pass. "average" reads
    # SF's x directly (responsive, current state). "shadow" reads EMA-over-x
    # (slow, 1/(1-ema_decay) lag). Training-time shadow update is unaffected.
    # NOTE: on a non-SF optimizer (AdamW) there is no averaged iterate, so
    # "average" would read the raw live params (never deployed). run_eval
    # upgrades "average" -> shadow automatically when EMA is active on AdamW.
    eval_snapshot: Literal["average", "shadow"] = "average"
    # Refined SF decoupling parameter C (Through the River, arxiv 2507.09846,
    # Algorithm 2 line 11). ``None`` keeps vanilla SF (`c_t = γ_t²/Σγ_i²`); a
    # float ``C`` scales that weight by ``(1−β₁)·C`` (clamped to 1), decoupling
    # β₁ from x's averaging window. Ignored unless ``optimizer_type ==
    # "schedule_free"``.
    decoupling_parameter: float | None = None

    # Per-bucket strong weight decay. Maps a substring pattern → multiplier;
    # weights whose ``named_parameters`` name contains any key get routed to
    # a dedicated AdamW param group with ``weight_decay = multiplier · weight_decay``.
    # First match wins (substrings checked in iteration order). Empty dict
    # (default) disables — all decay-eligible params land in the standard group.
    #
    # Motivation: ``single_transition_residual_gain=0.35`` (s035 era) is a
    # forward-time 0.35× scaling of single_transition's residual contribution.
    # Under AdamW the stationary norm ``|w|²* ~ η/(2λ_w)·E|Pg|²``, so achieving
    # |w|* ≈ 0.35·|w|_normal requires ``λ_w / λ_w_normal = 1/0.35² ≈ 8.16``.
    # With multiplier=8 on the targeted weights they settle to ~0.35× normal
    # magnitude — approximating the gain's steady-state forward output (the
    # transient differs: gain is immediate, wd-driven shrinkage is gradual).
    #
    # Example (replicate s035 single_transition brake on the output projection):
    #   strong_decay:
    #     ".single_transition.linear2": 8.0
    strong_decay: dict[str, float] = field(default_factory=dict)

    # Data
    max_tokens: int = 384
    max_atoms: int = 4096
    max_msa_seqs: int = 16384

    # Recycling (random 0..max_recycles each microstep, AF3 convention).
    # Total trunk forwards = num_recycles + 1.
    # max_recycles=3 → up to 4 trunk passes (matches AF3 paper).
    max_recycles: int = 3
    # MSA pre-crop (training/eval forward_training). When True the featurizer
    # emits a FIXED ``[(max_recycles+1) * msa_crop_size, N]`` stack of one crop
    # per recycle (CPU resample), so the GPU MSA shape is constant every step —
    # removes the per-step variable MSA allocation that drives caching-allocator
    # fragmentation (the dt-spike straggler). ``profile`` / ``deletion_mean``
    # still use the full pool. Inference (``forward``) keeps the legacy on-GPU
    # resample. False = legacy variable-depth pool everywhere.
    msa_precrop: bool = False

    # Diffusion training
    diffusion_batch_size: int = 48
    diffusion_chunk_size: int = 0  # 0 = no chunking; >0 = process S samples in chunks
    mini_rollout_steps: int = 20
    mini_rollout_samples: int = 1

    # Chain-permutation stability gate (default off). When on, the shared chain
    # permutation is resolved from ``chain_perm_gate_k`` INDEPENDENT mini-rollout
    # samples and applied ONLY when all K samples agree on the full assignment;
    # otherwise it falls back to identity (un-permuted GT). This suppresses the
    # oscillating-target latch that stalls symmetric-multimer symmetry-breaking:
    # while a homo-oligomer is collapsed the K noisy rollouts disagree on the
    # coin-flip assignment → identity (a deterministic separating target); once
    # the copies genuinely separate the K rollouts agree → the AF3 permutation
    # fires. Sim (export/chain-perm-latch/): K=3 recovers the df3v2 latch zone,
    # K=5 is safe to extreme rollout noise; majority voting does NOT work — the
    # gate must be unanimous. K raises the mini-rollout to K samples (no_grad);
    # the confidence head still sees only ``mini_rollout_samples`` of them.
    chain_perm_stability_gate: bool = False
    chain_perm_gate_k: int = 5

    # Loss weights
    alpha_diffusion: float = 4.0
    alpha_distogram: float = 0.03
    alpha_confidence: float = 1e-4
    alpha_pae: float = 0.0
    alpha_bond: float = 0.0
    # Feed the symmetry-swappable atom pairs (ASP OD1/OD2, PHE/TYR ring,
    # ARG NH1/NH2, LEU/VAL gem-dimethyls, nucleic OP1/OP2, atomized/ligand
    # 2-cycles) into the bond loss as extra distance constraints. Their target
    # ``||gt_a - gt_b||`` is invariant to the a<->b labeling, so this is a
    # coin-flip-free, per-pair-normalised (un-diluted) penalty that stops the
    # symmetric-atom midpoint collapse the per-residue symmetry renaming
    # otherwise leaves ~loss-free under diffusion MSE. Shares ``alpha_bond``
    # (requires ``alpha_bond > 0``); no separate weight. Now REDUNDANT — the
    # permute-then-noise ordering fix below is the root fix; keep False.
    bond_include_symmetric: bool = False
    # Permute-then-noise (AF3 SI §4.2/4.3, matches OpenFold3/Protenix). When True,
    # ``forward_training`` resolves the chain + intra-residue atom/ligand symmetry
    # ONCE from the mini-rollout and applies it to the GT BEFORE the diffusion
    # noising, so the denoiser's noised input and its MSE target share one atom
    # labelling. DeepFold3's legacy path noised the un-permuted GT and permuted
    # the target only at loss time (noise-then-permute) — when the rollout swaps a
    # symmetric pair / chain, input and target conflict and the expected-loss
    # minimiser is the pair/assembly midpoint (symmetric-atom collapse + homomer
    # chain latch). Fixing the order removes that conflict at the root: makes
    # ``bond_include_symmetric`` and the K-rollout chain-perm gate redundant.
    # Default True (AF3/OF3/Protenix-conformant): empirically confirmed to clear the
    # symmetric-atom collapse (better than bond_include_symmetric, bond loss off) and
    # reverse the homomer docking plateau. Set False for the legacy noise-then-permute.
    permute_gt_before_noise: bool = True
    # Masked-MSA (BERT) cross-entropy weight. Auxiliary self-supervised
    # objective on the MSA representation; predicts the original residue
    # identity at masked (row, token) entries. Off by default. Requires
    # ``ModelConfig.masked_msa.enabled``. Keep it small (our distogram CE is
    # 0.03) so it nudges the MSA stack without dominating the structure losses.
    alpha_masked_msa: float = 0.0
    # Chirality quadratic loss weight — AF3-spec deviation. Off by default;
    # turn on (e.g. ``1.0``) to add quadratic-toward-target supervision on
    # the Cβ projection onto the (N/CA/C) frame normal. Provides explicit
    # sign-and-magnitude signal that distance-based losses (smooth_lddt,
    # distogram) lack. See ``loss.py:chirality_loss`` and the
    # featurizer-side ``chirality_*`` features. Currently protein-only
    # (RNA/DNA TODO via N1/N9).
    alpha_chirality: float = 0.0
    # Diagnostic threshold (Å) for the logged ``chirality_ambiguous_rate``
    # — residues with ``correctness < margin`` are counted as ambiguous.
    # NOT a loss hyperparameter; the quadratic chirality loss is
    # margin-free. The CCD ideal projection is ≈ ±0.95 Å so ``0.5`` flags
    # residues that have collapsed toward the (N, CA, C) plane.
    chirality_margin: float = 0.5
    # Backbone-trace chirality loss weight — Cα 4-quadruplet pseudo-dihedral
    # ``sin``-match. Complements ``alpha_chirality`` (which only fixes the
    # local Cβ stereocentre): empirically, the Cβ loss drove mirror_rate
    # 38→5 % on ``df3-chiral`` but ``det(V·U^T)`` stayed −1 for every
    # inference ckpt (backbone helix still mirror). This term targets the
    # missing gradient on backbone handedness. Off by default; flip on
    # (e.g. ``1.0``) for stage-1 mirror remediation.
    alpha_backbone_chirality: float = 0.0
    # FAPE (Frame-Aligned Point Error) — AF2 §4.2.7. For every ordered
    # token pair (i, j), projects token j's representative atom into
    # token i's local rigid frame in both pred and GT, clamps the L2
    # error at ``fape_clip_distance``, and (when ``fape_normalize``)
    # divides by the clamp so the loss is dimensionless ``∈ [0, 1]``.
    # The Gram-Schmidt frames encode chirality, so a mirrored prediction
    # gives saturated loss everywhere — the gradient signal that diffusion
    # MSE + smooth_lddt lacks (cf. ``df3-chiral`` 2026-05-19→20: Cβ-only
    # and Cβ+backbone-Cα-dihedral chirality losses both failed to flip
    # the helix on 11CF). Off by default; AF2 used ``0.5`` paired with
    # ``alpha_distogram=0.3`` (our distogram is at ``0.03`` instead, so
    # the FAPE/distogram ratio doesn't transfer — tune against
    # ``alpha_diffusion=4.0`` instead).
    alpha_fape: float = 0.0
    fape_clip_distance: float = 10.0
    fape_normalize: bool = True
    # Which (frame, point) pairs FAPE scores:
    #   "pooled" — every pair in one term under ``fape_clip_distance``. Plain
    #              AF2, and the default, so existing ``alpha_fape`` runs are
    #              unchanged. ``alpha_fape_inter`` is ignored.
    #   "both"   — AF-Multimer §2.1: same-chain pairs at ``fape_clip_distance``
    #              (weight ``alpha_fape``), cross-chain at
    #              ``fape_clip_distance_inter`` (weight ``alpha_fape_inter``).
    #   "intra"  — same-chain pairs only; cross-chain pairs are dropped.
    #   "inter"  — cross-chain pairs only (weight ``alpha_fape_inter``).
    #
    # Two reasons the cross-chain half is its own term rather than pooled under
    # one clamp: (1) at 10 Å a chain docked 50 Å from its site is fully
    # saturated, i.e. gets *zero* gradient, so pooled/intra FAPE cannot correct
    # a pose at all; (2) the cross-chain share of all pairs swings with how many
    # chains a crop happens to hold (~50% for a dimer, <10% for a large
    # assembly), so pooling makes the docking signal's strength a function of
    # crop composition. Separate denominators remove that, and ``fape_inter`` is
    # logged on its own — the term exists to be watched.
    #
    # Turning the cross-chain term on raises the cost of a wrong chain
    # permutation: the intra term is permutation-invariant, the inter one is
    # not. Check ``chain_perm_gate_k`` before enabling.
    fape_mode: str = "pooled"
    alpha_fape_inter: float = 0.0
    fape_clip_distance_inter: float = 30.0
    # Per-token atom set projected into every frame. The featurizer
    # always emits K=5 slots — [0..2] frame atoms (N/CA/C for protein,
    # C1'/C3'/C4' for nucleic), [3] rep (CB protein, C1' nucleic —
    # masked when duplicating a frame slot), [4] carbonyl O (protein
    # only). ``fape_atom_set`` is a runtime mask selecting which slots
    # contribute, so the same checkpoint can run any option without
    # re-featurising:
    #   * ``"CA"``       — K=1, frame centre only.
    #   * ``"NCAC+CB"``  — K=4 historical layout (N/CA/C/CB, no O); the
    #                     pre-2026-05-24 fapeS3 atom set.
    #   * ``"NCACO"``    — K=4 AF2 §4.2.7 backbone FAPE (N/CA/C/O).
    #   * ``"NCACO+CB"`` — K=5 AF2 backbone + CB (default). Supervises
    #                     C=O direction (strand H-bond ladder) on top
    #                     of the local-stereocentre chirality channel.
    #   * ``"ALL"``      — AF2 §4.2.7 full all-atom FAPE: every atom slot
    #                     of each token is projected into the backbone
    #                     frames (bypasses the K=5 subset). Gives the
    #                     symmetric terminal atoms (ASP OD1/OD2, ARG
    #                     NH1/NH2, PHE/TYR ring, LEU/VAL gem-dimethyls) —
    #                     which the K sets never probe — a local-frame
    #                     positional target, so they can no longer collapse
    #                     onto their pair midpoint under the symmetry
    #                     renaming (diffusion MSE / smooth_lddt leave that
    #                     collapse ~free). Cost is O(N²·M) with
    #                     M=MAX_ATOMS_PER_TOKEN=24 (~5× the K=5 path); use
    #                     ``fape_chunk_size`` to bound peak memory.
    fape_atom_set: str = "NCACO+CB"
    # Frame-axis chunk size for the FAPE loss (0 = process all N frames at
    # once). Only affects peak memory, not the result (the (i, j, atom)
    # reduction is a plain sum). Recommended for ``fape_atom_set="ALL"`` at
    # large crops, e.g. 64.
    fape_chunk_size: int = 0
    # Smooth LDDT weight. Set to None to disable entirely (AF3 Table 6:
    # smooth LDDT is turned off in fine-tune stage 1).
    alpha_smooth_lddt: float | None = 1.0

    # ---- Shape complementarity (OpenDDE arXiv:2607.03787 §A.2.1) ----------
    # SE(3)-invariant, orientation-aware, cross-chain protein-protein interface
    # loss: a Huber regression of the predicted surface-complementarity field
    # onto the field computed from the ground truth. Sits next to smooth-lDDT on
    # the diffusion output (the paper's ``L_geom = L_diff + λ_shape·L_shape + …``).
    #
    # Off by default, like ``alpha_fape`` / ``alpha_chirality`` — merging must
    # not change any existing run. **OpenDDE's own value is 3.0e-2**, which the
    # arm config sets explicitly; at that weight the term's gradient measured
    # 0.06-0.21x the smooth-lDDT gradient on rigidly perturbed complexes.
    #
    # Range of action is ~8 A of interface error (saturated by ~12 A, exactly
    # zero beyond ``shape_comp_interface_cutoff``): a refinement objective, not
    # a docking one. See ``export/shape-complementarity/``.
    alpha_shape_comp: float = 0.0
    # Relative weights of the three granularities (OpenDDE's ``{pair,token,
    # global}_weight``). The token term carries 0.69-0.88 of the weighted sum in
    # practice; the global term is a single degenerate scalar and is kept only
    # for fidelity with the published objective.
    shape_comp_pair_weight: float = 0.4
    shape_comp_token_weight: float = 1.0
    shape_comp_global_weight: float = 0.4
    # ``beta`` of ``F.smooth_l1_loss`` — NOT the ``delta`` of ``F.huber_loss``.
    # PyTorch defines ``huber(d) = d · smooth_l1(beta=d)``, so the huber spelling
    # would scale this term and its gradient by 0.01. Chosen at the near-native
    # residual scale so the term is L1-like above ~1 A of interface error.
    shape_comp_huber_delta: float = 0.01
    # Geometry — OpenDDE's shipped config values (which override the looser
    # defaults in their own function signature).
    shape_comp_density_sigma: float = 1.5
    shape_comp_interface_cutoff: float = 16.0
    shape_comp_gap_mean: float = 6.0
    shape_comp_gap_scale: float = 3.0
    shape_comp_clash_distance: float = 2.0
    shape_comp_clash_scale: float = 0.5
    shape_comp_pool_temperature: float = 16.0
    shape_comp_normal_strength_min: float = 1e-4
    # Token-axis chunk for the dense [N, A, 3] density step (0 = no chunking).
    # Only affects peak memory, not the result.
    shape_comp_chunk_size: int = 128
    # Optional noise gate: skip the term for diffusion samples whose sigma
    # exceeds this (at high sigma the denoised structure has no surface). None
    # = never gate. Left off until the sigma profile says otherwise.
    shape_comp_sigma_max: float | None = None

    # Single-representation magnitude regularization.  IntFold observed
    # activation explosions in later PairFormer transitions preceding the
    # gradient spikes from the diffusion loss backward; a mild penalty on
    # the single repr bounds magnitudes before they escape.
    #   ``single_reg_p``   — per-token norm (1 = Σ|s|/C, 2 = Σs²/C).
    #   ``alpha_single_reg`` — master switch + weight:
    #       * ``None`` → disabled entirely (no compute, no log).
    #       * ``0.0``  → log-only observation (computed under no_grad).
    #       * ``> 0``  → log + add ``alpha * L_p`` to total loss
    #                    (try ~1e-4 with p=2 to start regularizing).
    single_reg_p: int = 2
    alpha_single_reg: float | None = None
    # Same pattern for the pair representation (Pairformer transitions are
    # where IntFold observes the explosion *first*).
    pair_reg_p: int = 2
    alpha_pair_reg: float | None = None
    # Regularization metric form.
    #   ``"lp"`` (default) — per-token / per-pair L_p norm:
    #       loss term = α · L_p, logged key = ``single_l{p}`` / ``pair_l{p}``.
    #   ``"log_rms"`` — per-token / per-pair RMS = sqrt(mean(x²)):
    #       loss term = α · log(RMS + eps), logged key = ``single_rms`` / ``pair_rms``
    #       (the RMS value, not log(RMS); log only enters the loss).
    # Default ``"lp"`` preserves the existing behaviour bit-for-bit. The
    # ``"log_rms"`` form is wired so a follow-up arm can raise α without a
    # code change; with α=0 the metric is still recorded, no gradient.
    reg_form: Literal["lp", "log_rms"] = "lp"

    # Molecule-type loss reweighting (Protenix: DNA/RNA 5×, ligand 10×)
    nucleotide_loss_weight: float = 5.0
    ligand_loss_weight: float = 10.0

    # AF3 §3.7.1 eq. 2: weighted rigid alignment of GT onto denoised before
    # the diffusion MSE. Removes the global-rotation component of the error
    # signal (the augmented GT is only random-rotated; the denoiser's output
    # lives in a different frame when it's imperfect). Default True matches
    # AF3; disable for A/B against the DF3 pre-alignment baseline.
    use_rigid_align: bool = True

    # AF3 §4.2 resolves the whole permutation (chain + atom/ligand) ONCE from the
    # mini-rollout and shares it with every GT-based loss, including the noised GT
    # for the diffusion MSE. We follow that: chain + atom/ligand are resolved from
    # the mini-rollout and applied to the shared GT (all losses) and, by default,
    # to the MSE's augmented GT too. Set this True to instead resolve the MSE's
    # atom/ligand symmetry per noise sample (argmin vs each denoised sample —
    # a slightly tighter MSE, but it disagrees with the other losses' mini-rollout
    # labelling and can inflate grad-norm; see the shared-permutation notes).
    diffusion_mse_per_sample_symmetry: bool = False

    # Smooth LDDT thresholds. AlphaFold 3 supplement eq. 6 / Protenix
    # ``loss.py``: nucleotide-nucleotide pairs use 30 Å, everything else
    # uses 15 Å.
    lddt_cutoffs: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    lddt_inclusion_radius: float = 15.0
    lddt_nucleotide_inclusion_radius: float = 30.0
    lddt_min_distance: float = 1e-3  # minimum GT distance for LDDT pair inclusion

    # Confidence resolution gating
    min_resolution: float = 0.1
    max_resolution: float = 4.0

    # Fused loss (chunked linear + CE for pair-scale heads)
    use_fused_ce_loss: bool = False  # saves ~650 MB at N=768 (distogram + PAE + PDE)
    fused_ce_chunk_n: int = 32  # rows of N per chunk

    # Mixed precision
    dtype: str = "bf16"

    # DataLoader
    num_workers: int = 4
    eval_num_workers: int = 2
    prefetch_factor: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    # Tier-4 diagnostic: warn (with per-phase breakdown + chain counts) when a
    # single ``__getitem__`` exceeds this wall-clock budget, so periodic
    # step-time spikes can be attributed to specific records. 0 = disabled.
    slow_sample_log_threshold_s: float = 0.0
    # Tier-4b diagnostic: warn (PER RANK, no collective) when one rank's
    # microstep compute (forward+backward+step, incl. all_reduce wait) exceeds
    # this — so a DATA-TIED GPU-side straggler names its own record(s). Catches
    # what ``slow_sample_log_threshold_s`` can't (that times only the loader).
    # Set just above the normal step time (floor ~19s → use ~25). 0 = disabled.
    slow_step_log_threshold_s: float = 0.0

    # DDP
    find_unused_parameters: bool = False
    static_graph: bool = True
    bucket_cap_mb: int = 25
    gradient_as_bucket_view: bool = True
    # NCCL watchdog + ``collective_guard`` barrier deadline. Named sync
    # points in the trainer surface cross-rank desync as ``RuntimeError``
    # with the guard name, instead of an anonymous NCCL timeout.
    collective_timeout_s: float = 600.0

    # Run control
    max_steps: int = 200000
    seed: int = 42
    deterministic: bool = False  # enable torch.use_deterministic_algorithms + cudnn.deterministic
    checkpoint_interval: int = 1000
    eval_interval: int = 5000
    log_interval: int = 50  # in optimizer steps (after grad accumulation)
    log_microsteps: bool = False  # debug: log every microstep instead of every effective step
    # Single-track stability probe (per-block RMS / cos(s, u) / weight norms).
    # When enabled, attaches a :class:`SingleTrackProbe` to the PairFormer trunk
    # and writes one JSONL row per probed step to ``output_dir/single_track_probe.jsonl``.
    # Compute cost is small (one extra no_grad RMS/cos per block per probed step);
    # safe under autocast.
    single_track_probe: bool = False
    single_track_probe_interval: int = 50  # in optimizer steps
    # Always-on per-PairFormer-layer activation RMS sentinel. When enabled,
    # records single/pair RMS at the PF input plus each of the 48 layer
    # outputs every training forward and emits them as
    # ``act_rms/pf_in_single``, ``act_rms/pf_00_single``, ... at every
    # ``log_interval``. Cost: ~98 scalar reductions per microstep + one
    # packed all_reduce per log boundary; eval forwards are skipped to
    # keep train-window averages clean. See
    # :class:`deepfold3.training.activation_meter.ActivationRMSMeter`.
    log_activation_rms: bool = False
    # Multiplier on ``log_interval`` controlling how often the activation-RMS
    # meter is flushed to WandB. The meter accumulates every training
    # forward (cheap, rank-local); only the all-reduce + dict emission is
    # gated. ``act_rms_log_interval = 10`` with the default ``log_interval = 1``
    # gives one ρ snapshot per 10 effective steps — sufficient for the
    # slow-dynamic depth profile, and reduces ~960 keys × 10 in WandB cost.
    # ``<= 0`` skips the act_rms flush entirely (no emission); use
    # ``log_activation_rms = False`` to also skip attaching/running the meter.
    act_rms_log_interval: int = 1
    # Diagnostic-only accuracy metrics computed inside ``training_loss`` (under
    # ``no_grad``, not added to ``total``). Both are O(N²) and run every step, so
    # disable them to recover throughput when they aren't being watched.
    #   log_tm_true:           tm_true / itm_true from mini-rollout frames
    #                          (express_coordinates_in_frame → [1, N, N, 3]).
    #                          Auto-computed whenever ``alpha_pae > 0`` regardless
    #                          of this flag — the PAE loss already does the same
    #                          frame expansion, so tm_true is ~free in FT3; the
    #                          flag only suppresses it in stages 1–3 (no PAE).
    #   log_distogram_entropy: H(p_θ) of the distogram head ([·, N, N, 64] softmax).
    log_tm_true: bool = True
    log_distogram_entropy: bool = True
    uniform_recycle: bool = (
        True  # all DDP ranks use the same num_recycles per microstep (avoids straggler at no_grad warmups)
    )
    batch_size_per_rank: int = 1  # number of samples per rank per microstep (effective = world × batch × accumulate)
    epoch_size: int = (
        100_000  # sampler reseed-chunk size (infinite sampler: no epoch boundary; epoch is a derived log counter)
    )
    eval_before: bool = False  # run one eval pass before the training loop starts (also fires on resume)
    max_checkpoints: int = 0  # keep N most recent checkpoints (0 = keep all)
    # Node-local directory to write the checkpoint to first; a background
    # thread then publishes it to ``output_dir``. Takes the multi-GB write to
    # shared storage off the training thread, where it stalls every other rank
    # at the next collective. "" = write straight to ``output_dir`` (blocking).
    # The path must be node-local disk — a memory-backed emptyDir (/dev/shm)
    # would trade the stall for exactly the cgroup pressure this avoids.
    checkpoint_staging_dir: str = ""
    checkpoint_every_hours: float = 1.0  # time-based checkpoint interval (0 = disabled)
    auto_resume: bool = False  # auto-detect latest checkpoint in output_dir

    # Multi-stage training (use --override or JSON config to set these per stage)
    train_confidence_only: bool = False  # freeze trunk, only train confidence head
    finetune_scheduler: str = "af3"  # "af3", "cosine", or "wsds"
    cosine_min_lr_ratio: float = 0.1  # min LR = lr * ratio (for cosine scheduler)
    cosine_max_steps: int = 20000  # total steps for cosine annealing

    # Warmup-Stable-Decay-Simplified (WSD-S) — continual-pretraining schedule
    # with multiple [decay_start, decay_end] checkpoints. After warmup the LR
    # holds at ``lr`` (stable phase), then within each [start, end) window
    # decays inverse-proportionally from ``lr`` → ``lr * wsds_min_lr_ratio``.
    # After ``end`` the LR jumps back to ``lr`` (no rollback). See
    # https://arxiv.org/abs/2410.05192 (Hägele et al., 2024) for the
    # river-valley justification and the 8–12% decay-fraction guidance.
    #
    # Example ``configs/initial.yaml``:
    #   wsds_decay_windows: [[9000, 10000], [18000, 20000]]
    #   wsds_min_lr_ratio: 0.01
    wsds_decay_windows: list[list[int]] = field(default_factory=list)
    wsds_min_lr_ratio: float = 0.01

    # Checkpoint loading
    load_strict: bool = True  # strict state_dict loading
    skip_load_optimizer: bool = False
    skip_load_scheduler: bool = False
    skip_load_step: bool = False

    # Experiment tracking
    wandb_enabled: bool = False  # explicit on/off; takes precedence over wandb_project
    wandb_project: str = ""
    wandb_entity: str = ""  # WandB team/org entity; empty → personal workspace
    # Optional explicit run name. ``None`` (or empty string) → derive from
    # ``output_dir`` and append a YYYYMMDD-HHMMSS timestamp so re-runs are
    # never accidentally collapsed under one wandb run.
    wandb_run_name: str | None = None

    # Evaluation
    eval_cutoff_date: str = ""  # ISO date: records released AFTER this → eval set (e.g. "2021-09-30")
    eval_max_date: str = ""  # ISO date: records released AFTER this are excluded from eval
    eval_samples: int = 128  # max records per eval pass (0 = use all; cluster-weighted when clusters available)
    eval_datasets: list[DatasetConfig] = field(default_factory=list)  # YAML eval_datasets section

    # Paths (set at runtime)
    output_dir: str = "output"
    data_dir: str = ""
    params_path: str = ""

    # Templates
    mmcif_dir: str = ""  # PDB mmCIF mirror path; non-empty enables template loading
    # Per-chain template cache (memory LRU + hierarchical sharded disk). When set,
    # training-time template extraction reads a cached per-chain slice instead of
    # re-parsing the full template mmCIF per hit (the dominant dataloader-stall
    # cost). Pre-build with ``deepfold3 build-pdb-cache``. ``mmcif_dir`` is the
    # cold-parse fallback for chains absent from the cache. Empty = disabled.
    pdb_chain_cache_dir: str = ""
    pdb_chain_cache_mem: int = 512  # max chains held in the per-worker memory LRU

    # Unified CCD cache (atoms + ref_pos + automorphisms) on the shared PVC.
    # Non-empty enables ligand atom-permutation loss via ``load_ligand_automorphisms``.
    ccd_cache_path: str = ""

    # Distillation dataset (optional, legacy — prefer extra_datasets)
    distillation_data_dir: str = ""  # empty = disabled
    distillation_weight: float = 0.25  # sampling weight relative to PDB (PDB gets 1.0)

    # Multi-dataset training: list of additional datasets beyond --data-dir.
    # Can be defined in YAML/JSON config under ``datasets`` top-level key
    # or via CLI ``--dataset name:path:weight:strategy:distill:format``.
    extra_datasets: list[DatasetConfig] = field(default_factory=list)

    def __setstate__(self, state):
        # Migrate train configs pickled under an older schema. See
        # ``_backfill_dataclass``.
        self.__dict__.update(state)
        _backfill_dataclass(self, TrainingConfig())

    def __post_init__(self):
        """Validate configuration values."""
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.start_decay_after < self.warmup_steps:
            raise ValueError(
                f"start_decay_after ({self.start_decay_after}) must be >= warmup_steps ({self.warmup_steps})"
            )
        if self.decay_every <= 0:
            raise ValueError(f"decay_every must be > 0, got {self.decay_every}")
        if self.dtype not in _COMPUTE_DTYPE_CHOICES:
            raise ValueError(f"dtype must be one of {_COMPUTE_DTYPE_CHOICES}, got '{self.dtype}'")
        if self.accumulate_grad_batches < 1:
            raise ValueError(f"accumulate_grad_batches must be >= 1, got {self.accumulate_grad_batches}")
        if self.max_recycles < 0:
            raise ValueError(f"max_recycles must be >= 0, got {self.max_recycles}")
        # Records with no resolution metadata are emitted as
        # ``const.RESOLUTION_UNKNOWN`` (0.0) and rely on this window excluding
        # them. A min_resolution of 0.0 would silently promote every such
        # record into the confidence loss instead.
        if self.min_resolution <= const.RESOLUTION_UNKNOWN:
            raise ValueError(
                f"min_resolution must be > {const.RESOLUTION_UNKNOWN} (the 'unknown resolution' "
                f"sentinel), got {self.min_resolution}"
            )

    @property
    def torch_dtype(self):
        """Returns the ``torch.dtype`` corresponding to ``self.dtype``."""
        import torch

        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype]


_COMPUTE_DTYPE_CHOICES: tuple[str, ...] = ("bf16", "fp16", "fp32")


@dataclass
class DistogramConfig:
    num_bins: int = 64
    first_break: float = 2.3125
    last_break: float = 21.6875
    use_half_logits_ln: bool = False
    # ``half_logits`` weight init: stddev = α / sqrt(fan_in). Default α=0 is the
    # AF3/OF3 stock zeros init — the distogram starts as the uniform 1/num_bins
    # distribution. Set α>0 (e.g. 0.05) for a small-normal init: the softmax
    # stays close to uniform (CE ≈ log(num_bins)) but the weight is non-zero, so
    # the distogram gradient reaches ``pair`` from step 0.
    half_logits_init_alpha: float = 0.0


@dataclass
class PrecisionConfig:
    """Mixed-precision policy — region guards + non-compute dtype axes.

    Two flavours of knob live here:

    * ``fp32_*`` region flags: disable autocast and upcast that specific
      region's inputs to fp32. Use to guard numerically sensitive
      projections (especially final logit heads) against bf16 overflow.
    * ``{grad_acc,reduce,loss}_dtype`` axes: the DS-inspired policy knobs
      for gradient accumulation / DDP all-reduce / loss computation.
      Compute dtype remains governed by ``TrainingConfig.dtype`` (the
      autocast context); these axes compose on top.

    Defaults preserve the pre-refactor behaviour bit-for-bit: grads
    accumulate and all-reduce in fp32, losses are fp32. ``reduce_dtype
    = "bf16"`` is an opt-in speedup (registers DDP's ``bf16_compress``
    comm hook and halves all-reduce bandwidth at a small numeric cost).
    """

    # Compute ``training_loss`` in fp32.
    fp32_loss: bool = True
    # Run ConfidenceHead final LN+Linear blocks and the distance embedding in fp32.
    fp32_confidence_head: bool = True
    # Run DistogramHead.half_logits in fp32. Off by default; flip on if the
    # distogram loss shows bursts beyond its theoretical max (ln(num_bins)).
    fp32_distogram_head: bool = False
    # Run the trunk-input atom conditioning encoder
    # (:class:`EvoformerConditioningEncoder`) in fp32. The 3D coordinate
    # projection ``embed_ref_pos`` (Linear(3→128)) and the offset/distance
    # projections handle features with high dynamic range that bf16
    # accumulates noisily — matching Protenix
    # (``precision=torch.float32`` per-Linear on ref_pos / d / r) and OF3
    # (``with torch.amp.autocast(...dtype=fp32)`` wrap on
    # ``input_embedders.py:127``). Default on; flip off only to A/B the
    # numerical effect of the bf16 path.
    fp32_evo_cond_encoder: bool = True
    # Run the diffusion sampler (inference) in fp32. Mirrors
    # ``DiffusionConfig.skip_amp_sample`` — ``ModelConfig.__post_init__``
    # keeps the two in sync.
    fp32_diffusion_sample: bool = False
    # Run the training-time diffusion rollout in fp32. Mirrors
    # ``DiffusionConfig.skip_amp_sample_training``.
    fp32_diffusion_sample_training: bool = False

    # Gradient accumulation dtype. DeepSpeed-inspired contract — master
    # weights stay fp32 (AdamW requirement), grads accumulate across
    # microsteps in this dtype. ``bf16`` is incompatible with
    # ``accumulate_grad_batches > 1`` (bf16 rounding swallows small
    # contributions); asserted at trainer startup.
    grad_acc_dtype: Literal["bf16", "fp32"] = "fp32"
    # DDP all-reduce dtype. ``fp32`` (default) uses PyTorch's stock
    # allreduce. ``bf16`` registers ``bf16_compress_hook``, halving
    # inter-rank bandwidth; the cost is one-extra-cast per bucket and
    # mild numerical drift vs. fp32 allreduce.
    reduce_dtype: Literal["bf16", "fp32"] = "fp32"
    # Loss-computation dtype. Always ``fp32`` today; kept as a knob to
    # make the policy explicit and to reserve the axis for future bf16-
    # loss experiments.
    loss_dtype: Literal["fp32"] = "fp32"

    def __post_init__(self) -> None:
        if self.grad_acc_dtype not in ("bf16", "fp32"):
            raise ValueError(f"grad_acc_dtype must be 'bf16' or 'fp32', got {self.grad_acc_dtype!r}")
        if self.reduce_dtype not in ("bf16", "fp32"):
            raise ValueError(f"reduce_dtype must be 'bf16' or 'fp32', got {self.reduce_dtype!r}")
        if self.loss_dtype != "fp32":
            raise ValueError(f"loss_dtype must be 'fp32' (other values reserved), got {self.loss_dtype!r}")

    @classmethod
    def from_shorthand(cls, shorthand: str, **overrides) -> PrecisionConfig:
        """Build a policy from a single string (sugar for common presets).

        ``"bf16-mixed"`` → grad_acc=fp32, reduce=bf16, loss=fp32 (maximum bandwidth savings).
        ``"bf16-safe"``  → grad_acc=fp32, reduce=fp32, loss=fp32 (current DF3 default).
        ``"fp16-mixed"`` → grad_acc=fp32, reduce=bf16, loss=fp32 (GradScaler required).
        ``"fp32"``       → grad_acc=fp32, reduce=fp32, loss=fp32.

        The shorthand only sets the non-compute axes; compute dtype is
        governed by ``TrainingConfig.dtype``. ``overrides`` override any
        axis post-expansion (including the region-guard flags).
        """
        presets: dict[str, dict[str, str]] = {
            "bf16-mixed": {"grad_acc_dtype": "fp32", "reduce_dtype": "bf16", "loss_dtype": "fp32"},
            "bf16-safe": {"grad_acc_dtype": "fp32", "reduce_dtype": "fp32", "loss_dtype": "fp32"},
            "fp16-mixed": {"grad_acc_dtype": "fp32", "reduce_dtype": "bf16", "loss_dtype": "fp32"},
            "fp32": {"grad_acc_dtype": "fp32", "reduce_dtype": "fp32", "loss_dtype": "fp32"},
        }
        if shorthand not in presets:
            raise ValueError(f"unknown precision shorthand {shorthand!r}; choices: {sorted(presets)}")
        return cls(**{**presets[shorthand], **overrides})


@dataclass
class DataConfig:
    max_templates: int = 4
    # Pad the template axis to ``max_templates`` instead of emitting the
    # sampled count. AF3 draws ``k ~ min(Uniform[0, n], max_templates)`` per
    # chain, so T varies 0..max_templates per sample and the O(N²) template
    # stack costs a different amount on every rank — under DDP the step clock
    # is then set by whichever rank drew the most templates (a rotating
    # forward-straggler). Padding + a ``template_mask`` makes the per-step
    # template cost constant with NO change in model output (pad slots are
    # zeroed and excluded from the average denominator).
    #
    # This buys determinism, not throughput: at DDP world sizes the fleet
    # already pays max-T on essentially every step, so padding raises each
    # rank's own work to what it was already waiting for. Enable it to remove
    # the last data-dependent source of per-rank compute variance (e.g. when
    # attributing step-time spikes); leave it off to save the GPU-seconds.
    pad_templates: bool = False
    max_neighborhood: int = 40
    # Cropping mixture weights (Protenix-style:
    # ContiguousCropping / SpatialCropping / SpatialInterfaceCropping).
    contiguous_crop_weight: float = 0.2
    spatial_crop_weight: float = 0.4
    interface_crop_weight: float = 0.4
    # Distance threshold (Å) for "interface" tokens used by the
    # interface-seeded spatial cropper.
    interface_distance: float = 15.0
    # AlphaFold 3 §2.4 template release-date guard: a template's release
    # date must precede the query example's release by at least this many
    # days (default 60 for PDB training data).
    template_cutoff_days: int = 60
    # Optional global hard cap on template release dates (ISO YYYY-MM-DD).
    # AF3 uses 2018-04-30 for distillation and the training cutoff for
    # inference; leave None to skip the global cap.
    max_template_date: str | None = None
    # Global template cutoff (ISO YYYY-MM-DD) used as the per-record bound when
    # a record has no ``released`` date (so the per-record ``released -
    # cutoff_days`` guard cannot be computed). Set this to the training split's
    # date boundary. If neither this nor ``max_template_date`` is set and a
    # record lacks a release date, template loading is fail-closed (no
    # templates) rather than leaking future structures.
    default_template_cutoff_date: str | None = None
    # Prefer per-sequence monomer templates over per-record multimer
    # templates when both directories are present.
    prefer_monomer_templates: bool = True
    # Maximum polymer chains to keep per sample at training time
    # (AF3 §2.5.4 / Table 6: 20 for initial + fine-tune 1/2, 50 for
    # fine-tune 3).  When a structure exceeds this limit, the same
    # greedy interface-maximising selection from prepare-time is applied
    # at sampling time.  Set 0 to disable.
    max_chains: int = 20
    # Drop whole records whose total chain count (polymer + ligand) exceeds
    # this, BEFORE the dataset is built — they are never sampled or loaded.
    # Giant assemblies (e.g. 5iwa: 228 chains) cost 5-50× a normal record to
    # featurise (per-chain MSA load + template parse + MSA profile) and cause
    # periodic dataloader stalls. 0 = disabled (keep all records).
    max_total_chains: int = 0
    # Stochastic atomization (training-only regularizer). With probability
    # ``atomize_prob`` each *eligible* standard polymer residue is expanded from
    # a single per-residue token into one token per heavy atom (the same layout
    # non-standard/modified residues and ligands already use). Inference NEVER
    # atomizes. This augments the atomized-token code path (which IS exercised
    # at inference for modified residues / bonded ligands / glycans) and acts as
    # a mild token-boundary regularizer. Applied BEFORE cropping so the token
    # budget accounts for the expansion. 0.0 = disabled (default; enable from a
    # fine-tune stage). Train split only unless ``atomize_eval`` says otherwise.
    # See ``data/tokenize/atomize.py``.
    atomize_prob: float = 0.0
    # Which residues are eligible for atomization:
    #   "random"    — every standard polymer residue.
    #   "interface" — residues within ``interface_distance`` of another chain.
    #   "region"    — residues whose "region" token-mask channel is ≥ 1 (e.g.
    #                 CDR labels for Ag-Ab); falls back to none if no region mask.
    atomize_source: str = "random"
    # Hard cap on the number of residues atomized per sample (0 = no cap).
    # Bounds the token-budget blow-up on large interfaces / many-CDR inputs.
    atomize_max_residues: int = 0
    # Whether the eval split atomizes too. Default False: eval mirrors what is
    # served (inference never atomizes), the same way dropout runs in train mode
    # only, and the per-sample Bernoulli draw would otherwise re-roll every
    # epoch and add noise to the metric. Set True to score eval on the *training*
    # distribution instead — e.g. to read the eval loss as a like-for-like
    # continuation of the training curve while atomization is on. Inherits
    # ``atomize_prob`` / ``atomize_source`` / ``atomize_max_residues``; a no-op
    # at ``atomize_prob = 0``.
    atomize_eval: bool = False


@dataclass
class ModelConfig:
    evoformer: EvoformerConfig = field(default_factory=EvoformerConfig)
    pairformer: PairFormerConfig = field(default_factory=PairFormerConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    template: TemplateEmbeddingConfig = field(default_factory=TemplateEmbeddingConfig)
    distogram: DistogramConfig = field(default_factory=DistogramConfig)
    masked_msa: MaskedMSAConfig = field(default_factory=MaskedMSAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    num_recycles: int = 10
    # MSA rows fed to the Evoformer per recycle. Defaults to AF3's
    # Nmsa = 16384 (the same cap used by the featurizer), so per-recycle
    # subsampling is a no-op when the featurizer already capped.
    msa_crop_size: int = 16_384

    def __setstate__(self, state):
        # Migrate configs pickled under an older schema (e.g. checkpoints saved
        # before ``masked_msa`` was added). See ``_backfill_dataclass``.
        self.__dict__.update(state)
        _backfill_dataclass(self, ModelConfig())

    def __post_init__(self):
        # Mirror the legacy ``DiffusionConfig.skip_amp_sample*`` flags onto the
        # unified ``PrecisionConfig`` fields so setting either side has the
        # same effect regardless of entry point.
        def _mirror(pre_field: str, diff_field: str) -> None:
            val = bool(getattr(self.precision, pre_field) or getattr(self.diffusion, diff_field))
            setattr(self.precision, pre_field, val)
            setattr(self.diffusion, diff_field, val)

        _mirror("fp32_diffusion_sample", "skip_amp_sample")
        _mirror("fp32_diffusion_sample_training", "skip_amp_sample_training")

        # A trunk with no single track still *has* a single output — it is just
        # ``single_activations(target_feat)``, a pointwise map of the features
        # the diffusion conditioning already receives. Concatenating it there is
        # legal (it is what the reference implementation of this ablation does)
        # but carries no information, so say so once at construction rather than
        # leaving a 384-wide duplicate to be discovered in a probe later.
        if not self.pairformer.use_single_track and self.diffusion.conditioning_use_trunk_single:
            import logging

            logging.getLogger(__name__).warning(
                "pairformer.use_single_track=False with diffusion.conditioning_use_trunk_single=True: "
                "the trunk single output is a pointwise function of target_feat, so the diffusion "
                "conditioning is concatenating a duplicate of a channel it already has. Set "
                "diffusion.conditioning_use_trunk_single=false unless the duplicate is deliberate."
            )


# Key under which ``ModelConfig`` JSON is embedded in ``safetensors`` metadata.
# safetensors metadata is ``dict[str, str]``; one ASCII-JSON-serialised entry
# is enough to round-trip the full nested ``ModelConfig`` for inference.
SAFETENSORS_MODEL_CFG_KEY = "model_cfg_json"


def model_cfg_to_metadata(cfg: ModelConfig) -> dict[str, str]:
    """Serialize ``ModelConfig`` to safetensors-compatible metadata.

    Returns a ``dict[str, str]`` suitable for the ``metadata=`` kwarg of
    :func:`safetensors.torch.save_file`. Round-trippable via
    :func:`model_cfg_from_metadata`.
    """
    import json
    from dataclasses import asdict

    return {SAFETENSORS_MODEL_CFG_KEY: json.dumps(asdict(cfg))}


def model_cfg_from_metadata(metadata: dict[str, str] | None) -> ModelConfig | None:
    """Reconstruct ``ModelConfig`` from safetensors metadata, or ``None``.

    Returns ``None`` if ``metadata`` is missing or lacks the embedded key
    (e.g. converted weights produced before this metadata convention was
    introduced) — the caller falls back to ``ModelConfig()`` defaults.

    Unknown fields in the JSON are silently dropped: a checkpoint produced
    by a future codebase that added new fields stays loadable, with the
    extra fields ignored. Missing fields fall back to the dataclass
    defaults at the current revision.
    """
    if not metadata:
        return None
    raw = metadata.get(SAFETENSORS_MODEL_CFG_KEY)
    if raw is None:
        return None

    import json
    import typing
    from dataclasses import fields, is_dataclass

    data = json.loads(raw)

    def _coerce(v, t):
        # JSON has no tuple type — restore tuple-typed fields from list.
        if isinstance(v, list) and typing.get_origin(t) is tuple:
            return tuple(v)
        if isinstance(v, dict) and is_dataclass(t):
            return _rec(t, v)
        return v

    def _rec(cls, d):
        if not is_dataclass(cls) or not isinstance(d, dict):
            return d
        hints = typing.get_type_hints(cls)
        valid = {f.name for f in fields(cls)}
        kwargs = {}
        for k, v in d.items():
            if k not in valid:
                continue
            kwargs[k] = _coerce(v, hints.get(k))
        return cls(**kwargs)

    return _rec(ModelConfig, data)
