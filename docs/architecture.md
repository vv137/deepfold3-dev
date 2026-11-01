# Architecture Deep-Dive

DeepFold3 is a diffusion-based generative model for biomolecular structure prediction, architecturally equivalent to AlphaFold 3. This document covers the forward pass, module internals, and key design decisions.

## Forward Pass Overview

```text
Features dict (featurizer output)
  │
  ├─ create_relative_encoding()        -> rel_feat    [N, N, 139]
  ├─ one_hot(res_type) + profile       -> raw_feat    [N,  63]
  │    + deletion_mean
  ├─ EvoformerConditioningEncoder()     -> evo_cond    [N, 384]
  │    (ref atoms -> atom transformer
  │     -> masked mean pool to tokens)
  │
  ├─ concat [raw_feat, evo_cond]       -> target_feat [N, 447]
  ├─ MSA feat (one_hot + has_deletion  -> msa_feat    [S, N,  34]
  │    + deletion_value)
  │
  ╰─ Evoformer trunk (10 recycles)
       ├─ Input projections
       │    single = proj(target_feat)               [N, 384]
       │    pair   = outer_sum(left(tf), right(tf)) + rel_feat   [N, N, 128]
       │    msa    = proj(msa_feat) + proj(tf)        [S, N,  64]
       ├─ + recycled prev_single, prev_pair (.detach())
       ├─ + bond embedding (optional)
       ├─ + template embedding (optional, T templates)
       │    T x SingleTemplateEmbedding -> avg -> ReLU -> proj  [N, N, 128]
       ├─ MSA stack:       4 x EvoformerBlock
       ╰─ PairFormer stack: 48 x PairFormerBlock
              -> single [N, 384]  +  pair [N, N, 128]

  DiffusionHead.sample()
       200-step denoising with random augmentation
       -> atom_positions [S, N, M, 3]     (S=num_samples, M=24)

  ConfidenceHead (per sample)
       4-layer PairFormer on embeddings + dgram from predicted positions
       -> pLDDT [S, N, M*50],  PAE [S, N, N, 64]
         PDE   [S, N, N, 64], exp_resolved [S, N, M*2]

  DistogramHead
       half_logits(pair) + transpose  ->  [N, N, 64]
```

Key constants (`model/model.py`):

- `_RAW_TARGET_FEAT_DIM = 63` (31 one-hot + 31 profile + 1 deletion_mean)
- `_AATYPE_NUM_CLASSES = 31`
- `_MSA_FEAT_DIM = 34` (32 one-hot + 1 has_deletion + 1 deletion_value)
- `_MSA_NUM_CLASSES = 32`

## Module-by-Module Breakdown

### `primitives.py` -- Building Blocks

| Class                | Description                                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------------- |
| `LinearNoBias`       | `nn.Linear(bias=False)` -- matches AF3's `hm.Linear` default (`use_bias=False`)                       |
| `LayerNorm`          | `nn.LayerNorm(elementwise_affine=True)` -- always has learnable scale+offset                          |
| `AdaLayerNorm`       | Adaptive LN: base LN (no learnable params) + conditioning -> sigmoid(scale) \* x + bias               |
| `SwiGLU`             | `swish(a) * b` where `[a, b] = linear(x).chunk(2)`                                                    |
| `TransitionBlock`    | LN -> SwiGLU feed-forward; `linear1: [c -> 2*c*factor]`, `linear2: [c*factor -> c]`; default factor=4 |
| `AdaTransitionBlock` | AdaLN-zero conditioned FF for diffusion; `zero_cond` gate init at bias=-2; default factor=2           |

### `config.py` -- Configuration Hierarchy

`ModelConfig` is the top-level config, containing nested dataclasses:

| Config                       | Key Defaults                                                                                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ModelConfig`                | `num_recycles=10`                                                                                                                                                  |
| `EvoformerConfig`            | `num_layer=4`, `seq_channel=384`, `pair_channel=128`, `msa_channel=64`, `num_msa=1024`, `max_relative_idx=32`, `max_relative_chain=2`                              |
| `PairFormerConfig`           | `num_layer=48`, `pair_channel=128`, `seq_channel=384`, `single_attention_num_head=16`                                                                              |
| `AtomCrossAttConfig`         | `per_token_channels=768`, `per_atom_channels=128`, `per_atom_pair_channels=16`, `queries_subset_size=32`, `keys_subset_size=128`                                   |
| `AtomTransformerConfig`      | `num_blocks=3`, `num_head=4`, `key_dim=128`, `value_dim=128`                                                                                                       |
| `DiffusionTransformerConfig` | `num_blocks=24`, `num_head=16`, `super_block_size=4`, `num_intermediate_factor=2`                                                                                  |
| `DiffusionConfig`            | `steps=200`, `num_samples=5`, `sigma_data=16.0`, `smin=0.0004`, `smax=160.0`, `noise_p=7.0`, `gamma_0=0.8`, `gamma_min=1.0`, `noise_scale=1.003`, `step_scale=1.5` |
| `TemplateEmbeddingConfig`    | `num_channels=64`, `pair_channel=128`, `dgram_num_bins=39`, `dgram_min_bin=3.25`, `dgram_max_bin=50.75`, `aatype_num_classes=31`                                   |
| `TemplatePairFormerConfig`   | `num_layer=2`, `num_channels=64`, `num_heads=4`, `num_intermediate_factor=2`                                                                                       |
| `ConfidenceConfig`           | `num_layer=4`, `num_plddt_bins=50`, `num_distogram_bins=64`, `max_error_bin=31.0`, `max_atoms_per_token=24`                                                        |
| `OptimConfig`                | Fused kernel toggles; `FUSED_STATE_DICT_MARKERS` (tuple of key substrings for derived params)                                                                      |
| `DistogramConfig`            | `num_bins=64`, `first_break=2.3125`, `last_break=21.6875`                                                                                                          |

### `embedders.py` -- Relative Encoding

`create_relative_encoding(token_index, residue_index, asym_id, entity_id, sym_id)` -> `[N, N, 139]`

Components:

- **rel_pos** (66 dims): residue_index diff, clamped to `+/-max_relative_idx=32`, one-hot + extra bin for cross-chain
- **rel_token** (66 dims): token_index diff within same residue, same encoding
- **entity_same** (1 dim): `entity_id[i] == entity_id[j]` as float
- **rel_chain** (6 dims): sym_id diff, clamped to `+/-max_relative_chain=2`, one-hot + extra bin for cross-entity

### `attention.py` -- Three Attention Variants

**`GridSelfAttention`** -- Row/column attention on pair representation `[N, N, C]`

- `transpose=False` for row-wise, `transpose=True` for column-wise
- 4 heads, `head_dim = max(c // num_heads, 16)`
- Pair bias: `pair -> LinearNoBias -> [H, N, N]` broadcast over all rows (`pb.unsqueeze(0)`)
- Gated output: `out * sigmoid(to_g(x))`

**`MSAAttention`** -- Row-wise MSA attention `[S, N, msa_channel]`

- 8 heads; **no Q/K projections** -- attention weights derived from pair activations
- `pair_logits(pair_norm(pair))` -> shared attention weights `[H, N, N]` across all S sequences
- `einsum("bhqk,bskhc->bsqhc", w, v)` applies shared weights to per-sequence values

**`SingleAttention`** -- Self-attention on single representation `[N, 384]`

- 16 heads, `key_dim=384`, `value_dim=384`
- Receives **pre-computed** `pair_logits [H, N, N]` from `PairFormerBlock.single_to_pair`
- Note: `to_q` uses `nn.Linear` (with bias); all other projections use `LinearNoBias`

### `ops.py` -- Triangle Operations

**`TriangleMul`** -- Triangle multiplicative update

- Outgoing: `equation="ikc,jkc->ijc"` (default); Incoming: `equation="kic,kjc->ijc"`
- LayerNorm -> project to 2xchannels -> sigmoid gate -> einsum -> LayerNorm -> project back + sigmoid gate

**MHR-TriMul** (off by default; `num_heads=0` is AF3). With `num_heads=H`, the
contraction is unchanged -- `einsum("ikhd,jkhd->ijhd")` is bitwise equal to
`einsum("ikc,jkc->ijc")` under `c=(h,d)`, since only `k` is contracted -- and
two steps are inserted between the einsum and `out_proj`:

- `head_norm="head_rms"` / `"head_ln"`: normalize over `d` within each head
  instead of over all `c_z`. The affine reuses `norm_center.weight/bias` viewed
  as `[H, d]`, so `state_dict` keys are unchanged.
- `router=True`: `g = H * softmax_h(W_r(LN(z)_ij))`, applied per head before
  `out_proj`. Zero-init gives `g == 1`, i.e. exact AF3 behaviour at step 0.
  This makes `out_proj` input-dependent over its column blocks, which the
  output gate (applied *after* `out_proj`) cannot express.

`router/entropy` is logged with the activation meter; `1.0` means the router is
still uniform, i.e. inert. See `export/mhr-trimul/README.md`.

**`OuterProductMean`** -- MSA-to-pair information flow

- MSA `[S, N, c_msa]` -> two projections `[S, N, 32]` -> outer product -> mean over S -> pair `[N, N, c_pair]`

### `evoformer.py` -- Trunk

**`EvoformerBlock`** (x4): MSA column attention -> MSA row attention (pair-biased) -> MSA transition -> OuterProductMean -> pair TriangleMul (out+in) -> pair GridSelfAttention (row+col) -> pair transition. With `OptimConfig.use_fused_msa_transition_inference` and grad disabled, the MSA transition runs as one in-place kernel (`triton/msa_transition.py`); weights and state-dict keys stay on the same `msa_transition` block.

**`PairFormerBlock`** (x48): pair TriangleMul (out+in) -> pair GridSelfAttention (row+col) -> pair transition -> SingleAttention (pair-biased) -> single transition (the last two are dropped by `use_single_track=False`)

`use_triangle_attention=False` (on either `EvoformerConfig` or
`PairFormerConfig`) drops the two `GridSelfAttention` branches from every block
of that stack; the branches are still metered, as exactly zero, so per-branch
`rho_l` stays comparable against a run that has them. Triangle attention is the
only pair operator whose selection over the intermediate node `k` is
conditioned on the target pair `(i, j)` -- MHR-TriMul's router selects a
relation subspace, not an intermediate node, so it does not replace it.
Template and confidence-head PairFormers are unaffected.

`PairFormerConfig.use_single_track=False` drops the per-token track from every
trunk block -- `SingleAttention`, its `single_to_pair` PairBias projection, and
the single transition -- leaving a block that updates pair only and returns
`single` untouched. `single` is a pure *sink* in the block (it reads pair
through the PairBias logits and never writes back), so the pair track is
bit-identical with and without it; the deleted branches are metered as zero,
same convention as above. At `c_z=256, c_s=384, L=48` this track is 120.6M
parameters, 28% of the model.

Two consequences follow, and both are wired to the flag rather than left
implicit:

- The trunk single output becomes `single_activations(target_feat)` -- a
  pointwise map with no token mixing and no dependence on MSA, templates, or
  the recycled pair. The recycle projection (`prev_single_embedding`) is
  therefore not built at all: `s = s_init + P(LN(s_prev))` would be a
  fixed-point iteration on `s_init`. The projection itself survives, and its
  live consumer is the diffusion **atom** encoder
  (`AtomEncoder.compute_static(trunk_single=...)`), which receives no
  `target_feat` of its own -- so the trunk single is its only token-level
  identity channel, and is what keeps `single_activations` trained.
- `DiffusionConfig.conditioning_use_trunk_single=False` drops the trunk half of
  the diffusion conditioning's `cat([s_trunk, target_feat])`, which under this
  arm is a duplicate of a channel the conditioning already has -- and one that
  moves the shared LayerNorm's statistics, since the LN spans the concatenation.
  Leaving it `True` is legal and warns at `ModelConfig.__post_init__`.

Scope is the trunk. The confidence head keeps its single track (it rebuilds a
per-token summary from trunk pair + the predicted distogram) and reads its own
config class. See `configs/df3-stage0-r19.yaml` and
`export/mhr-trimul/README.md`.

Optional Residual Output Scaling (ROS) on every block branch: when `PairFormerConfig.residual_output_scaling=True` (and equivalently `ConfidenceConfig.residual_output_scaling`), the block multiplies every branch's output by `s_ros = 1/sqrt(layer_idx + 1)` before adding to the residual stream. The depth-uniform variant `uniform_residual_scale` uses `1/sqrt(num_layer)` instead. The single-track stability ablation adds branch-specific multipliers `single_transition_residual_gain` and `pair_transition_residual_gain` (each composes with `s_ros`); see `docs/single-track-stability-experiments.md` §11. Output projections retain zero-init throughout.

Because these residual scaling factors only affect forward-pass scalars (no parameter shape change), `load_state_dict(strict=False)` will silently accept a checkpoint trained with one setting into a model built with the other and produce wrong outputs. Both `.pt` and `.safetensors` writers therefore embed the architectural `ModelConfig` (pickled blob and JSON metadata respectively); inference reconstructs the dataclass at load time. See `docs/training.md` § _Configuration persistence_.

**`Evoformer`** input projections:

- `target_feat [N, 447]` -> single `[N, 384]`, pair `[N, N, 128]` (outer sum of left/right projections)
- `msa_feat [S, N, 34]` -> msa `[S, N, 64]` (+ broadcast target_feat projection)
- `rel_feat [N, N, 139]` -> pair (additive)
- Recycled `prev_pair` and `prev_single` added through LayerNorm + projection (the `prev_single` path is absent when `use_single_track=False`)
- Template embedding (optional): `pair += template_embedding(pair, template_feats)` -- after bonds, before MSA stack

### `template_embedding.py` -- Template Embedding

Processes structural templates (known PDB structures similar to the query) into pair-space embeddings.

**`TemplateEmbedding`** -- Top-level module:

- Loops over T templates, each processed by `SingleTemplateEmbedding`
- Averages over T, applies ReLU, projects `LinearNoBias(64, 128)` -> `[N, N, 128]`
- Haiku scope: `diffuser/evoformer/template_embedding/`

**`SingleTemplateEmbedding`** -- Processes one template:

- 9 input projections (all `LinearNoBias -> 64`, summed):
  - Distance histogram `[N, N, 39]`, pseudo-beta mask `[N, N, 1]`
  - Aatype one-hot (row and column) `[N, 31]`
  - Unit vectors (x, y, z) from backbone frames `[N, N, 1]`
  - Backbone mask `[N, N, 1]`, query pair `[N, N, 128]`
- 2-layer `TemplatePairFormerBlock` stack + output LayerNorm
- Template pair features are masked to **intra-chain pairs only**: `pb_mask[i] × pb_mask[j] × multichain_mask` (`34326d5`). Template features come from per-chain extraction, each chain in its own frame, so cross-chain template distances/orientations are not real interface geometry and are masked out. The query passthrough and per-token aatype stay unmasked

**`TemplatePairFormerBlock`** -- Simplified PairFormer (no single track):

- tri_mul_out, tri_mul_in, pair_attn_row, pair_attn_col, pair_transition (factor=2)

**Helper functions**:

- `pseudo_beta_from_positions()` -- CB for all residues, CA for GLY
- `backbone_frames_from_positions()` -- Gram-Schmidt from N/CA/C atoms
- `dgram_from_positions()` -- Squared-distance binning into 39 bins (3.25-50.75 Å)

### `atom_cross_attn.py` -- Atom-Level Processing

**`AtomSubsets`** -- Flat atom layout with rectangular block attention:

- Packs real atoms from `[N, M=24]` grid into flat array, padded to multiple of Q=32
- Non-overlapping **query subsets** of Q=32 atoms (~6 residues each)
- Overlapping **key windows** of K=128 atoms centered on each query subset
- Fields: `ta_to_q_idx [S,Q]`, `q_to_k_idx [S,K]`, `token_for_query [S,Q]`, `token_for_key [S,K]`, etc.

**`EvoformerConditioningEncoder`**:

- Ref atom features (element, charge, name_chars, mask) -> per-atom embeddings
- 3-block atom transformer with pair conditioning
- Masked mean pool over atoms per token -> `[N, per_token_channels=384]`
- This becomes `evo_cond`, concatenated with `raw_feat` to form `target_feat [N, 447]`

**`AtomCrossAttEncoder`** / **`AtomCrossAttDecoder`**:

- Encoder: maps noisy atom positions to flat subset space, conditions with trunk single/pair
- Decoder: maps denoised subset features back to `[N, M, 3]` atom positions
- Trunk pair broadcast: `pair[token_for_query, token_for_key]` -> `[S, Q, K, pair_channels]`
- Reference-conformer geometry (`ref_pos` offsets + inverse distances) is gated by
  `v_lm = (ref_space_uid_l == ref_space_uid_m)` (AF3 Algorithm 5 line 3, `_offsets_valid`).
  `ref_space_uid` numbers each `(chain, residue_index)` conformer, so all atoms of one
  residue see each other — including one-atom tokens (ligand, modified residue, atomized
  residue), where a token-identity gate would collapse to the self-pair and drop the
  geometry entirely. Absent the feature the gate falls back to token identity.

### `diffusion.py` -- Denoising

**`DiffusionTransformer`** -- 24 blocks, organized in super-blocks of 4:

- Each block: AdaLN -> self-attention (pair-biased) -> AdaTransitionBlock -> atom cross-att encode/decode
- Conditioning: `single_cond [N, 384+447=831]`, `pair_cond [N, N, 128+139=267]`, noise embedding (Fourier, 256-dim)
- EDM-style skip/output scaling: `result = skip_scaling * noisy_pos + out_scaling * pos_update`

**`DiffusionHead.sample()`** -- 200-step reverse process:

1. `t = linspace(0, 1, 201)` -> `sigma = noise_schedule(t)` (cosine schedule with `sigma_data` from `DiffusionConfig`)
2. Initial noise: `x ~ N(0, smax^2) * atom_mask`, shape `[S, N, M, 3]`
3. Each step: `_random_augmentation()` -> denoise -> Euler update with stochastic gamma injection
4. Gamma: `gamma_0=0.8` if `sigma_next > gamma_min=1.0`, else 0 (no stochasticity near convergence)

### `confidence.py` -- Confidence Head

4-layer PairFormer operating on:

- **Pair input**: target_feat outer-sum `[N,N,128]` + distogram from predicted positions (39 bins, 3.25-50.75 Å range)
- **Single input**: single representation `[N, 384]`

Output projections (each preceded by LayerNorm):

- **pLDDT**: `LinearNoBias(384, M*50=1200)` -> `[S, N, 1200]`
- **PAE**: `LinearNoBias(128, 64)` -> `[S, N, N, 64]`
- **PDE**: `LinearNoBias(128, 64)` -> symmetrized -> `[S, N, N, 64]`
- **exp_resolved**: `LinearNoBias(384, M*2=48)` -> `[S, N, 48]`

Confidence is computed **per diffusion sample** -- the head runs S times.

### `model.py` -- Top-Level Orchestrator

**`DeepFold3`**: wires all modules together in `forward()`. Orchestrates recycling, calls evoformer, diffusion, confidence, and distogram heads.

**`DeepFold3Output`** -- 7 fields:

- `atom_positions [S, N, M, 3]`, `atom_mask [N, M]`
- `plddt [S, N, M*50]`, `pae [S, N, N, 64]`, `pde [S, N, N, 64]`, `exp_resolved [S, N, M*2]`
- `distogram [N, N, 64]`

**`DistogramHead`**: single `LinearNoBias(128, 64)` + symmetrize. 64 bins from 2.3125 to 21.6875 Å.

Also exposes `compute_distogram(features, num_recycles)` which runs only the evoformer (no diffusion) for fast validation.

### `masked_msa.py` -- Masked-MSA (BERT) Head (optional)

Self-supervised BERT-style MSA objective, gated by `ModelConfig.masked_msa.enabled` (default off). Training-only; runs in `forward_training`, never in `forward`/inference.

- **`apply_bert_mask`**: corrupts a `replace_fraction` (0.15) of valid MSA entries on the **gradient (final-recycle) pass only**, before the MSA stack. Replacement is the categorical mixture `uniform·random_aa + profile·profile + same·one_hot(msa)`, remaining mass → `const.MSA_MASK` (=31, the free slot in the 32-class MSA vocab, so the 34-d `msa_feat` input is unchanged). Returns `(bert_msa, true_msa, bert_mask)`.
- **`MaskedMSAHead`** -- single-query MSA cross-attention: for each token `i`, `h_i = Attn(q=s_i, k/v=m[:,i])` over the MSA rows, then `logits_ri = Linear(MLP([m_ri, s_i, h_i]))`. Inputs LayerNorm-ed; projections `LinearNoBias` with random (`linear`/`relu`) init — learns from scratch on structure-only warm-start. Output `[S, N, 32]`.

`Evoformer.forward` now also returns the final MSA-stack representation under key `"msa"`; the head reads `embeddings["msa"]` + `embeddings["single"]` (both attached), so its gradient trains the MSA stack and the single track. Loss: `training.loss.masked_msa_loss` (BERT CE over `bert_mask` entries), weight `TrainingConfig.alpha_masked_msa`.

## Atom Representation

Two parallel layouts coexist throughout the model:

**Token-level** `[N]` -- One representative atom per token:

- Protein: Cα (one token per residue)
- Nucleic acid: C1' (one token per residue)
- Ligand: the single atom itself (one token per atom)

**Token-atoms** `[N, M=24]` -- All heavy atoms per token, padded to `MAX_ATOMS_PER_TOKEN=24`:

- `ref_pos [N, M, 3]`: ideal CCD coordinates, randomly rotated during training
- `ref_mask [N, M]`: which atom slots are occupied
- `ref_element [N, M]`, `ref_charge [N, M]`, `ref_atom_name_chars [N, M, 4]`

**Flat subset mapping** (`AtomSubsets`):

- The `[N, M]` grid is flattened to a 1D array of real atoms (mask=True), padded to a multiple of Q=32
- Partitioned into non-overlapping query subsets of Q=32 atoms
- Each query subset gets an overlapping key window of K=128 atoms (centered on the query)
- `build_atom_subsets(ref_mask, Q=32, K=128)` computes all index tensors
- Used by atom cross-attention in both encoder and decoder

## Recycling

- Default `num_recycles=10`; the evoformer runs `num_recycles + 1` times total
- `prev_pair [N, N, 128]` and `prev_single [N, 384]` carry information across iterations
- `.detach()` between recycles prevents gradient flow through the recycling chain
- Recycled representations are added to fresh projections via LayerNorm + LinearNoBias

## Diffusion Sampling

The diffusion head implements a 200-step reverse denoising process:

**Noise schedule**: `sigma(t) = sigma_data * (smax^(1/p) + t * (smin^(1/p) - smax^(1/p)))^p`

- All parameters (`sigma_data`, `smin`, `smax`, `noise_p`) read from `DiffusionConfig` (defaults: 16.0, 0.0004, 160.0, 7.0)

**Random augmentation** (critical -- applied at every denoising step):

1. Center: subtract masked center of mass
2. Rotate: Gram-Schmidt orthogonalization from two random N(0,1) vectors -> SO(3)
3. Translate: add random N(0,1) vector
4. Zero out masked atoms

**Stochastic sampling**:

- `gamma = gamma_0 (0.8)` when `sigma_next > gamma_min (1.0)`, else 0
- `sigma_hat = sigma * (1 + gamma)` -- inflated noise level
- Inject noise: `x_hat = x + noise_scale * sqrt(sigma_hat^2 - sigma^2) * noise`
- Euler step: `x_next = x_hat + step_scale * (sigma_next - sigma_hat) * (x_hat - denoised) / sigma_hat`

**Output**: `[num_samples, N, M, 3]` atom positions (default `num_samples=5`).

**Step-invariant state**: `DiffusionHead.compute_static` runs once per sample call, once per
guidance branch, and the S samples share one batched denoiser call per step. With
`DiffusionConfig.hoist_sampler` (on for `deepfold3 inference`), `hoist_static` also drops
the empty atom windows (`AtomSubsets.trim_empty`), precomputes the token and atom pair
logits and atom AdaLN terms (`DiffusionTransformer.forward_hoisted`,
`AtomCrossAttTransformer.forward_hoisted`), and shares one copy across samples. See
`docs/optimizations.md` § _Diffusion sampler_.

### Distance restraints (`model/restraints.py`)

Restraints act on **coordinates inside the sampler**, not on the conditioning as
`--guidance` does, and nothing was trained for them. Each is a flat-bottom
penalty whose gradient is taken with autograd on a copy — under
`inference_mode(False)`, because a tensor made under inference mode records no
graph even inside `enable_grad`, and without that the restraint is a silent
no-op:

| CLI | energy | meaning |
|---|---|---|
| `--restraint A/25-40:B@6` | `Σ_i max(0, d_i − D)²` | every residue of the left selector must come within `D` Å of the right side, `d_i` a soft minimum over that side's atoms |
| `--restraint-pair A/59+33:C/97+94@5` | the same, per **paired** residue | residue 59 to residue 97, 33 to 94; a group restraint is satisfiable by the wrong pose, a pairing is not |
| `--repel A/12+15:B@8` | `Σ_i max(0, D − d_i)²` | the same with the sign reversed, so it stops pushing once the bound is cleared instead of driving the chains apart |

The soft minimum is a softmax-weighted mean of the distances, **not**
`−logsumexp(−λd)/λ`: the latter sits below the true minimum by `ln(count)/λ`, so
a restraint on a large group would read as satisfied while still that far away
and the bias would grow with the selection's size.

`--restraint-rigid` (the default) projects the per-atom gradient onto the six
rigid degrees of freedom of the chain it acts on, so the chain translates and
rotates and its internal geometry is untouched; `--restraint-residues` pulls the
named residues alone and stretches the chain where the selection ends.
`--restraint-weight` is the fraction of the remaining violation removed per
step and `--restraint-max-step` caps the displacement — there is no momentum in
the loop, so nothing overshoots on its own and instability could only come from
the step size. `@LO..HI` on a selector, or
`--restraint-after-step --restraint-after-sigma`, limits the push to a σ̂ range;
the last step never pushes, so the structure returned is always a denoiser
output.

## Batch Dimension Support

All core model modules support both unbatched and batched (leading B dim) inputs:

- Pattern: `has_batch = x.dim() == expected_batched_dim; if not: unsqueeze(0) -> compute -> squeeze(0)`
- Supported: attention, ops, evoformer, diffusion, confidence, embedders
- **Not yet batched**: `atom_cross_attn.py` (subset building is per-sample)

## Training

Training infrastructure is in `src/deepfold3/training/`. See [training.md](training.md) for full details.

- **Loss**: Multi-component (diffusion MSE + smooth LDDT + distogram + confidence)
- **DDP**: Native PyTorch `DistributedDataParallel` (no Lightning)
- **Optimizer**: Adam with AF3-style LR schedule (warmup → plateau → step decay)
- **EMA**: Exponential Moving Average with constant decay (params-only; buffers are init-time constants and not tracked)
- **Config**: All hyperparameters in `TrainingConfig` dataclass — nothing hardcoded

## Fused Kernel Checkpoint Portability

When fused Triton kernels are enabled (via `OptimConfig`), modules create derived parameter copies (e.g. `_fused_gate.*`, `_fused_ada_ln_linear.*`). These are **excluded** from checkpoints and EMA:

- **`DeepFold3.state_dict()`** filters out keys matching `OptimConfig.FUSED_STATE_DICT_MARKERS`
- **`DeepFold3.load_state_dict()`** relaxes `strict=True` when fused keys are absent; derived params are rebuilt from base weights at the next forward pass
- **`EMA`** skips fused parameters during shadow initialization (they would double-decay against the base params they shadow)

This ensures checkpoints are portable — the same `.pt` file loads correctly whether fused kernels are enabled or not. The marker list is defined once in `OptimConfig.FUSED_STATE_DICT_MARKERS`.

`inference` and `inference-batch` load with `strict=False`, then `check_loaded_keys` (`scripts/_common.py`) fails the run on any missing key outside those markers, because that parameter would stay random. `--allow-missing-keys` is the opt-out; `inference` records the keys in `run_manifest.json`. Raw AF3 params always need it, as they lack the Fourier weights.

## Weight Compatibility

- All weights load from AF3 Haiku params via `convert/af3.py` (Haiku scope -> PyTorch state_dict)
- Zero missing / zero unexpected keys (validated in `test_weight_loading.py`)
- All AF3 scopes converted, including template embedding
- Haiku `[in, out]` transposed to PyTorch `[out, in]` during conversion
- Template embedding scalar projections (`num_input_dims=0` in Haiku) require special handling: weight shape `(C,)` -> `(C, 1)` via `_linear_scalar()`
- Each module's docstring documents its corresponding Haiku scope hierarchy
