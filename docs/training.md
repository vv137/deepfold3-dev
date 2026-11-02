# Training

DeepFold3 uses native PyTorch DDP for distributed training. The training procedure follows AF3 §3.7 with Protenix-style implementation choices.

## Quick Start

```bash
# Single GPU
deepfold3 train --data-dir /data/training --output runs/exp1

# Multi-GPU (DDP via torchrun)
torchrun --nproc_per_node=4 -m deepfold3.scripts.train \
    --data-dir /data/training --output runs/exp1

# Resume from checkpoint
deepfold3 train --data-dir /data/training --output runs/exp1 \
    --resume runs/exp1/checkpoint_1000.pt

# With config overrides
deepfold3 train --data-dir /data/training --output runs/exp1 \
    --config config.json --lr 0.001 --max-tokens 512

# Multi-dataset training (primary PDB + distillation AFDB)
deepfold3 train --data-dir /data/pdb --output runs/exp1 \
    --dataset afdb:/data/afdb:0.5:uniform:true

# Multi-dataset via JSON config
deepfold3 train --data-dir /data/pdb --output runs/exp1 \
    --config configs/multi_dataset.json
```

## Training Config

All training hyperparameters are in `TrainingConfig` (`src/deepfold3/model/config.py`). Nothing is hardcoded — override via JSON config or CLI flags.

```json
{
  "training": {
    "lr": 0.0018,
    "weight_decay": 0.05,
    "betas": [0.9, 0.95],
    "warmup_steps": 1000,
    "start_decay_after": 50000,
    "decay_every": 50000,
    "decay_factor": 0.95,
    "grad_clip_norm": 10.0,
    "accumulate_grad_batches": 1,
    "ema_decay": 0.999,
    "max_tokens": 384,
    "max_atoms": 4096,
    "max_recycles": 3,
    "diffusion_batch_size": 48,
    "diffusion_chunk_size": 0,
    "mini_rollout_steps": 20,
    "alpha_diffusion": 4.0,
    "alpha_distogram": 0.03,
    "alpha_confidence": 1e-4,
    "alpha_pae": 0.0,
    "alpha_bond": 0.0,
    "nucleotide_loss_weight": 5.0,
    "ligand_loss_weight": 10.0,
    "min_resolution": 0.1,
    "max_resolution": 4.0,
    "train_confidence_only": false,
    "finetune_scheduler": "af3",
    "max_checkpoints": 0,
    "wandb_project": "",
    "dtype": "bf16",
    "max_steps": 200000,
    "seed": 42
  },
  "model": {
    "num_recycles": 10,
    "msa_crop_size": 512
  }
}
```

## Training Procedure

### Forward Pass (per batch)

```
1. Evoformer trunk (random 0–3 recycles = 1–4 trunk passes, gradient on last only)
     → single [N, 384] + pair [N, N, 128]

2. Diffusion training step (gradient flows)
     - Augment GT positions: random rotation + translation per sample
     - Sample noise: log-normal σ ~ σ_data × exp(p_mean + p_std × z)
     - Add noise: x_noisy = x_gt_aug + σ × ε
     - Denoise once → x_denoised
     - Losses: weighted MSE + smooth LDDT

3. Mini-rollout (no gradient, eval mode)
     - 20-step denoising → predicted positions
     - Used as input to confidence head

4. Confidence head (gradient flows through confidence PairFormer)
     - Runs on mini-rollout positions
     - pLDDT, PAE, PDE, exp_resolved losses against GT

5. Distogram head
     - Cross-entropy on distance bin predictions
```

### Chain permutation & the stability gate

The chain permutation (AF3 §4.3) is resolved **once** from the mini-rollout and shared with every
GT-based loss (diffusion MSE, smooth-lDDT, distogram, FAPE, pLDDT/PAE/PDE, tm_true) — `loss.py`
`chain_perm_from_mini_rollout`. For a symmetric multimer whose rollout is still collapsed, the
greedy centroid assignment is a **coin-flip**; across steps the relabelling oscillates and the
symmetry-breaking gradient self-cancels, latching the assembly near the collapsed
(symmetric-average) point. See `export/chain-perm-latch/` for the mechanism, the offline probe, and
the simulation.

The **stability gate** (`chain_perm_stability_gate`, default off; `chain_perm_gate_k`, default 5)
mitigates this: the mini-rollout is run with `K = chain_perm_gate_k` independent samples and the
permutation is applied **only when all K agree** on the whole-assembly assignment, else it falls
back to identity (un-permuted GT). While copies overlap the K noisy rollouts disagree → a
deterministic separating target; once they separate the K agree → the AF3 permutation fires. The
simulation shows K=3 recovers the latch zone and K=5 is safe to extreme rollout noise; **unanimity,
not majority, is required**. The K samples live under `pred["chain_perm_gate_pos"]`; the confidence
/ tm_true / atom-symmetry consumers still see only `mini_rollout_samples` of them via
`pred["mini_rollout_pos"]`.

### Masked (partial) diffusion & conditioning dropout

Three **off-by-default** knobs on the diffusion training step. At their defaults none of them draws
from the RNG, so enabling the code cannot perturb an existing run's random stream
(`tests/test_diffusion_masking.py`).

| knob | default | on | source |
|---|---|---|---|
| `model.diffusion.diffusion_mask_prob` | `0.0` | fraction of diffusion samples that keep part of the GT clean | written from scratch — **OF3 has none** |
| `model.diffusion.use_conditioning_prob` | `1.0` | probability of *keeping* the trunk conditioning | OF3 `use_conditioning_prob`, YAMLs use `0.8` |
| `model.diffusion.cond_drop` | `"pair"` | what a drop zeroes: trunk pair only, or `"both"` single + pair | `"both"` is OF3's drop |
| `model.diffusion.guidance_scale` | `1.0` | inference CFG weight w: `D_u + w·(D_c − D_u)`; needs a checkpoint trained with the drop | written from scratch — **OF3 has none** |
| `model.diffusion.cond_drop_interface_frac` | `0.0` | fraction of drops that zero a random subset of inter-chain pair blocks instead (r ~ U(0,1) per chain pair, ≥1 pair); single-chain crops take the full drop | written from scratch |
| `model.diffusion.guidance_sigma_range` | `null` | `[lo, hi]`: apply guidance only for σ̂ in range | Kynkäänniemi et al. 2024 |
| `model.diffusion.guidance_drop` | `"full"` | unconditional branch: `"full"` (`cond_drop`) or `"interface"` (all inter-chain pair blocks; needs `cond_drop_interface_frac > 0` in training). `DiffusionHead.sample(guidance_pair_keep=…)` takes any chain-pair subset | written from scratch |
| `model.diffusion.hoist_sampler` | `false` | inference sampler speedup, on for `deepfold3 inference` (`--no-hoist-sampler` to disable); see `docs/optimizations.md` § _Diffusion sampler_ | written from scratch |
| `deepfold3 inference --guidance W[@TARGET[@LO..HI]]` | none | repeatable; replaces the three `guidance_*` knobs with a linear combination `D_c + Σ_k (w_k − 1)·(D_c − D_u,k)`. `TARGET` = `full`, `all` or chain pairs `A:B,AA:B`; either side may name residues of its chain in the input's own numbering, `+` between ranges (`A/25-40+88:B`), for epitope-level steering; `LO..HI` is absolute σ̂ (default schedule 0.0077..4608, bounds inclusive). The pairs of one term union into one mask (one extra denoiser pass); a block dropped twice inside a term is an error and across terms a warning, since it would carry the summed weight — `--allow-guidance-overlap` accepts both. A residue subset is a finer drop than training's whole chain-pair blocks, so it is extrapolation. Refuses a checkpoint trained without the drop. See `model/guidance.py` | written from scratch |
| `model.confidence.use_pair_emb_prob` | `1.0` | probability of *keeping* the trunk pair emb in the confidence head | OF3 `use_confidence_emb_prob`, YAMLs use `0.8` |

With `use_conditioning_prob < 1` every training forward logs `train/cond_drop/{none,full,interface}` (one-hot, so window means are the realised rates), `train/cond_drop/multichain` (share of forwards whose crop could take an interface drop) and, on interface drops only, `train/cond_drop/interface_pair_frac` (fraction of inter-chain token pairs zeroed).

**Masked / partial diffusion (motif scaffolding).** With probability `diffusion_mask_prob` a
diffusion sample keeps a subset of the (augmented) ground truth **clean** and noises only the rest,
so the model learns to place the noised part relative to a given anchor.
`diffusion_mask_mode="chain"` keeps a random non-empty *proper* subset of chains (always ≥1 clean
and ≥1 noised; falls back to `"span"` on a single-chain crop); `"span"` keeps one contiguous token
run. The clean fraction is uniform in `[diffusion_mask_min_frac, diffusion_mask_max_frac]`. The mask
is built in `forward_training` **after** the chain permutation, so "clean chain A" names the copy
the GT was actually permuted to, and applied in `_sample_training_impl` **after** the per-sample
random augmentation, so the anchor shares its sample's frame.

Motivation (`export/pose-decomposition/`): the inter-chain pose is committed by the *first* denoise
call and resampling does not escape it, and uniform noising cannot break that — any σ low enough to
preserve good local structure also preserves the wrong pose, while any σ high enough to erase the
pose erases everything and re-commits identically. Selective noising is the only way to erase a pose
while keeping a structure.

Two limitations to know before turning it on:

- **Representation is RePaint-style replacement at the single global σ.** The EDM input scaling
  `c_in = 1/√(σ²+σ_d²)` is global, so at σ ≫ σ_data the anchor arrives numerically ~300× smaller
  than the noised cloud. The principled fix is a per-token σ, which needs the noise embedding and
  the `c_in`/`c_skip`/`c_out` broadcasts to go per-token. (OF3 cannot do it either — its `t` is
  `[B, S]`.)
- **The loss is not masked to the noised region.** At high σ `c_skip → 0`, so reproducing the clean
  region is a real reconstruction task rather than a free copy, and leaving the loss alone keeps
  this option from touching the loss code at all.

**Conditioning dropout** mirrors OF3 exactly, including a wrinkle worth not "fixing" by accident:
`single`/`pair` are zeroed only on the way into `DiffusionConditioningBlock.compute_static`, while
`target_feat`, `rel_feat` and the Fourier noise embedding survive **and the un-zeroed trunk single
still reaches the atom encoder** (OF3 `diffusion_module.py:198-204`, AF3 Algorithm 20). It is
therefore a *partial* drop, not an unconditional pass. One draw per forward, shared by all diffusion
samples. Zeroing an input still reaches every parameter, so DDP `static_graph=True` is unaffected.
OF3 ships no inference-time guidance that consumes this; training with it is what would make
classifier-free guidance possible later.

For the confidence head only the **pair** repr is dropped — single, `target_feat` and the predicted
positions are untouched — and the distogram head reads the un-zeroed `embeddings["pair"]`
separately, so the distogram loss is never affected (matching OF3's ordering).

### Loss Components

| Loss            | Weight                             | Description                                                                                                                                                                    |
| --------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Diffusion MSE   | `alpha_diffusion` (4.0)            | EDM-weighted MSE per AF3 §5.3 eq. 3: `(1/3) · Σ‖Δ‖²·w·mask / Σ mask` (per-coordinate average; denominator is atom count, not sum-of-weights). w(σ) = (σ²+σ_d²)/(σ·σ_d)²; mol-type reweight: DNA/RNA 5×, ligand 10×. |
| Smooth LDDT     | `alpha_diffusion` (4.0)            | Distance-based accuracy (cutoffs: 0.5, 1.0, 2.0, 4.0 Å)                                                                                                                        |
| Bond violation  | `alpha_diffusion × alpha_bond` (0) | MSE on bonded atom distances vs observed GT; enable for stage 2                                                                                                                |
| Shape complementarity | `alpha_shape_comp` (0)             | SE(3)-invariant cross-chain protein interface loss (OpenDDE `arXiv:2607.03787` §A.2.1); Huber regression of the predicted surface-complementarity field onto the GT field. See below. |
| Distogram       | `alpha_distogram` (0.03)           | Cross-entropy on 64-bin distance predictions                                                                                                                                   |
| Masked MSA      | `alpha_masked_msa` (0)             | BERT-style MSA reconstruction CE over corrupted entries. Requires `ModelConfig.masked_msa.enabled`. Single-query MSA cross-attention head at the end of the MSA module; trains the MSA stack + single track. All samples incl. distillation. |
| pLDDT           | `alpha_confidence` (1e-4)          | Per-atom confidence cross-entropy (50 bins); gated on resolution ∈ [0.1, 4.0] Å. AF3 §4.3.1: j-axis restricted to representative atoms (CA / C1') of standard polymer tokens, with inclusion radius 15 Å for protein-rep j and 30 Å for nucleic-rep j; ligand j atoms drop. |
| PDE             | `alpha_confidence` (1e-4)          | Predicted distance error cross-entropy                                                                                                                                         |
| exp_resolved    | `alpha_confidence` (1e-4)          | Binary resolved/unresolved classification. Supervision domain is `ref_mask` (canonical heavy-atom slots) AND-gated on `gt_seq_mask`, so tokens the chain permutation left unmatched leave the denominator instead of being taught "unresolved". |
| PAE             | `alpha_confidence × alpha_pae` (0) | Frame-aligned predicted aligned error (AF3 §4.3.2, Algorithm 29 — disabled in stage 1). Frames whose ∠(a, b, c) lies within 25° of co-linearity are dropped per-sample on both pred and GT. |
| Single-repr reg | `alpha_single_reg` (0)             | Magnitude regularizer on trunk single output: mean per-token Σ_c &#124;s&#124;^p / C. Observation-only at α=0.                                                                 |
| Pair-repr reg   | `alpha_pair_reg` (1e-4)            | Same form on pair output. Controls DistogramHead logit magnitude; prevents late-trajectory distogram destabilization (see [regularization-story.md](regularization-story.md)). |

#### Shape complementarity (`alpha_shape_comp`)

Off by default. A Huber regression of the predicted surface-complementarity field
onto the field computed from the ground truth, applied to the diffusion output
next to smooth-lDDT (OpenDDE's `L_geom = L_diff + λ_shape·L_shape + …`).

Each protein token gets an outward surface normal from the **same-chain** atomic
density, `n_u = g_u / (‖g_u‖ + ε)` with
`g_u = Σ_{a ∈ chain(u)} exp(-‖c_u-x_a‖²/2σ²)(c_u-x_a)/σ²` — the negative gradient
of a Gaussian density, so it points away from the token's own chain, and its
magnitude is a burial read-out (a buried token sits in an isotropic density and
the gradient cancels). Every cross-chain pair inside
`shape_comp_interface_cutoff` is then scored by whether the two surfaces face
each other, carry opposed normals, sit at a plausible gap, and avoid a clash:

```
q_uv = ReLU(n_u·d̂) ReLU(n_v·(-d̂)) · (1 - n_u·n_v)/2
       · exp(-((d - gap_mean)/gap_scale)²) · (1 - sigmoid((clash_distance - d)/clash_scale))
q_u  = Σ_v softmax_v(-d²/pool_temperature) q_uv        q_global = mean_u q_u
L    = λ_p·H(q_uv, q*_uv) + λ_t·H(q_u, q*_u) + λ_g·H(q_global, q*_global)
```

Properties worth knowing before enabling it:

- **SE(3)-invariant** — no rigid alignment, so it constrains inter-chain geometry
  without routing through a superposition dominated by the larger chain. Also
  mirror-invariant (carries no chirality signal), and protein–protein only.
- **GT-only masking.** The ground-truth field fixes which pairs are scored, so
  the pair index set is static across the `S` diffusion samples and is built once
  under `no_grad`. An interface the prediction *invents* where GT has none is
  therefore invisible here (smooth-lDDT and the distogram already penalise it).
- **~8 Å range of action.** Monotone in interface error to ~8 Å, saturated by
  ~12 Å, exactly zero beyond the cutoff. A refinement objective, not a docking
  one — it will not rescue a mis-docked pose.
- **`shape_comp_huber_delta` is `F.smooth_l1_loss`'s `beta`, not
  `F.huber_loss`'s `delta`.** PyTorch defines `huber(d) = d·smooth_l1(beta=d)`,
  so the huber spelling would scale the term and its gradient by 0.01.
- OpenDDE's own weight is `alpha_shape_comp = 3.0e-2`; at that value the term's
  gradient measured 0.06–0.21× the smooth-lDDT gradient on rigidly perturbed
  complexes. Neither the paper nor the released code ablates the term.

Logged components: `shape_comp_{pair,token,global}` (raw Huber),
`shape_comp_l1_token` (raw L1 — the interpretable scale),
`shape_comp_coverage` (fraction of examples with a scorable interface), and
`shape_comp_glob_{gt,pred}`.

Full investigation, measurements and the arm plan: `export/shape-complementarity/`.

#### GT-presence gating of geometric masks

The featurizer's `frames_mask`, `fape_atom_mask` and `rep_atom_present` all derive from
`ref_mask` — the **CCD reference conformer's** atom set — not from `atom_gt_mask` (what the
experimental structure actually resolved). `atom_gt_positions` is zero-filled at unresolved
slots, so any loss that dereferences a CCD-present-but-GT-missing atom silently reads
`(0, 0, 0)`. The 25° co-linearity filter does not catch this: an origin-substituted atom
yields an arbitrary, usually well-conditioned angle.

`training_loss` / `diffusion_only_loss` therefore re-derive presence from the
**chain-permuted** `atom_gt_mask` before use:

| Gate                 | Covers                                     | Consumers                          |
| -------------------- | ------------------------------------------ | ---------------------------------- |
| `rep_present`        | the rep atom (CB / C4 / C2 / CA)            | distogram, PDE, PAE, `tm_true`     |
| `frames_gt_present`  | all three frame atoms (a, b, c)             | PAE, FAPE frames, `tm_true`        |
| point-mask AND       | each FAPE K-slot point atom                 | FAPE point set                     |

Because these gates read the permuted mask (`& c_keep`), they also drop tokens the chain
permutation left without a GT counterpart — otherwise FAPE would score them against another
chain's coordinates.

#### Diagnostic metrics (logged, not in total)

`training_loss` also emits scalars that are logged via `losses[...]` but never added to `total`:

| Key        | Source                              | Description                                                                                                                                                                                                          |
| ---------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tm_true`  | mini-rollout positions + GT frames  | TM-like global accuracy in [0, 1] computed from the *true* per-frame distance error `‖pred_local − gt_local‖` via the same `1/(1+(d/d0)²)` kernel used for inference pTM. Uses N (effective token count) for `d0`.   |
| `itm_true` | as above, restricted to cross-chain | Cross-chain TM-like accuracy (`asym_id_i ≠ asym_id_j`). Skipped on single-chain examples (no cross-chain pairs to score).                                                                                            |

Both fire from the mini-rollout (S=1) and are computed under `torch.no_grad()`, so they're available **independent of `alpha_confidence`** — they show up in stage 0/1/2 (when the confidence head forward is gated off) and can be compared directly to inference-time pTM/ipTM. PAE-derived `tm_pred` / `itm_pred` are not currently emitted; `scripts/run_status.py` parses both pairs of keys when present.

These diagnostics are O(N²) and run **every step**, so they're individually gateable for throughput:

- `log_tm_true` (default `True`) — `tm_true` / `itm_true`. Suppresses the frame expansion in stages 1–3; **computed regardless when `alpha_pae > 0`** (FT3), since the PAE loss already builds the same `express_coordinates_in_frame` error from the same mini-rollout frames, making `tm_true` ~free on the margin there.
- `log_distogram_entropy` (default `True`) — `distogram_entropy` (`H(p_θ)` over the `[·, N, N, 64]` distogram softmax). The distogram *loss* is unaffected; only the entropy readout is gated.

### LR Schedule

Two schedulers available, selected by `finetune_scheduler`:

**AF3 schedule** (default, `finetune_scheduler="af3"`):

```
1. Warmup (0 → warmup_steps): linear 0 → lr
2. Plateau (warmup_steps → start_decay_after): constant lr
3. Step decay: lr × decay_factor^n, n = (step − start_decay_after) // decay_every
```

Default: lr=0.0018, warmup=1000, `start_decay_after`=50k, factor=0.95 every 50k.

Note the floor in phase 3: `start_decay_after` is where the decay **windows start being counted**, not where the first drop happens. With `start_decay_after == decay_every == 50000` the multiplier is `0.95^0 = 1` for the whole of `[50k, 100k)` and the first drop lands at **100k**. To put the first drop at step `S`, set `start_decay_after = S − decay_every` (e.g. 10k + 50k → 60k, 110k, 160k) or shorten `decay_every`.

**Warmup → constant → decay** (`finetune_scheduler` = `cosine`, `linear` or `1-sqrt`): linear warmup over `warmup_steps`, constant `lr` until `finetune_decay_start` (default 0: no constant phase), then a decay of the named shape to `lr × finetune_min_lr_ratio` at `finetune_max_steps`, held there afterwards. `1-sqrt` is the warmup-stable-decay cooldown of Hägele et al. 2024 (arXiv:2405.18392). `cosine_max_steps` / `cosine_min_lr_ratio` are the deprecated spellings, still read when the `finetune_*` key is unset. `configs/stage3.yaml` uses this for the 640-token stage.

**Frozen submodules** (`freeze_modules: [evo_cond_encoder, evoformer]`): sets `requires_grad=False` on the named top-level `DeepFold3` children before DDP, optimizer and EMA are built, so they get no optimizer state and no EMA shadow (`--snapshot shadow` overlays the shadow onto the saved weights, which carry the frozen part unchanged). With no trainable parameter and no grad-requiring input the frozen trunk should build no autograd graph; confirm the memory drop in a smoke test. Exclusive with `train_confidence_only`. `configs/stage2-frozen-trunk.yaml` trains the diffusion head under stage2 conditioning dropout on a fixed trunk.

### EMA

Constant-decay exponential moving average (matches OF3 `ExponentialMovingAverage`):

```
shadow = (1 - decay) × param + decay × shadow
```

EMA weights are swapped in for evaluation (zero-copy pointer swap on `param.data`) and saved in checkpoints. Buffers are not tracked — DF3 buffers are constants set once at init (Fourier weights, ref-atom tables) or shadows derived from base params (fused/tiled kernels), so EMAing them would either be a no-op or actively wrong. Fused-kernel parameters are excluded from EMA shadow copies for the same reason — they are re-derived from base weights at every step.

### Recycling

During training, the number of recycles is randomly sampled per microstep: `n ∈ [0, max_recycles]` (AF3 convention — total trunk passes = n + 1). With `uniform_recycle=true` (default), all DDP ranks use the same n per microstep (seeded on `seed + global_step`) to avoid straggler divergence. Only the final trunk pass propagates gradients; earlier passes run under `torch.no_grad()`.

### DDP Configuration

- Backend: NCCL
- `static_graph=True` when unchunked; auto-disabled when `diffusion_chunk_size > 0` (chunked path uses `model.no_sync()` which conflicts with static_graph)
- `find_unused_parameters=False` (default)
- `use_reentrant=False` for all gradient checkpoints (required for DDP compatibility)
- `destroy_process_group()` called at end of training loop with a `dist.barrier()` first
- `faulthandler` registered on SIGUSR1 for DDP deadlock diagnosis
- Bucket-padded collation (`bucket_pad=True`) for deterministic kernel shapes across microsteps
- Launch with `torchrun` for multi-GPU

### Host Memory Policy

`Trainer._init_memory_policy` sets two process-wide policies at startup. Both target host-memory pressure that grows with *samples consumed* rather than wall-clock, which is why the resulting stalls recur on a step-tied period (see `export/burst-periodic-stall/`) and reset when the job restarts. Under DDP, one rank in direct reclaim holds all ranks at the next collective.

| Policy | Mechanism | Scope |
| --- | --- | --- |
| Page cache | `set_drop_page_cache(True)` makes `data/io_cache.py:read_npz` read whole + `POSIX_FADV_DONTNEED` | Every dataset-path NPZ read (`Structure.load`, `MSA.load`, `TokenMask.load`, OF3 loaders). Records are drawn at random from 10⁶–10⁷-structure datasets, so a cached NPZ is never re-read. Packed members (`data/pack.py`) drop their byte range widened to page bounds, since `DONTNEED` keeps partly-covered pages and tar members are 512-aligned. |
| Copy-on-write | `os.register_at_fork(before=gc.freeze)` | Dataloader workers fork from the trainer and inherit its heap; a GC pass in a worker writes to every object's GC header and un-shares those pages. Bounds the growth — refcount updates still dirty objects on first touch. |

The page-cache policy is **off by default**, so offline CLIs (`prepare`, `import-of3`, `refilter`, `build-masks`, …) that share those loaders keep normal caching and readahead. Loaders with genuine reuse must not use `read_npz` — `data/parse/pdb_chain_cache.py` runs at a ~99.8% hit rate, and dropping its pages would turn every hit into a re-read.

The large immutable CCD tables are memoized per process (`ccd.py:load_ligand_symmetry`, `lru_cache`): each parse retains ~0.6 GB and every `TrainingDataset` in a rank asks for the same file (3 under the current configs).

### Gradient Management

- Gradient clipping: L2 norm (default 10.0)
- Gradient accumulation: configurable via `accumulate_grad_batches`
- Loss scaled by `1/accumulate_grad_batches` before backward
- Optimizer steps only on accumulation boundaries
- **NaN detection**: distributed all-reduce ensures all ranks agree on NaN/Inf; affected steps are skipped with a warning

**Global vs. per-rank clipping.** By default the clip runs on the DDP-averaged gradient (all-reduce fires, overlapped with backward, then `clip_grad_norm_` clips the mean). `per_rank_grad_clip: true` (OF3-style) instead runs the **entire backward under `no_sync`** so `.grad` stays each rank's local contribution, clips that local gradient to `grad_clip_norm`, and only then averages the clipped per-rank grads with a single **manual all-reduce** (`SUM ÷ world_size`). This bounds any one rank before it is mixed into the mean, so a pathological sample cannot dominate the batch-mean gradient. Trade-offs: the manual all-reduce runs after the full backward (no backward/all-reduce overlap), the `reduce_dtype` comm hook is bypassed (fp32 reduction), and `static_graph` is forced off. With `batch_size_per_rank > 1` the clip is per-rank over the within-rank sample mean, not literally per-sample. The finite/outlier skip consensus uses the per-rank pre-clip norm (catches a blow-up before the reduce spreads it). Two norms are logged: `grad_norm` (per-rank pre-clip) and `grad_norm_global` (`train/grad_norm_global`, post-reduce, ≤ clip).

### Mixed Precision

Training runs under `bf16` autocast by default (`TrainingConfig.dtype`). A few numerically sensitive regions are pinned to `fp32` via `ModelConfig.precision` (`PrecisionConfig`):

| Flag                             | Default | Region                                                                       |
| -------------------------------- | :-----: | ---------------------------------------------------------------------------- |
| `fp32_loss`                      | `true`  | `training_loss()` — softmax / CE see fp32 logits                             |
| `fp32_confidence_head`           | `true`  | ConfidenceHead final LN+Linear + distance embedding                          |
| `fp32_distogram_head`            | `false` | DistogramHead logit projection                                               |
| `fp32_diffusion_sample`          | `false` | Inference Euler-Maruyama sampler (mirrors `DiffusionConfig.skip_amp_sample`) |
| `fp32_diffusion_sample_training` | `false` | Training-time rollout (mirrors `DiffusionConfig.skip_amp_sample_training`)   |

Each flag disables `torch.autocast("cuda")` over its scope and upcasts inputs to fp32. Rigid alignment (`weighted_rigid_align` in `training/loss.py`) and ligand frame construction (`express_coordinates_in_frame`) are always fp32 — they hold their own `autocast(enabled=False)` contexts regardless of these flags.

Flip `fp32_distogram_head: true` if the per-pair distogram loss exceeds `ln(num_bins) ≈ 4.16`, which signals bf16 overflow in the final projection. Expected runtime cost is <5% — the guarded regions are tiny head-level matmuls.

YAML:

```yaml
model:
  precision:
    fp32_loss: true
    fp32_confidence_head: true
    fp32_distogram_head: false
```

### Experiment Tracking

Optional WandB integration: `--wandb <project>`. Logs train/eval losses, learning rate, epoch, and gradient norms. Only rank 0 logs.

## Data Requirements

Training data directory structure (`format: df3`; every file is sharded by the first two characters of its id, see `data/paths.py`). The inventory of every dataset on `/data`, including the `of3` / `of3pdb` layouts, is in [datasets.md](datasets.md).

```
data_dir/
├── manifest.json                            # Record index (optional — auto-scanned if missing)
├── structures/{ab}/{id}.npz                 # Structure (atoms, residues, chains, bonds)
├── templates/{ab}/{id}.json                 # Multimer template index (optional)
├── monomer_templates/{ab}/{msa_id}.json     # Monomer template index, per sequence (optional)
└── msa/
    ├── unpaired/{ab}/{msa_id}.npz           # MSA (sequences, mask, deletion_matrix)
    └── paired/{ab}/{id}_paired_{hash}.npz   # Paired MSA (optional)
```

Records are loaded via `discover_records()` from `data/prepare/manifest.py`: reads `manifest.json` if present, otherwise scans `structures/` for NPZ files.

**Train/eval split**: records are shuffled with a fixed seed (42) then split 95/5 (at least 1 eval record). The seeded shuffle ensures reproducible splits regardless of manifest ordering.

**Sampling (two-stage, AF3 §2.5.1)**: Per-anchor cluster-balanced sampling is factored into a **record marginal × anchor conditional** decomposition — mathematically identical to a flat per-anchor table by the chain rule (`p(record)·p(anchor|record) = w_a / W`), but keeping the sampler per-record (DDP-simple) and moving the anchor choice next to the cropper that consumes it:

1. **Record marginal** (the sampler): `WeightedClusterSampler` (single-dataset) / `MultiDatasetSampler` (multi-dataset) draw a *record* with probability `∝ compute_cluster_weights` = `Σ_{a∈record} w_a`, and yield record indices.
2. **Anchor conditional** (the dataset, at load time): a weighted dataset (`anchor_sampling=True`) draws one of the record's chain/interface anchors with probability `w_a / Σ w_a` in `_draw_crop_anchor`, seeding the crop (chain → contiguous/spatial seed; interface → interface-seeded spatial crop). Uniform datasets skip this and let the cropper self-seed.

Per-anchor weight `w = β[type] × Σ(α[mol] × count) / N_clust` with α(protein)=3, α(nucleic)=3, α(ligand)=1, β(chain)=0.5, β(interface)=1.0. For chains, `N_clust` is the chain cluster size. For interfaces, `N_clust` is the **interface cluster** size per AF3 §2.5.3 — interfaces share a cluster iff their constituent chain pairs map to the same pair of chain clusters. Both stages use the **same** α/β so the chain-rule identity holds exactly. Synthetic epoch size of 10,000 samples. Falls back to uniform `DistributedSampler` if no cluster data (anchor sampling still active on the dataset, so anchors remain mol-type-weighted).

The one behavioural note vs a flat per-anchor sampler: `_draw_crop_anchor` seeds its RNG from `(worker-epoch, idx)` only, so repeated draws of the *same* record within one `epoch_size` chunk reuse the same anchor. This is unbiased across chunks (each chunk reseeds) and negligible in practice (epoch_size ≪ dataset size → same-record repeats within a chunk are rare); full per-draw anchor diversity would require the sampler to carry the flat table.

**Cropping**: Three strategies with weighted random selection — contiguous (20%), spatial KD-tree (40%), interface-seeded (40%). Interface cropping seeds from tokens within 15 Å of another chain, then expands spatially.

**Preprocessing**: End-to-end recipe from a PDB mmCIF mirror to training-ready data:

```bash
# 1. Process structures: mmCIF → Structure NPZ + manifest
#    Filters: resolution ≤ 9 Å, max 300 chains, min 4 residues per chain
deepfold3 prepare --mmcif-dir /scratch/pdb/mmCIF --output /data/training \
    --max-resolution 9.0 --workers 16

# 2. Enrich manifest with release dates (needed for template date cutoff).
#    ``scan-mmcif`` defaults to ``--output data/mmcif_meta.json``, which is
#    also the default ``--mmcif-cache`` for ``prepare-templates`` and
#    ``enrich-manifest`` — just run them from the repo root and wiring
#    happens automatically.
deepfold3 scan-mmcif --mmcif-dir /scratch/pdb/mmCIF --data-dir /data/training --workers 32
deepfold3 enrich-manifest --data-dir /data/training --idx entries.idx

# 3. Apply sequence clustering (40% identity, wwPDB pre-computed)
wget -q https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt
deepfold3 apply-clusters --manifest /data/training/manifest.json \
    --cluster-file clusters-by-entity-40.txt --output /data/training/manifest.json

# 4. Generate unpaired MSAs (local MMseqs2, GPU-accelerated)
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ --gpu

# 5. Generate paired MSAs for multimer complexes
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ \
    --gpu --use-pairing

# 6. Discover monomer templates (per-chain m8 hits). Auto-uses
#    ``data/mmcif_meta.json`` from step 2 as the release-date source; without
#    it every hit gets ``release_date=None`` and is dropped at training time.
deepfold3 prepare-templates --data-dir /data/training \
    --m8-dir /data/msa/pdb/merged/m8 --max-templates 20 --workers 16

# 7. Drop obsolete PDB entries (optional, if using a stale mirror)
deepfold3 drop-obsolete --data-dir /data/training
```

See `docs/data-pipeline.md` for details on each step.

### Multi-Dataset Training

Train with multiple data sources (e.g. PDB experimental + AFDB distillation) by specifying extra datasets. Each dataset has its own `data_dir`, sampling weight, strategy, and distillation flag.

**Via CLI** (`--dataset name:path[:weight[:strategy[:distill]]]`):

```bash
# PDB (primary, cluster-weighted) + AFDB (distillation, uniform, 25% weight)
deepfold3 train --data-dir /data/pdb --output runs/exp1 \
    --dataset afdb:/data/afdb:0.25:uniform:true

# Three datasets
deepfold3 train --data-dir /data/pdb --output runs/exp1 \
    --dataset afdb:/data/afdb:0.25:uniform:true \
    --dataset esmfold:/data/esmfold:0.1:uniform:true
```

**Via JSON config** (`extra_datasets` field):

```json
{
  "training": {
    "extra_datasets": [
      {
        "name": "afdb",
        "data_dir": "/data/afdb",
        "weight": 0.5,
        "strategy": "uniform",
        "is_distillation": true
      },
      {
        "name": "pdb_highres",
        "data_dir": "/data/pdb_highres",
        "weight": 2.0,
        "strategy": "weighted",
        "is_distillation": false
      }
    ]
  }
}
```

**Fields per dataset:**

| Field             | Default     | Description                                                    |
| ----------------- | ----------- | -------------------------------------------------------------- |
| `name`            | (required)  | Identifier for logging                                         |
| `data_dir`        | (required)  | Path to `structures/` + `msa/`                                 |
| `weight`          | 1.0         | Relative sampling probability (primary dataset always 1.0)     |
| `strategy`        | `"uniform"` | `"weighted"` (cluster-balanced) or `"uniform"`                 |
| `is_distillation` | `false`     | If true, disables confidence/distogram losses for this dataset |

The primary dataset (`--data-dir`) always uses cluster-weighted sampling when cluster IDs are available. Extra datasets are combined via `ConcatDataset` + `MultiDatasetSampler`, which draws *records* from each dataset proportionally to weights (per-dataset probability × per-record cluster marginal); each `weighted` sub-dataset then draws its crop anchor at load time (the two-stage decomposition above), while `uniform` sub-datasets let the cropper self-seed.

**Resume-aware sampler state**: both `WeightedClusterSampler` (single-dataset path) and `MultiDatasetSampler` implement `state_dict` / `load_state_dict`, exposing `epoch + sample_idx + seed`. `CheckpointManager.save` snapshots the sampler under `blob["sampler"]`; on resume the iterator fast-forwards past indices already trained on, so within-epoch position is exact (no record duplication). `set_epoch` only resets `sample_idx` when the epoch number actually changes — the trainer's `set_epoch(state.epoch)` at the top of the loop does not clobber a freshly-loaded position. A seed mismatch between yaml and saved state emits a runtime warning (the per-epoch RNG derives from `seed + epoch`).

**Legacy compatibility**: `distillation_data_dir` and `distillation_weight` in `TrainingConfig` still work — they're internally converted to an `extra_datasets` entry.

### Sequence Clustering

Cluster-balanced sampling requires integer cluster IDs in the manifest. Two options:

#### **Option A: Download pre-computed wwPDB clusters**

```bash
wget https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt
```

Apply during prepare or to an existing manifest:

```bash
# During prepare
deepfold3 prepare --mmcif-dir /scratch/pdb/mmCIF --output /data/training \
    --cluster-file clusters-by-entity-40.txt

# Or apply to existing manifest (writes manifest_clustered.json by default)
deepfold3 apply-clusters --manifest /data/training/manifest.json \
    --cluster-file clusters-by-entity-40.txt --output /data/training/manifest.json
```

#### **Option B: Compute locally with MMseqs2**

Use `deepfold3 cluster` to extract sequences from processed structures and cluster with MMseqs2. Requires the `mmseqs` binary on PATH.

```bash
# Default: 40% identity, 80% coverage
deepfold3 cluster --data-dir /data/training --output clusters.txt

# Custom thresholds
deepfold3 cluster --data-dir /data/training --output clusters.txt \
    --min-seq-id 0.3 --coverage 0.8 --cov-mode 0

# Then apply to manifest
deepfold3 apply-clusters --manifest /data/training/manifest.json \
    --cluster-file clusters.txt --output /data/training/manifest.json
```

The `cluster` command extracts all unique polymer entity sequences from the `structures/` directory, runs `mmseqs cluster`, and writes a wwPDB-format cluster file (one cluster per line, space-separated `pdb_chain` identifiers).

Chains not matched to any cluster keep their SHA256 hash fallback IDs. The trainer detects integer cluster IDs and automatically activates `WeightedClusterSampler`; otherwise it falls back to uniform sampling.

### MSA Generation

Generate MSAs for all unique protein sequences in the training data:

```bash
# Search all unique sequences via local MMseqs2 (GPU-accelerated)
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ --gpu

# Resume-safe: re-running skips sequences that already have NPZ files.
# Sequences with A3M but no NPZ are converted without re-searching.

# Convert-only mode (A3M→NPZ, no search)
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ --convert-only
```

The command extracts unique protein sequences from the manifest, deduplicates by `msa_id` (SHA256 of sequence), searches against UniRef30 + ColabFoldDB, and writes per-sequence A3M + NPZ files to `{data_dir}/msa/`. Requires local MMseqs2 databases (see `scripts/setup_databases.sh`).

#### Paired MSAs for Multimers

For multimer complexes (>1 protein chain), paired MSAs provide cross-chain co-evolutionary signal. Add `--use-pairing` to the batch command:

```bash
# Generate unpaired + paired MSAs
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ \
    --use-pairing --gpu
```

This runs a two-phase pipeline: (1) unpaired search for all unique sequences, then (2) paired search per multimer record using MMseqs2's `pairaln` taxonomy-based pairing. Paired MSAs are written as `{record_id}_paired_{chain_idx}.npz` and referenced via `ChainInfo.paired_msa_id` in the manifest.

The featurizer merges paired and unpaired MSAs into a single full-width `[S, N]` tensor: paired rows first (cross-chain aligned), then dense unpaired rows (all chains' row _i_ on the same full-width row). See `docs/data-pipeline.md` § Paired MSA for the full row layout and comparison with OpenFold 3 and Protenix.

### Distillation Datasets (AFDB, ESMFold, etc.)

Distillation databases like AFDB provide predicted structures (CIF) alongside pre-computed MSAs (A3M). The pipeline handles these without re-searching:

```bash
# Option 1: Import during prepare (CIF + A3M together)
deepfold3 prepare --mmcif-dir /data/afdb/cifs --output /data/distill \
    --msa-dir /data/afdb/a3ms

# Option 2: Import A3Ms into an existing dataset
deepfold3 import-msa --data-dir /data/distill --msa-dir /data/afdb/a3ms

# Then search only the missing sequences (if any)
deepfold3 msa-batch --data-dir /data/distill --db-dir /data/mmseqs_dbs/ --gpu
```

Both `--msa-dir` and `import-msa` match A3M files to chains by reading the query sequence from each A3M file, computing its SHA256 hash, and copying to `{data_dir}/msa/{msa_id}.a3m`. This works regardless of file naming convention (AFDB `AF-{uniprot}-F1-model_v4.a3m`, PDB-style, or custom).

### Importing from OpenFold 3

OpenFold 3 (Apache 2.0, AlQuraishi Laboratory) stores preprocessed structures as biotite AtomArray NPZ files. These can be converted to DeepFold3 format:

```bash
# Convert all OF3 preprocessed structures
deepfold3 import-of3 --of3-dir /data/of3/structures --output /data/training

# With OF3 dataset cache (provides resolution, cluster IDs, metadata)
deepfold3 import-of3 --of3-dir /data/of3/structures --output /data/training \
    --cache /data/of3/dataset_cache.json

# Then generate MSAs for the imported data
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ --gpu
```

The converter (`data/prepare/openfold3.py:convert_of3_atom_array`) handles mol-type mapping (OF3 uses `PROTEIN=0, RNA=1, DNA=2, LIGAND=3` while DeepFold3 uses `PROTEIN=0, DNA=1, RNA=2, NONPOLYMER=3`), bond remapping, and entity/chain grouping. Three details of the OF3 atom array decide whether the result is correct:

- **`is_atomized` is authoritative for `is_standard`.** OF3 stores its own tokenization decision per atom — True for ligands *and* for modified / non-standard polymer residues. Re-deriving it from the residue name instead maps modified residues onto their standard parent (`get_res_type("SEP") → SER`), which emits one SER token over SEP's atom set and then slots those atoms against SER's reference atoms. Verified on the upstream smoke set: token counts come out at OF3's documented values (7fb8 → 53, 7ohe → 24, 7vus → 87).
- **Unresolved atoms carry `NaN` coordinates.** OF3 materialises every polymer residue's full reference atom set whether or not it was observed. DeepFold3's convention is coordinates `0` + `is_present=False`, so the NaNs are zeroed at conversion — a NaN reaching the cropper poisons every distance it takes part in and surfaces only as a NaN loss with no offending atom named.
- **`chain_id` is OF3's chain *key*, a string of an integer (`"1"`, `"3"`) — not a chain letter.** It is the join key into the dataset cache's per-structure `chains` dict; `label_asym_id` carries the letter and becomes DeepFold3's `chain_name`. `convert_of3_atom_array` returns the keys in `asym_id` order as its third element.

Supported OF3 directory layouts:

- Nested: `{pdb_id}/{pdb_id}.npz`
- RODA monomer: `{id}/structure.npz`
- Flat: `{pdb_id}.npz`

### Training on OpenFold 3's `pdb_training_set` in place (`format: of3pdb`)

`import-of3` bulk-converts structures; `format: of3pdb` instead reads an `aws s3 sync s3://openfold3-data/pdb_training_set/` mirror **as-is** and is the AF3 weighted-PDB training set. Nothing is re-preprocessed — structures, MSAs and templates are read from the upstream tree at `__getitem__` time and fed through DeepFold3's own tokenize → crop → featurize pipeline.

The one offline step is the **dataset cache**. Upstream ships it as a ~1.7 GB JSON and reads it either whole (tens of GB resident) or through an LMDB store; neither is acceptable per DDP rank × dataloader worker, and LMDB is not a dependency we want in the data path. `deepfold3 build-of3-cache` streams that JSON once (`data/prepare/of3_pdb_cache.py`, bounded memory — the reader is verified byte-exact against `json.load`) into a directory of plain `.npy` **columns** that `OF3PdbCache` opens with `mmap_mode="r"`: startup is a handful of `open()` calls, resident cost is only the pages touched, and forked workers share them through the page cache. ~50 MB of columns for the full set; the validation cache converts in 0.1 s.

```bash
deepfold3 build-of3-cache \
    --cache-file /data/pdb_training_set/dataset_caches/training_cache_with_templates.json \
    --output     /data/pdb_training_set/df3_cache
```

```yaml
datasets:
  - name: of3_pdb
    data_dir: /data/pdb_training_set          # the set's ROOT
    of3_cache_dir: /data/pdb_training_set/df3_cache   # default: {data_dir}/df3_cache
    weight: 1.0
    strategy: weighted
    format: of3pdb
    cutoff_date: "2021-09-30"
```

What the dataset reads, and from where:

| Artefact | Path under `data_dir` | Notes |
| --- | --- | --- |
| Structure | `preprocessed_pdb_data/standard/structure_files/{id}/{id}.npz` | biotite `AtomArray`, converted per sample |
| MSA | `alignment_arrays/{alignment_representative_id}.npz` | one block per search database; the representative is a *sequence-identical* chain, often in another entry |
| Template hits | `templates/train_template_cache/{rep}.npz` | override with `template_cache_dir`; falls back to `val_template_cache` only when the training one is absent |
| Template geometry | `templates/template_structure_arrays/{pdb}/{pdb}_{chain}.npz` | see below |

Points that differ from `format: df3`:

- **Sampling is AF3 §2.5.1 computed on the cache's columns** (`of3_anchor_weights`), not on `Record` objects: the sampler draws a structure by the cluster marginal and the dataset draws that structure's chain / interface crop anchor at load time — the same two-stage decomposition `data/sample/weighted.py` applies, and a unit test asserts the two agree anchor for anchor. `N_clust` is recounted over the structures that survive `cutoff_date` rather than read from upstream's frozen `cluster_size`. The per-item distribution reaches `MultiDatasetSampler` through its `weights` key.
- **Templates need no mmCIF mirror and no realignment.** OF3 renders each template chain over its full SEQRES with `res_id` as the SEQRES index, so a hit's `idx_map` (1-based query position, template `res_id`) addresses it directly. `mmcif_dir` is not consulted and `template_struct_space` does not apply. Frame verified by re-deriving hit identity: 99.3% on a close homolog (`5enn_A` ← `3ls8_A`), and 0.45 pooled over remote validation hits versus 0.07–0.11 for every off-by-one alternative.
- **MSA blocks each repeat the query as row 0**, so they cannot simply be stacked; `build_chain_msa` concatenates in OF3's `aln_order`, deduplicates, and keeps the query exactly once. `uniprot_hits` is held out for **species pairing** (`pair_uniprot_msas`, AF2-Multimer style: rank *r* of species *s* pairs across chains, species seen by one chain only are dropped, rows covering more chains sort first). Chains sharing a representative decode the file once, and the UniProt block is skipped entirely when the crop has fewer than two distinct representatives (that alone took a monomer sample from 454 ms to 278 ms). **DNA chains have no `alignment_representative_id` at all** — OF3 searches proteins and RNA only — so, as in the df3 path, any polymer chain without an alignment falls back to a single-sequence MSA read off its own tokens; without it the featurizer would leave the whole chain at `MSA_GAP` and the model would never see its query sequence.
- **Crop-then-load ordering** as in `TrainingDataset`: tokenization never reads MSA content, so alignments and templates are only read for the chains that survive the crop.
- **Templates need both halves, and say so.** The hit lists (`templates/*_template_cache/`) and the geometry (`templates/template_structure_arrays/`) are separate directory trees, and `aws s3 sync` lands them in whatever order. If the geometry is missing the dataset logs a warning **once at construction** and disables templates outright — previously every hit resolved to a missing file, got skipped at debug level, and a half-synced mirror was indistinguishable in the logs from a deliberately template-free run.
- **Date filtering is fail-closed on both ends.** `cutoff_date` drops structures with no release date in either direction, and the template cap is `min(max_template_date, query_release − template_cutoff_days)` — when neither can be established, every template is dropped rather than used unfiltered.
- **An eval split needs its own two pointers.** The same tree serves both splits, but the dataset cache and the template hit lists are per-subset:

  ```yaml
  eval_datasets:
    - name: of3_pdb_eval
      data_dir: /data/pdb_training_set
      of3_cache_dir: /data/pdb_training_set/df3_cache_val
      template_cache_dir: /data/pdb_training_set/templates/val_template_cache
      format: of3pdb
      cutoff_date: "2021-09-30"
  ```

  Eval specs keep records released **after** `cutoff_date`, and the training cache ends the day before it (180,975 entries, all ≤ 2021-09-29), so leaving `of3_cache_dir` on the training conversion yields zero records — the dataset is dropped with a `no records in …, skipping.` warning and, if it was the only eval dataset, `eval_dl` is `None` and eval stops happening at all. `template_cache_dir` matters for the same reason in a quieter way: both hit-list directories exist on a synced mirror, so the constructor's fallback never fires and the val split resolves hits for 155/833 of its chain representatives instead of 640/833.

The featurizer produces these training-specific features:

- `atom_gt_positions [N, M, 3]`: experimentally observed atom coordinates.
- `atom_gt_mask [N, M]`: which atoms have resolved positions.

These are separate from `ref_pos` (CCD ideal coordinates used for atom cross-attention conditioning).

## Optimizer

AdamW (or Schedule-Free AdamW when `optimizer_type="schedule_free"`) with **decoupled weight decay split into two groups** (`build_optimizer` in `training/optim.py`):

- **No-decay** group (`weight_decay=0`):
  - 1D params (LayerNorm γ/β, biases) and 2D weights with `min(shape)==1` (per-channel scale-like projections).
  - **Gates**: `to_g`, `out_gate`, `.gate.`, AdaLN-zero `cond_scale` / `cond_bias` / `zero_cond` (incl. `attn_cond_scale.{i}`, `k_cond_scale.{i}`, `ffn_cond_scale.{i}`, `*_cond_bias.{i}` `ModuleList` variants in the atom transformer). Decay would pull sigmoids toward 0.5 (always-on), erasing the conditional behaviour.
  - **Input embeddings**: `single_activations`, `msa_activations`, `extra_msa_target_feat`, `{left,right}_single`, `position_activations`, `bond_embedding`, `prev_embedding`, `prev_single_embedding`, plus per-atom and template `embed_*` linears. No preceding LN, so decay shrinks the input representation directly.
- **Decay** group (`weight_decay=cfg.weight_decay`): all remaining 2D+ matmul weights — attention QKV / output projections, FFN linears, triangle multiplication output projections, OPM `out_w`, etc.

See `_NO_DECAY_NAME_SUBSTRINGS` in `training/optim.py` for the exact pattern list.

## Checkpointing

Checkpoints are saved at configurable intervals (default: every 1000 steps) and contain:

- Model state dict (**excludes** fused kernel derived params — see below)
- Optimizer state dict
- LR scheduler state dict
- EMA state (if enabled, also excludes fused params)
- Step counters
- Config (model + training) — `model_cfg` and `train_cfg` are pickled into the blob and used at inference time to rebuild the model with the exact architecture flags (e.g. `pairformer.residual_output_scaling`) that were active at training. See _Configuration persistence_ below.

**Checkpoint rotation**: keeps the `max_checkpoints` most recent checkpoints by modification time. Default is `0`, which disables rotation (all checkpoints are kept). `checkpoint_final.pt` and `checkpoint_best.pt` are always preserved.

Resume training by passing `--resume path/to/checkpoint.pt`.

**LR override on resume**: `optimizer.load_state_dict` and `LambdaLR.load_state_dict` both restore the saved `lr` / `initial_lr` / `base_lrs` verbatim, which silently shadows any `--config` change. The trainer's `[resume]` block detects this and force-overrides `param_groups['lr','initial_lr']` + `scheduler.base_lrs` to the configured value, so `--config new.yaml --resume ...` honors the new lr. Schedule-Free AdamW's accumulator state (`z` buffer, `lr_max`, `weight_sum`) is preserved across the override. A `[resume] LR override: <saved> → <target>` line is logged when this fires.

**Schedule override on resume**: `LambdaLR.state_dict()` is `self.__dict__` minus `optimizer` / `lr_lambdas`, so every schedule field the scheduler stores as an attribute (`max_lr`, `warmup_steps`, `start_decay_after`, `decay_every`, `decay_factor`, `finetune_max_steps`, `finetune_min_lr_ratio`, `finetune_decay_start`, the decay shape) is baked into the checkpoint and restored verbatim — a `--config` change to any of them used to be silently ignored on resume. The trainer now forces the yaml value back in (list: `_SCHEDULER_CFG_FIELDS` in `trainer.py`, shared with the audit below) and logs `[resume] scheduler schedule override (yaml wins over checkpoint): <field>: <saved> → <target>` for each one. `last_epoch` / `_step_count` are untouched (the counter must stay aligned with `global_step`) and `base_lrs` is owned by the LR override above. `wsds_decay_windows` is not forced — rebuilding the sorted/validated window list belongs in the constructor — so it still only warns. `WSDSScheduler.min_ratio` maps to `wsds_min_lr_ratio` (`_scheduler_cfg_fields`); before that it was overwritten with the cosine floor on every WSDS resume.

**Resume audit (warn-only)**: every other yaml-controlled field that is silently shadowed by saved state — `weight_decay` (per param-group), Adam `betas` / `eps`, `ema_decay`, `wsds_decay_windows` — is compared against the live config and logged as `[resume] yaml fields silently shadowed by checkpoint state` when divergent. The audit is read-only. Fix the yaml or accept the saved value (most fields can also be re-injected via `--override`).

**Partial loading** for stage transitions: use `--params` with a previous stage's checkpoint plus config overrides. Control what gets loaded via `load_strict` (default true), `skip_load_optimizer`, `skip_load_scheduler`, `skip_load_step`.

**Post-write I/O** runs on a single background thread (`max_workers=1`), drained by `wait_for_io` before the next save and at shutdown:

- **Always**: `_flush_and_evict` — `fsync` then `fadvise(DONTNEED)`. Nothing on the node reads a checkpoint again, so several GB per save is pure cgroup pressure, and it lands on rank 0's node only — the shape of a single-node stall. Both halves must be off the training thread: `DONTNEED` only drops *clean* pages so it needs the `fsync` first, and an inline `fsync` would serialize the multi-GB shared-storage write into the step while every other rank waits at the next collective. Once `torch.save` returns the blob is complete, so the thread needs only the path.
- **With `checkpoint_staging_dir`** (default `""`): the blocking write goes to node-local disk instead, and `_publish` moves the shared-storage copy to the same thread (`copyfile` → flush/evict → atomic `os.replace` → rotation). Do **not** point this at a memory-backed mount (`/dev/shm`) — that charges the checkpoint to the memory cgroup, which is the pressure staging exists to relieve.

The read side is symmetric: `_evict_after_read` drops the blob after `torch.load` / `load_file`. Every rank on a node reads its own copy on resume (8× the blob size per node), immediately before the trainer forks dataloader workers.

**Fused kernel portability**: `DeepFold3.state_dict()` automatically filters out derived fused kernel parameters (identified by `OptimConfig.FUSED_STATE_DICT_MARKERS`). On load, `load_state_dict()` re-syncs fused params from base weights. This means checkpoints saved with fused kernels enabled can be loaded without them, and vice versa.

### Configuration persistence

A fresh `ModelConfig()` at inference does **not** match a checkpoint trained with non-default flags (`pairformer.residual_output_scaling`, `pairformer.{single,pair}_transition_residual_gain`, layer counts, …). These flags often only change a forward-pass scalar, so `load_state_dict(strict=False)` reports `missing=0, unexpected=0` and the silent failure is a wrong prediction.

The checkpoint pipeline embeds the architectural `ModelConfig` on both sides of the format:

- **`.pt` training checkpoints**: `CheckpointManager.save` already pickles `model_cfg`. `deepfold3 inference --params <ckpt>.pt` deep-copies it out of the blob and uses it as the base.
- **`.safetensors` files**: `convert_weights.py` and any future writer embed a JSON-serialised `ModelConfig` under the `model_cfg_json` metadata key (`SAFETENSORS_MODEL_CFG_KEY`). `deepfold3 inference --params <weights>.safetensors` reads the metadata via `safe_open(...).metadata()` (header-only, cheap) and reconstructs the dataclass via `model_cfg_from_metadata`. Forward-compatible: unknown future fields are silently dropped.

The `--optim` flag decides only `OptimConfig`, never the architectural fields; `train-matched` keeps the `OptimConfig` the checkpoint recorded. Legacy `.safetensors` (no metadata) and raw AF3 `.bin` / `.bin.zst` paths fall back to `ModelConfig()` defaults with a one-line warning.

## Training Stages

Stages are controlled by config overrides — no hardcoded presets. Use `--config` with stage JSON files or `--override key=value`.

AF3 trains in 4 stages (Table 6 of supplementary). Each loads weights from the previous stage.

### Stage 1: Initial Training

```json
{
  "alpha_diffusion": 4.0,
  "alpha_pae": 0.0,
  "alpha_bond": 0.0,
  "diffusion_batch_size": 48,
  "max_tokens": 384
}
```

### Fine-tune 1: Structure (crop 640)

```json
{
  "alpha_bond": 1.0,
  "alpha_smooth_lddt": null,
  "diffusion_batch_size": 32,
  "max_tokens": 640
}
```

Bond loss on, smooth LDDT off. Disorder PDB distillation weight reduced to 0.01; masked diffusion for non-protein disabled. Load Stage 1 checkpoint with `--params`.

### Fine-tune 2: Structure (crop 768)

```json
{ "alpha_bond": 1.0, "diffusion_batch_size": 32, "max_tokens": 768 }
```

Smooth LDDT back on (default). Transcription factor distillation sets enabled. Disorder PDB weight back to 0.02. Load Fine-tune 1 checkpoint with `--params`.

### Fine-tune 3: Confidence

```json
{
  "train_confidence_only": true,
  "alpha_diffusion": 0.0,
  "alpha_distogram": 0.0,
  "alpha_pae": 1.0,
  "diffusion_batch_size": 32,
  "max_tokens": 768
}
```

Freezes all parameters except confidence head. Only confidence losses (pLDDT, PDE, PAE, exp_resolved) are active. Max chains increased to 50. Load Fine-tune 2 checkpoint with `--params`.

## AF3 Data Pipeline Conformance

Systematic comparison of DeepFold3's data pipeline against the AF3 supplementary material (§2 + §5). See `docs/af3-paper-reference.md` for the full organized reference.

### ✅ Matching

| Paper section | Item                                                        | Implementation                                                                                        |
| ------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| §2.5.1        | Weighted PDB sampling (β/α weights, chain+interface)        | `data/sample/weighted.py`                                                                             |
| §2.5.3        | Protein 40% clustering                                      | wwPDB clusters via `data/prepare/clustering.py`                                                       |
| §2.5.3        | 100% clustering for nucleic acids + peptides                | Hash-based cluster_id = 100% identity; `scripts/prepare.py:_apply_clusters`                           |
| §2.5.3        | CCD identity clustering for ligands                         | `ChainInfo.res_name` + `_apply_clusters` maps `ccd_{res_name}`                                        |
| §2.5.4        | Resolution ≤ 9Å, max 300 chains, min 4 residues             | `data/prepare/filters.py`                                                                             |
| §2.5.4        | Clashing chains (30%/1.7Å, iterative 3-level tie-break)     | `data/prepare/filters.py:filter_clashing_chains`                                                      |
| §2.5.4        | Consecutive Cα > 10Å                                        | `data/prepare/filters.py`                                                                             |
| §2.6          | Standard AA/nucleotide = 1 token, ligand = per-atom         | `data/tokenize/tokenizer.py`                                                                          |
| §2.7          | Cropping weights (0.2/0.4/0.4), interface 15Å               | `data/crop/cropper.py`                                                                                |
| §2.8          | ref_pos (random rot+trans), ref_element (one-hot 128)       | `data/feature/featurizer.py`                                                                          |
| §2.8          | token_bonds filtered to < 2.4Å during training              | `data/feature/featurizer.py:_build_bond_features`                                                     |
| §3.7.1        | Diffusion batch 48, MSE mol-type weight (5×/10×)            | `model/config.py`, `training/loss.py`                                                                 |
| §3.7.1        | EDM weighting, smooth LDDT (15/30Å radii)                   | `training/loss.py`                                                                                    |
| §3.7.1        | Bond loss (Eq. 5 — GT distances)                            | `training/loss.py:bond_loss`                                                                          |
| §3.7.1        | Cross-entity bonds (disulfide, metal, covalent)             | `data/prepare/structure.py` extracts from `_struct_conn`; `tokenizer.py` propagates with atom indices |
| §2.5.4        | >20 polymer chains → greedy select 20 maximising interfaces | `data/prepare/filters.py:select_interface_chains`                                                     |
| §2.5.3        | Modified residue normalization for clustering               | `data/prepare/structure.py:_chain_sequence` uses `_NONSTANDARD_TO_STANDARD`                           |
| §2.5.4        | CCD atom name validation                                    | `data/ccd.py:build_atom_name_cache` + `process_mmcif(ccd_atom_names=...)`                             |
| §2.5.4        | Leaving atom removal                                        | `process_mmcif(ccd_leaving_atoms=...)` parses `pdbx_leaving_atom_flag` from `_struct_conn`            |
| §2.6          | Modified residue per-atom tokenization                      | `data/tokenize/tokenizer.py` checks `is_standard`; non-standard → per-atom                            |
| §5.3          | Loss weights (α_diff=4, α_disto=0.03, α_conf=1e-4)          | `model/config.py`                                                                                     |
| §5.4          | Adam β₁=0.9 β₂=0.95 ε=1e-8, LR 1.8e-3, warmup 1000          | `model/config.py`                                                                                     |
| §5.4          | LR decay 0.95/50k, grad clip 10, batch 256                  | `model/config.py`, `training/trainer.py`                                                              |

### ⚠️ Divergences

| Paper section | Item               | Our approach                                        | Impact                                    |
| ------------- | ------------------ | --------------------------------------------------- | ----------------------------------------- |
| §3.7.1        | Smooth LDDT weight | Now configurable via `alpha_smooth_lddt` (None=off) | AF3 turns it off in fine-tune 1 (Table 6) |
| §4.2          | Symmetric (ambiguous) atom resolution | Hardcoded per-residue swap table (`data/symmetry/atom_perms.py`: ASP/GLU/ARG/PHE/TYR/LEU/VAL + nucleic OP1/OP2). **TODO: replace with RDKit conformer automorphism per chemical component** — `mol.GetSubstructMatches(mol, uniquify=False, useChirality=False, maxMatches=1000)` → permute GT to min-RMSD, exactly AF3 §4.2 / OF3 `get_cropped_permutations` ([conformer.py:384](../references/openfold-3/openfold3/core/data/primitives/structure/conformer.py#L384)). | Table ≡ RDKit for the 7 standard residues; RDKit additionally covers modified/non-standard residues and 3-fold+ symmetries (e.g. 5′-terminal phosphate OP1/OP2/OP3) generically, and drops the manual HIS pseudo-symmetry that was removed to match AF3. |

### ❌ Missing (known gaps)

| Paper section | Item                                                               | Notes                                                  |
| ------------- | ------------------------------------------------------------------ | ------------------------------------------------------ |
| §5.2, Table 6 | Masked diffusion loss for non-protein in disorder PDB distillation | Not applicable — diffusion without masking is intended |

### Stage Configuration Quick Reference

From AF3 Table 6. Use `--config` with stage JSON files or `--override`:

```bash
# Stage 1: Initial training (crop 384, diffusion batch 48, max_chains 20)
deepfold3 train --data-dir /data --output runs/s1 --config configs/initial.json

# Fine-tune 1: crop 640, bond loss on, smooth LDDT off, diffusion batch 32
deepfold3 train --data-dir /data --output runs/ft1 --params runs/s1/checkpoint_final.pt \
    --config configs/finetune1_structure_640.json

# Fine-tune 2: crop 768, smooth LDDT back on, TF distillation sets on
deepfold3 train --data-dir /data --output runs/ft2 --params runs/ft1/checkpoint_final.pt \
    --config configs/finetune2_structure_768.json

# Fine-tune 3: PAE only, max chains 50
deepfold3 train --data-dir /data --output runs/ft3 --params runs/ft2/checkpoint_final.pt \
    --config configs/finetune3_confidence.json
```

## Source Layout

```
src/deepfold3/training/
├── __init__.py       # Package exports
├── loss.py           # Loss functions (smooth LDDT, distogram, confidence)
├── ema.py            # Exponential Moving Average
├── scheduler.py      # AF3 LR scheduler (warmup → plateau → decay)
└── trainer.py        # DDP trainer (training loop, checkpointing, eval)
```

## Module Dependencies

| Module         | Depends On                                           |
| -------------- | ---------------------------------------------------- |
| `loss.py`      | `model.config.TrainingConfig`                        |
| `ema.py`       | `torch.nn`, `model.config.OptimConfig`               |
| `scheduler.py` | `torch.optim`, `model.config.TrainingConfig`         |
| `trainer.py`   | All above + `model.model.DeepFold3`, `data.module.*` |
