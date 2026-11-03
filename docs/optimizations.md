# Performance Optimizations

DeepFold3 supports optional Triton kernel replacements and `torch.compile` for key
operations. All Triton kernels are self-contained under `src/deepfold3/model/triton/`.

Optimizations are toggled via `OptimConfig` — each can be independently enabled or disabled.

## B200 End-to-End Benchmark (bf16, 2026-06-10)

The RTX-5090 section below is **forward-only** and does not reflect training, where
each kernel also runs a backward and is recomputed under gradient checkpointing — so
a kernel that wins on the forward can be net-neutral or slower in a training step.
The numbers here are the real training/inference regime on B200, measured with
`scripts/bench_kernels.py` (8 `PairFormerBlock`s — the trunk component that hosts
these kernels). **These are the basis for the finalized flags.**

Baseline = fused evo-attention + fused dropout-add on, the four candidates off.
Speed is ×baseline; memory is Δ vs baseline (8-block trunk peak).

### Training (fwd + bwd, gradient checkpointing)

| Config | N=384 | N=640 | N=768 |
| --- | --- | --- | --- |
| baseline | 1.00× / 3.18 GB | 1.00× / 8.72 GB | 1.00× / 12.52 GB |
| +swiglu | 1.05× / −0.14 | 1.04× / −0.39 | 1.03× / −0.56 |
| +transition_pair | 1.05× / −0.21 | 1.04× / −0.59 | 1.04× / −0.84 |
| +triangle_mul | 1.05× / −0.42 | 1.05× / −1.17 | 1.06× / −1.69 |
| +pair_bias | 1.00× / +0.07 | 1.07× / +0.20 | 1.12× / +0.29 |
| **+3 safe (no pair_bias)** | **1.04× / −0.63** | **1.09× / −1.76** | **1.09× / −2.54** |
| +ALL four | 1.01× / −0.56 | 1.17× / −1.56 | 1.25× / −2.25 |

- **Training default = the safe trio (swiglu + transition_pair + triangle_mul)**, enabled
  in `configs/initial.yaml`: 1.04–1.09× with a ~20% trunk-activation memory cut that
  grows with crop size. Numerics validated (`tests/triton/test_fused_modules_robustness.py`
  46/46; full-model e2e fwd+bwd, 0 non-finite grads).
- **`pair_bias` stays OFF for training.** At N=384 it adds memory for ~0% speed, and its
  bf16 backward amplifies gradients ~1e11× alongside fused evo-attn (the load-bearing
  reason — see `configs/initial.yaml`). But it is a large speed lever at big crops
  (all-four 1.25× @768 vs 1.09×), so **rewriting its fp32-promoted backward is worth
  ~12–16% throughput if training moves to 640/768 crops.**

### Inference (fwd only)

| Config | N=384 | N=640 | N=768 |
| --- | --- | --- | --- |
| baseline | 52.98 ms / 0.9 GB | 164.85 ms / 2.4 GB | 257.34 ms / 3.4 GB |
| +pair_bias | 41.65 ms (1.27×) | 133.11 ms (1.24×) | 202.77 ms (1.27×) |
| +3 safe | 46.95 ms (1.13×) | 146.50 ms (1.13×) | 228.21 ms (1.13×) |
| **+ALL four** | **35.58 ms (1.49×) / 0.7 GB** | **114.78 ms (1.44×) / 1.9 GB** | **173.46 ms (1.48×) / 2.7 GB** |

`OptimConfig.inference()` already enables all four + `ln_transpose` + tiling — confirmed
optimal (1.44–1.49× and lower memory). With no backward, `pair_bias` is safe and is the
single biggest inference lever (1.24–1.27× alone).

### fused_dropout_add (default ON)

Per-call it is marginally slower than unfused PyTorch (72 vs 70 µs at the 384 pair shape)
but never materializes the `[N, N]` mask (~36 GB/step saved across fwd+bwd). The mask seed
is drawn from a dedicated **rank-aware** generator (`_next_mask_seed`) so the ~244 draws/step
neither pollute the global CPU RNG stream nor correlate masks across DDP ranks; a CPU draw
(~2 µs) is cheaper than an on-GPU seed (~10 µs, launch-bound), so the seed is never a
bottleneck.

## Diffusion sampler (`DiffusionConfig.hoist_sampler`, 2026-10-31)

`DiffusionHead.hoist_static` replaces `DiffusionStatic.expand_for_samples` in the
sampler. On for `deepfold3 inference` (`--no-hoist-sampler` disables it), off everywhere
else, including `inference-batch` and the training mini-rollout, whose path stays
byte-identical to the plain sampler. It does three things:

- **Drops empty atom windows.** `build_atom_subsets` sizes `S` from the `N*M` capacity so
  training shapes are data-independent, which leaves 291 of 425 windows empty at 566
  tokens / 4280 atoms. They were the largest single waste in the step.
- **Computes step-invariant terms once:** token pair logits, atom pair logits, atom AdaLN
  scale / bias / gates. The padding and key masks are folded into the logits.
- **Keeps one copy of the state.** Sample-shared tensors are views or broadcast, and the
  token `single_cond` is computed for one row per step.

Measured on an RTX 6000 Ada, 7st5 (566 tokens, 5 samples, 200 steps), checkpoint 5500,
fp32 + TF32, warm (first pass per arm discarded):

| Guidance | Plain | Hoisted | Speedup | Peak plain → hoisted |
| --- | --- | --- | --- | --- |
| none | 6.89 s/sample | 2.43 s/sample | 2.84× | 7.61 → 5.22 GiB |
| 1 term | 13.67 s/sample | 4.90 s/sample | 2.79× | 10.58 → 6.65 GiB |
| 2 terms | 20.46 s/sample | 7.38 s/sample | 2.77× | 12.80 → 7.77 GiB |
| 2 terms, one σ̂-limited | 16.51 s/sample | 5.91 s/sample | 2.79× | 12.80 → 7.77 GiB |

Peaks include 1.83 GiB of head + inputs. The gain shrinks with size: 1.16× at 64 tokens,
2.18× at 128, 2.90× at 256, 3.08× at 384 (crops of the same input, 2 terms). Each
guidance term still costs one denoiser pass per step, plus 1.1–1.4 GiB of hoisted state
(2.2–3.0 GiB on the plain path). Nothing in the path catches an OOM.

**Equivalence.** The noise and augmentation draws are the plain sampler's exactly. The
denoiser is not bit-identical in general, because cuBLAS TF32 results depend on GEMM
shape and window trimming changes those shapes. For the same reason the plain denoiser
disagrees with itself across batch sizes, so this is its existing tolerance:

| One denoiser call, σ = 100, max coordinate difference | `high` (TF32, CLI default) | `highest` (fp32) |
| --- | --- | --- |
| plain vs hoisted | 0 or 0.09–0.14 Å, depending on shape | 0.9–2.6e-4 Å |
| plain S=5 vs plain S=1, same sample | 0.07–0.11 Å | 0.9–2.6e-4 Å |

After 200 steps under `high`, aligned per-sample RMSD to the plain sampler is 0–0.09 Å
(a different seed: 0.4–8.9 Å), and 566 tokens / 5 samples happened to come out exactly
equal. Under `highest` it is 1e-5–6e-5 Å. Do not expect `--no-hoist-sampler` to reproduce
a hoisted run's files byte for byte.

**Measured and not done:**

- *CUDA graph for the step:* replay matches eager, 167 vs 166 ms plain and 60.6 vs 61.3 ms
  hoisted. The step is GPU-bound at this size, and a graph adds capture and invalidation
  risk for ~1%.
- *One denoiser call for the conditional and unconditional passes:* slower, not faster:
  553 ms for one S=15 call vs 514 ms for three S=5 calls, at 21.18 GiB peak vs 12.80.
- *Host syncs:* none per step; the 7 found are all in setup.
- *Vectorised augmentation:* 0.76 ms per step, 1.2% of the hoisted step.

## Benchmark Results (RTX 5090, bf16, 2026-03-16)

### Winners (2026-03 measurements; the torch.compile rows are in no current preset)

| Optimization                        | Speedup (fwd)             | Memory Savings                     | Where Used                                      | Count       |
| ----------------------------------- | ------------------------- | ---------------------------------- | ----------------------------------------------- | ----------- |
| **Triton EvoAttention (Grid)**      | **2.9-3.3x**              | **3.4-4.6x**                       | EvoformerBlock + PairFormerBlock pair attention | 104 attn    |
| **Triton FusedPairBias**            | **+25-30%** on top of evo | tl.dot GEMM (5-10x vs scalar loop) | GridSelfAttention pair bias                     | 104 attn    |
| **Combined evo+pair_bias**          | **3.7-4.6x**              | **3.4-4.6x**                       | GridSelfAttention (row + col)                   | 104 attn    |
| **Triton FusedTriangleMul (L3)**    | 1.24-1.32x                | Symmetric memory (both directions) | TriangleMul outgoing + incoming                 | 96 modules  |
| **torch.compile (MSAAttention)**    | 1.5-2.3x                  | 1.6-2.3x                           | EvoformerBlock (×4)                             | 4 modules   |
| **torch.compile (TriangleMul)**     | 1.25-1.55x                | ~1.0x                              | EvoformerBlock + PairFormerBlock                | 104 modules |
| **torch.compile (SingleAttention)** | 1.3-1.4x                  | ~1.0x                              | PairFormerBlock (×48)                           | 48 modules  |

### Grid Column Attention (evo + pair bias)

| N   | pytorch         | evo           | evo+pb            | Speedup         | Mem reduction |
| --- | --------------- | ------------- | ----------------- | --------------- | ------------- |
| 384 | 5.8ms / 2.8GB   | 2.0ms / 0.9GB | **1.6ms / 1.0GB** | **3.7x / 2.9x** |
| 512 | 13.3ms / 6.5GB  | 4.3ms / 1.6GB | **3.2ms / 1.7GB** | **4.1x / 3.8x** |
| 640 | 25.4ms / 12.3GB | 7.7ms / 2.5GB | **5.5ms / 2.6GB** | **4.6x / 4.7x** |

### Conditional winners

| Optimization                       | Speedup                     | Condition                                                          |
| ---------------------------------- | --------------------------- | ------------------------------------------------------------------ |
| **Triton SwiGLU (pair, M=N²)**     | 1.4-1.6x fwd, 1.9-2.1x full | Only at pair-transition scale (M=N²); single transitions too small |
| **Triton FusedTransition (C=384)** | 1.2-1.3x full               | Only beneficial at C≥256 and N≥1024; smaller shapes are slower     |
| **Triton LayerNormLinear (C=384)** | 1.1-1.2x full               | Composited inside FusedTransitionBlock; marginal standalone        |
| **Triton FusedLNTranspose**        | 1.7-2.3x fwd                | Column attention LN+transpose; inference only                      |

### Not recommended (removed from config)

| Optimization                       | Result                | Reason                                                                       |
| ---------------------------------- | --------------------- | ---------------------------------------------------------------------------- |
| Triton EvoAttention (Single)       | 0.87-1.03x (slower)   | N=1 sequence dim → kernel overhead dominates; torch.compile is better        |
| torch.compile (LayerNorm)          | 0.3x (3x slower)      | Inductor overhead dominates tiny kernel                                      |
| torch.compile (LinearNoBias)       | 0.3-0.5x (slower)     | cuBLAS already optimal                                                       |
| torch.compile (TransitionBlock)    | 0.75-0.98x (slower)   | Overhead > gain for most shapes                                              |
| torch.compile (AdaTransitionBlock) | 0.96-1.02x (neutral)  | No meaningful benefit                                                        |
| torch.compile (OuterProductMean)   | 1.06-1.25x (marginal) | Minimal speedup, not worth compile cost                                      |
| Triton FusedTransition (C=64/128)  | 0.8-1.0x (slower)     | Shapes too small for Triton overhead                                         |
| Triton SwiGLU (single, M=N)        | 0.5x (slower)         | Kernel launch overhead dominates at tiny M; use only inside pair transitions |

### Key Insight

The **dominant optimizations** are:

1. **Triton EvoAttention + FusedPairBias** for GridSelfAttention — **3.7-4.6x faster** with
   **3-5x less memory**. This covers 104 pair attention modules (the N² bottleneck).
   FusedPairBias now supports training (autograd backward) in addition to inference.

2. **Triton FusedTriangleMul (L3)** for TriangleMul — **1.2-1.3x faster** with symmetric
   memory usage for both Incoming and Outgoing directions.

`torch.compile` helps for MSAAttention (1.5-2.3x), TriangleMul (1.3-1.6x), and
SingleAttention (1.3-1.4x) but hurts for small primitives (LN, Linear) and transitions.

**Important**: Triton EvoAttention is NOT beneficial for SingleAttention (N_seq=1 makes
the 5D kernel overhead dominate). Use torch.compile for SingleAttention instead.

## Available Backends

| Backend                        | Module                    | Replaces                                                  | Source                            |
| ------------------------------ | ------------------------- | --------------------------------------------------------- | --------------------------------- |
| Triton FusedEvoAttention       | `_EvoformerAttention`     | Flash-attention with pair bias                            | `triton/evo_attention.py`         |
| Triton FusedPairBias           | `FusedPairBias`           | LN + Linear + permute for pair bias (tl.dot GEMM)         | `triton/fused_pair_bias.py`       |
| Triton FusedGatedDualGEMM      | `FusedGatedDualGEMM`      | Dual GEMM + sigmoid gate (TWO_INPUTS, mask, bias)         | `triton/fused_gated_dual_gemm.py` |
| Triton FusedTransitionBlock    | `FusedTransitionBlock`    | `TransitionBlock` (LN + Linear + SwiGLU + Linear)         | `triton/fused_transition.py`      |
| Triton fused MSA transition    | `fused_msa_transition`    | Evoformer `msa + s * msa_transition(msa)`, inference only | `triton/msa_transition.py`        |
| Triton FusedLayerNormLinear    | `FusedLayerNormLinear`    | Fused LayerNorm + Linear                                  | `triton/layernorm_linear.py`      |
| Triton FusedLayerNormTranspose | `FusedLayerNormTranspose` | LN + dim transpose (col attention)                        | `triton/layernorm_transpose.py`   |
| Triton FusedTriangleMulGate    | `FusedTriangleMulGate`    | LN + packed 3-projection GEMM + sigmoid                   | `triton/triangle_mul_gate.py`     |
| Persistent Autotune Cache      | `CacheManager`            | GPU-fingerprinted JSON cache for Triton configs           | `triton/cache.py`                 |
| TiledLinear                    | `TiledLinear`             | Memory-efficient tiled recomputation for pair transitions | `tiled_linear.py`                 |
| torch.compile                  | —                         | Wraps TriangleMul, MSAAttention, SingleAttention          | PyTorch built-in                  |

## Configuration

### OptimConfig

```python
@dataclass
class OptimConfig:
    use_fused_evo_attention: bool = False   # Triton flash-attn for GridSelfAttention
    use_fused_pair_bias: bool = False       # Triton fused LN+Linear+permute (tl.dot GEMM)
    use_fused_swiglu: bool = False          # Triton multi-row SwiGLU (1.5-2.1x at pair M=N²)
    use_fused_transition_pair: bool = False # Triton fused transition for pair (M=N², C=128)
    use_fused_triangle_mul: bool = False    # Triton dual GEMM + gate fusion for TriangleMul
    use_cueq_triangle_mul: bool = False     # cuEquivariance TriangleMul (optional cueq group)
    use_channel_major_triangle_mul: bool = False  # channel-first TriangleMul with a fused backward
    use_fused_ln_transpose: bool = False    # Triton LN+transpose for column attention
    use_fused_msa_transition_inference: bool = False  # in-place forward-only MSA transition (inference preset only)
    use_torch_compile: bool = False         # torch.compile for attention + TriangleMul
```

### Python API

```python
from deepfold3.model.config import ModelConfig, OptimConfig

config = ModelConfig()
config.optim = OptimConfig.inference()

# Check what's available on this system
from deepfold3.model.optim import print_availability
print_availability()
```

### CLI

| `--optim` | Kernels | Numerics |
| --- | --- | --- |
| `inference` (default) | Triton evo attention, pair bias, SwiGLU, channel-major TriangleMul, MSA transition; OPM and MSA chunking | Differs from training within each kernel's error |
| `train-matched` | What the checkpoint recorded; refuses a checkpoint with no `OptimConfig` | The kernels the weights were fitted through |
| `none` | Stock PyTorch | Reference; also makes `--matmul-precision auto` use true fp32 |
| `inference-cueq` | `inference` with cuEquivariance's TriangleMul; needs `uv sync --group cueq` | Slower than the channel-major TriangleMul, see below |

```bash
deepfold3 inference --json input.json --output out/ --params weights.safetensors --optim train-matched
```

### Channel-major TriangleMul

`use_channel_major_triangle_mul` (`triton/fused_trimul.py`) keeps the contraction operands as contiguous `[C, N, N]` planes. Its forward is three launches ported from Anthropic's `fpf_trimul_v4` (Apache-2.0, see `fused_trimul.NOTICE`); the backward is ours, on the cuEquivariance training design: the forward saves its planes, the backward recomputes nothing and uses two fused VJP kernels plus cuBLAS GEMMs.

Real activations from `checkpoint_120000` under bf16 autocast, 42 captures, RTX 6000 Ada, torch 2.13, cuEquivariance 0.13. Error is RMS against stock fp32 relative to stock bf16; gradient error is the medians for `z` against stock fp32. Single module, N=640:

| Kernel | fwd | fwd+bwd | Peak fwd+bwd | Output error | `z` gradient error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock | 24.8 ms | 79.2 ms | 6.61 GiB | 1.000 | 7.52e-3 |
| Level 3 dual GEMM | 18.6 ms | 63.4 ms | 5.23 GiB | 1.173 | 7.69e-3 |
| cuEquivariance | 13.8 ms | 45.6 ms | 5.23 GiB | 0.824 | 5.35e-3 |
| **Channel-major** | **4.0 ms** | **30.5 ms** | 6.60 GiB | 0.824 | **4.52e-3** |

GEMMs are not the cost: one takes 0.8 ms at this shape, while the output LayerNorm and its autograd backward take 36 ms. Calls the kernels cannot serve (CPU, C=64, fp32 GEMMs, N < 101) run the same layout in plain torch and are no faster than stock in fwd+bwd. Training presets leave the flag off. Tiles come from `persistent_autotune` like the other fused kernels, so run `deepfold3 warmup-triton` before enabling the flag in DDP training; `--det` serves the first config of each list.

A requested kernel that cannot run is never a silent stock run:

- **Backend not installed:** model construction raises `NotActive`.
- **Gated by a condition:** the kernel runs stock, logs `NOT ACTIVE` once, and appears under `kernels_inactive` in `run_manifest.json`.

`use_fused_transition_pair` and `use_tiled_linear` are gated to C in (64, 128). At `pair_channel=256` they reach only the C=128 MSA and template transitions, never the pair transitions.

`cuequivariance-ops-torch-cu13` is NVIDIA proprietary software, so the `cueq` group stays optional. Below its own size threshold the package runs its PyTorch path.

### Recommended Configuration

See the **B200 End-to-End Benchmark** at the top for the measured training/inference numbers these recommendations come from.

- **For training**: the `optim:` block in `configs/initial.yaml` — fused evo-attention + fused dropout-add + the safe pair-scale trio (`use_fused_swiglu`, `use_fused_transition_pair`, `use_fused_triangle_mul`) + gradient checkpointing. `use_fused_pair_bias` stays **off** (bf16 backward gradient blow-up). Do **not** use `--optim all` for training — it enables `pair_bias` (training hazard) and the single/ada transitions (slower at C=384/768).
- **For inference**: `--optim inference`, or `--optim train-matched` to run the kernels the weights were fitted through.
- **Debugging**: `--optim none` — pure PyTorch, no compiled graphs.

### Inference output writes

`deepfold3 inference` compresses logits on a background thread (`io/background_writer.py`), so the next
seed starts while they are written. The files are byte-identical to inline writes. Measured on real
logits, 5 samples, `checkpoint_120000.pt`, RTX 6000 Ada:

| Tokens | Forward per seed | Logit writes | mmCIF + confidences | Time between seeds, before → after |
|---|---|---|---|---|
| 566 | 90 s | 10.8 s | 2.0 s | 12.6 s → 2.9 s |
| 1016 | 298 s | 41.5 s | 3.2 s | 44.9 s → 6.8 s (replay) |

Deflate does not get cheaper on real logits: 26 MB/s against 37 MB/s on random ones. Scaled by N² that
is about 160 s per seed at N=2000. Confidences stay on the caller's thread because they run on the GPU
and CPU arithmetic may round differently. `--plot-confidences` drains the writer first, as the plot reads
those files, so it gets no gain. A failed write stops the run at the next submit, and the run does not
exit until every queued write has finished.

`inference-batch` still writes inline. It writes no per-sample logits, so mmCIF and confidences are its
whole tail (about 1%), and `--skip-existing` trusts `model.cif` written after the distogram.

## Installation

Triton kernels require the `optim` dependency group:

```bash
# Install with optimization dependencies
uv sync --group optim

# torch.compile requires no extra dependencies (PyTorch built-in)
```

All optimizations are optional. If a backend is not installed, the corresponding
flag is silently ignored and the PyTorch baseline is used.

## Architecture

Optimizations are applied at module construction time. `OptimConfig` is threaded through:

```
DeepFold3(cfg.optim)
  -> Evoformer(optim)
       -> EvoformerBlock(optim)
            -> GridSelfAttention(optim)     # Triton EvoAttention (4-9x)
            -> TriangleMulOutgoing(optim)   # torch.compile (1.3-1.6x)
            -> TriangleMulIncoming(optim)   # torch.compile (1.3-1.6x)
            -> MSAAttention(optim)          # torch.compile (1.4-2.7x)
            -> make_transition_block(optim) # FusedTransition (C>=256 only)
       -> PairFormerBlock(optim)
            -> GridSelfAttention(optim)     # Triton EvoAttention (4-9x)
            -> TriangleMulOutgoing(optim)   # torch.compile (1.3-1.6x)
            -> TriangleMulIncoming(optim)   # torch.compile (1.3-1.6x)
            -> SingleAttention(optim)       # torch.compile only (1.3-1.4x)
            -> make_transition_block(optim) # FusedTransition (C>=256 only)
  -> DiffusionHead(optim)
       -> DiffusionConditioningBlock(optim)
            -> make_transition_block(optim)
       -> DiffusionTransformer(optim)
            -> DiffusionTransformerBlock(optim) # torch.compile (×24, 4800 calls/sample)
```

### FusedTransition min-channels guard

`make_transition_block()` only uses `FusedTransitionBlock` when `channels >= 256`.
For smaller channels (C=64 MSA, C=128 pair), benchmarks show Triton overhead makes
it slower than baseline PyTorch.

### torch.compile strategy

`torch.compile` is applied to inner `_compute` methods (not the full `forward`) to
avoid recompilation from dynamic batch dimensions. The `has_batch` unsqueeze/squeeze
wrapper stays outside the compiled region.

### Triton Kernel Files

```
src/deepfold3/model/triton/
├── __init__.py               # Precision enum, cvt_tf32_rn, precision_dot, lazy imports
├── evo_attention.py          # Flash-attention with pair bias (5 Triton JIT kernels)
├── fused_pair_bias.py        # Fused LN + tl.dot projection + permute + mask (cuEQ pattern)
├── fused_gated_dual_gemm.py  # Dual GEMM + sigmoid gate (TWO_INPUTS, bias, mask, TF32)
├── fused_transition.py       # FusedTransitionBlock (compose LN+Linear+SwiGLU+Linear)
├── fused_ada_transition.py   # FusedAdaTransitionBlock (AdaLN + SwiGLU + Linear)
├── layernorm_linear.py       # FusedLayerNormLinear (4 JIT kernels, GEMMs autotuned)
├── ada_layernorm_linear.py   # FusedAdaLayerNormLinear (AdaLN + Linear, GEMMs autotuned)
├── layernorm_transpose.py    # FusedLayerNormTranspose (LN + dim transpose, elementwise)
├── triangle_mul_gate.py      # FusedTriangleMulGate (LN + packed GEMM + sigmoid, K-only)
├── swiglu.py                 # FusedSwiGLU (silu(a)*b convention, elementwise)
└── cache.py                  # Persistent autotuning cache (GPU-fingerprinted, on disk)
```

**Autotune coverage** — kernels marked "autotuned" use `@persistent_autotune` so block
sizes are picked by `triton.autotune` per (M, N, K) shape, with results persisted to disk
and reloaded on later runs (warm the cache once via `deepfold3 warmup-triton`; inference pads tokens to a bucket and reuses the nearest cached shape for MSA depths it has not seen):

| Kernel                                                                            | Autotune key              | Notes                                                                                                                                     |
| --------------------------------------------------------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `_attn_fwd`, `_attn_bwd_dq`, `_attn_bwd_dk_dv` (evo_attention)                    | `[SEQ_LEN, DIM]`          | First-class autotune since the original commit                                                                                            |
| `_fused_dual_gemm_fwd_kernel`, `_fused_dual_gemm_bwd_pregemm_kernel`              | `[M, N, K]`               | First-class autotune since the original commit                                                                                            |
| `_second_forward_kernel`, `_first_backward_kernel` (layernorm_linear)             | `[M, N, K]` / `[M, K, N]` | Converted to persistent_autotune; previously used a hand-tuned table, since removed, that returned `None` (PyTorch fallback) for buckets ≤224 |
| `_ada_second_forward_kernel`, `_ada_first_backward_kernel` (ada_layernorm_linear) | `[M, N, K]` / `[M, K, N]` | Same conversion as above                                                                                                                  |

The remaining elementwise kernels (`swiglu`, `triangle_mul_gate.{fwd,bwd}`,
`layernorm_transpose`, `fused_pair_bias`) use a single shape-agnostic block-size
heuristic — `BLOCK_SIZE = next_pow2(K)`, `num_warps` derived from `BLOCK_SIZE` — that's
already optimal across all bucket sizes. There's no autotune surface for them.

### TF32 Precision Control

All GEMM-based Triton kernels support configurable precision via the `Precision` enum
(adapted from NVIDIA cuEquivariance):

| Mode      | Value | Behavior                                                                 |
| --------- | ----- | ------------------------------------------------------------------------ |
| `DEFAULT` | 0     | Hardware default (fastest, least precise)                                |
| `TF32`    | 1     | Explicit TF32 rounding via `cvt_tf32_rn` (deterministic 19-bit mantissa) |
| `TF32x3`  | 2     | Triton's built-in tf32x3 accumulation (higher precision than TF32)       |
| `IEEE`    | 3     | Full IEEE754 float32 (slowest, most precise)                             |

The `precision_dot(a, b, acc, PRECISION)` JIT helper dispatches `tl.dot` with the
requested precision mode. The pair bias kernel defaults to `TF32x3` for best
accuracy/speed tradeoff.

### Attention Precision Contract

`evo_attention.py` keeps **operands in the input dtype and everything derived from
the scores in fp32**: the `tl.dot` accumulator, the online-softmax running max
`m_i` and sum `l_i`, the `exp2`, the output accumulator `O_block`, the stored LSE
`M`, and the `dQ`/`dK`/`dV`/`d_pair_bias` accumulators. Only the `tl.dot` operands
themselves (`Q`, `K`, `V`, and the `P` / `dS` tiles fed back into a dot) are bf16.

#### The backward never forms `dP - D`

All three dS-forming kernels (`dq`, `db`, `dk_dv`) need
`dS_j = P_j (dP_j - D)` with `D = Σ_i P_i dP_i`, and at the softmax argmax that
difference is pure cancellation: `dP_imax` and `D` agree to `O(e^-m)` in the logit
margin `m` (arXiv:2610.31642 §2, the `1 + e^-m` structure). In this kernel three
separate roundings each destroy it on their own — measured at B1 S2 N256 H8 D32,
margin 14.45, as the fraction of rows whose `dS_imax` is **exactly** zero:

| mechanism | effect |
| --- | --- |
| (a) `D = rowsum(dO * O)` from the old `_attn_bwd_preprocess`: bf16 `P` operand before the PV dot | 77% of rows |
| (a) same, via the bf16 `O` store | 87% of rows |
| (b) `P = exp2((QK - M)/ln2)` rebuilt from the fp32 LSE: `P` is the true softmax times `1 ± ulp(|QK|)`, so at `|QK| ≈ 20` (ulp 2.4e-6 vs `e^-14.45` = 5.3e-7) `P_imax` rounds to exactly 1.0 and `Σ_j P_j` misses 1 by ~1e-6 | 40% of rows, *after* (a) is fixed |
| (c) `D += tl.sum(P * dP, 1)` per KV tile: `O(e^-m)` tile partials land below the ulp of the winner's `O(1)` term | folded into (b)'s 40% |

So the cliff was at the **bf16** ulp, `m ≈ 5.5`, despite every op after the load
being fp32 — and recomputing `D` in fp32 does **not** lift it to the fp32
reference's 16.6, because (b) is a property of the LSE representation, not of
`D`'s arithmetic. An eager fp32 reference escapes (b) by normalising by a sum
(`1 + e^-m` is representable) instead of subtracting a log-sum-exp; an LSE-based
backward's own cliff is `log(1/ulp(|QK_max|))`, which depends on the logit
magnitude (≈13 at `|QK| ≈ 20`).

`_attn_bwd_d` therefore supplies the paper's eq. 5 form instead. It takes one
extra KV pass over the scores and emits three per-row vectors — `dP` at the row's
argmax, that column's index, and `C = Σ_i P_i (dP_i - dP_imax)` accumulated
online with a re-centring step per tile — and the consumers form

```
u_j  = 0 at j = imax (by index), else dP_j - dP_imax
dS_j = P_j (u_j - C)
```

which is algebraically identical (`Σ_i P_i = 1`) but never subtracts two `O(1)`
quantities, and couples (b)'s normalisation drift only to `(D - dP_imax) =
O(e^-m)`. `u` is zeroed by index rather than by arithmetic because the three
consumers form `dP` with three different dots, and half an ulp of disagreement on
`|dP| ≈ 0.5` is 3e-8 against a true `dS_imax` of ~1e-7. It is a separate kernel
because `db` and `dk_dv` cannot build the triple themselves — their programs own
an `(i, j)` tile and a KV block, so neither ever sees a whole row of `j`.

Against an fp64 oracle at B1 S2 N256 H8 D32, margin 14.45 (the shape and spike of
`test_extreme_attention_concentration`), before → after:

| | before | after |
| --- | --- | --- |
| rows with `dS_imax` exactly 0 | 61.5% | **0.0%** |
| median fraction of the true `dS_imax` kept | 0.000 | **1.0000** |
| `d_pair_bias` row sum / peak element | 7.1 | **2.19e-3** (= the bf16 floor of the returned grad) |
| `dQ` max error / true peak | 1.085 | **4.4e-3** |

A margin sweep puts the old onset at `m ≈ 2.5` (row sums 10x the floor), half the
signal gone by `m ≈ 6`, all of it by `m ≈ 8` — and past `m ≈ 16` the failure
inverts: the stale `dS` stays at ~1e-8 while the truth keeps shrinking like
`e^-m`, so `dQ` max error over true peak reaches 12 at `m = 18.5` and 622 at
`m = 22.5`. Absolute magnitudes stay ≲1e-8 either way, but the regime injects
noise rather than merely dropping signal. The fixed kernel holds every metric at
the bf16 floor across `m = -1.5 … 22.5` with no cliff in range.

**Cost.** One extra KV pass in the backward: fwd+bwd +7–8%, backward alone
+10–12% (medians of 60 iterations on an RTX 6000 Ada, at B1 S2 N384 H8 D32 and
B1 S1 N768 H4 D32, interleaved against the base on one GPU). The forward is
byte-for-byte unchanged, so inference pays nothing. Backward memory *falls*: `O`
is no longer saved (it was the preprocess pass's only consumer — 64 B per
`(batch, msa, head, res)` row at D=32) in exchange for three per-row vectors
(12 B). See `tests/triton/test_evo_attn_softmax_collapse.py`.

`softmax_scale` is applied **after** the QK dot, on the fp32 accumulator, in every
kernel that recomputes scores. Folding it into Q (or K) before the dot re-rounds the operand to
bf16, and that ~2^-9 relative perturbation becomes an absolute logit error
proportional to the logit's magnitude — which the softmax exponentiates. At
`|QK^T/sqrt(D)| ~ 1e2` (activations with stddev ≳ 3) it cost ~5% of forward
elements their bf16 tolerance, and because the backward `dk_dv` kernel pre-scaled
`K` while the forward pre-scaled `Q`, its recomputed `P` also stopped matching the
saved LSE — the dominant `dK` / `dV` error. See
`tests/triton/test_evo_attn_robustness.py::test_row_norm_heterogeneity`.

### Persistent Autotuning Cache

Triton kernel autotuning results are persisted to `~/.cache/deepfold3-triton/` (or
`DEEPFOLD3_TRITON_CACHE_DIR`). The cache:

- **GPU fingerprinting**: Uses `torch.cuda.get_device_properties` (compute capability, memory, SM count)
- **Fuzzy matching**: When switching GPUs within the same architecture family, cached configs
  transfer if memory/SM count are within 20% tolerance
- **Atomic writes**: Thread-safe JSON persistence via temp files + `os.replace`
- **Pre-hook injection**: On first call for a (kernel, shape) pair, the wrapper checks the
  persistent cache and, if a config is present, injects it directly into Triton's internal
  `Autotuner.cache` so the benchmark phase is skipped entirely. This is what makes warmup
  worth running before training.
  > **Note.** A hardcoded `B200_PRETUNED` table used to seed the cache at import so a fresh
  > B200 paid no autotune cost. It was removed (~2,200 lines, incomplete — `_attn_bwd_db`
  > never had configs, crops > 384 weren't covered) in favor of the shared-cache + warmup
  > workflow below. On an empty cache, kernels now autotune on first call and persist.

#### `--det`: fixed configs

Tile sizes change the order of floating-point sums, so a config picked by timing, or read from a host's
cache, can move the low bits of a result. `deepfold3 inference --det` serves each kernel its first config
(`set_fixed_config`), and neither reads nor writes the cache. It also sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`
(unless already set), `torch.use_deterministic_algorithms(True)` (which raises on an op with no deterministic
path) and cuDNN deterministic mode. It refuses `torch.compile` presets and `DEEPFOLD3_TRITON_NO_CACHE=1`.

**Guarantee.** The same input, weights, seeds and flags give the same output bytes on the same GPU model,
driver, software stack and CPU thread count. Measured on 7st5 (N=566, two seeds, five samples): two `--det`
runs were byte-identical on all 34 files, one with an empty cache directory that stayed empty.
`PYTHONHASHSEED` cannot be set from inside the process, but feature tensors were identical across three
hash seeds on three inputs, including a ligand job.

**Not guaranteed.** Reproduction across GPU models, drivers, or torch/Triton versions (compare
`run_manifest.json`); agreement with a run without `--det`; agreement between `--bucket` and `--no-bucket`,
whose sampler noise differs. Only `--optim inference`-style presets were tested. `inference-batch` has no `--det`.

**Cost.** Forward 99.6 s against 90.0 s per seed (+10.7%) with the default warmed cache. Against that
default, coordinates and distogram logits were bit-identical; confidence outputs differed in the low bits
(per-atom pLDDT by up to 0.012, ranking scores unchanged at four decimals).

#### Recommended workflow: warmup once, read from a shared cache

This is the canonical path (it replaced the hardcoded `B200_PRETUNED` table).

1. Point `~/.cache/deepfold3-triton/` (or `DEEPFOLD3_TRITON_CACHE_DIR`) at **shared storage**
   mounted by every node — or bake a warmed cache into the training container image.
2. Run warmup **once, on a single rank**, with `--real-model` (drives the actual model at
   the production shapes, so every cached `(M, N, K)` and kernel specialization matches
   training exactly — no synthetic-shape-list drift):

   ```bash
   deepfold3 warmup-triton --config configs/initial.yaml --real-model
   ```

3. Every node and every later job then **reads** the persisted configs at startup (lazy
   load + pre-hook injection, `cache.py:204`) and skips autotuning entirely. The read is a
   plain JSON load, so it works over NFS/shared FS.

**Caveat — no cache misses during DDP.** Writes happen only on a miss, and many ranks
missing the *same* shape will write the *same* shared JSON concurrently (the file lock is
in-process, not cross-node) — a corruption/race risk. So warmup must cover **all** buckets
you will hit (384 today; also 640/768 if you move to larger crops). After warmup, the DDP
run should only ever read. `DEEPFOLD3_TRITON_NO_CACHE=1` (below) is a blunt escape hatch.

Environment variables:

- `DEEPFOLD3_TRITON_CACHE_DIR`: Override the cache directory (highest precedence).
- `XDG_CACHE_HOME`: If `DEEPFOLD3_TRITON_CACHE_DIR` is unset, the cache lives under
  `$XDG_CACHE_HOME/deepfold3-triton` (then falls back to `~/.cache/deepfold3-triton`).
- `DEEPFOLD3_TRITON_NO_CACHE=1`: Disable caching (re-tune every run).

> **Containers where `$HOME` is ephemeral (e.g. kakao).** The training image runs as a
> user whose real `$HOME` (`/home/mpiuser`) is *not* the persistent volume — the PVC is
> mounted at `/home/jovyan`. `Path.home()/.cache` would therefore land the cache on
> ephemeral per-pod storage, so all 64 ranks would re-autotune and race-write. The cache
> now honors `XDG_CACHE_HOME` (which the trainjob sets to `/home/jovyan/.cache`), and the
> trainjob also sets `DEEPFOLD3_TRITON_CACHE_DIR=/home/jovyan/.cache/deepfold3-triton`
> explicitly. Both point the cache at the shared PVC. Note this is a *different* cache from
> Triton's own `TRITON_CACHE_DIR` (compiled binaries) — they must each be redirected.

#### Warming the cache

With no hardcoded table, **every** GPU (B200 included) starts from an empty cache and
autotunes each shape on first encounter — wasted compute under DDP, where every rank
autotunes the same shapes redundantly and can desync. Warm the cache **on a single rank**
before launching DDP training (and point it at the shared/persistent path so all ranks
reuse it):

There are two modes:

**`--real-model` (recommended).** Builds the actual `DeepFold3` (same `ModelConfig` path as
`deepfold3 train`) and runs `forward_training` + backward at `fixed_bucket_n` (= `max_tokens`,
default 384) with `batch_size_per_rank` and the diffusion params read from `--config`. Every
cached shape and kernel specialization therefore matches training exactly — there is no
hand-maintained shape list to drift out of sync (the gap that previously left `dual_gemm`
uncached). It warms the per-chunk diffusion size (`min(diffusion_chunk_size, diffusion_batch_size)`).

By default every repeated block stack is **collapsed to depth 1** (`collapse_to_depth_1`):
PairFormer 48→1, diffusion transformer 24→1, Evoformer / confidence / template / atom
transformer →1. The Triton autotune cache is keyed by `(kernel, shape)` — the layer index is
not part of the key and every layer in a stack shares identical channel dims — so one layer
warms the *identical* set of cache entries as the full stack. Coverage is unchanged; the other
47 PairFormer layers were pure redundant fwd/bwd. This keeps activation memory at ~a single
layer (so warming at the production `--msa-depth` no longer OOMs at N=384) and runs in seconds.
Pass `--full-depth` to build the whole model instead — only an escape hatch; it costs ~N× more
memory/time for no extra coverage. `--msa-depth` defaults to `model.msa_crop_size`; lower it
(or `--buckets`) only if the forward still OOMs.

`--override key=value` (same parser as `deepfold3 train` — bare keys → training section, dotted
for nested) tweaks the shapes/params on top of `--config`, e.g. `--override batch_size_per_rank=1
--override diffusion_chunk_size=16 --override max_recycles=3 --override model.msa_crop_size=2048`.
Handy for cache-coverage or timing sweeps; an explicit `max_recycles` is honored uncapped (the
warming default caps recycles at 2 since they add no new shapes). The warmup pins the current CUDA
device to `--device` so non-default devices (e.g. `cuda:1`) launch Triton kernels on the right
context.

```bash
deepfold3 warmup-triton --config configs/initial.yaml --real-model
deepfold3 warmup-triton --config configs/stage2.yaml --inference-only   # inference forward only, no backward
```

**Synthetic (default, lighter).** Drives each fused kernel at hand-enumerated `(M, N, K)`
shapes without building the model — faster and lower-memory, but the shape list must be kept
in sync with the model (it can miss shapes/specializations the real run hits):

```bash
deepfold3 warmup-triton --config configs/initial.yaml
```

The synthetic script:

1. Loads channel dimensions from the model config (falls back to defaults).
2. Iterates over every bucket size in `bucket.BUCKET_SIZES ≤ 384` (pass `--buckets` to add
   larger crops, e.g. 640/768, if you train at those sizes).
3. For each bucket, drives `FusedLayerNormLinear`, `FusedAdaLayerNormLinear`,
   `fused_evo_attention`, and `FusedGatedDualGEMM` at the (M, N, K) shapes the model uses.
   (These kernels are stateless — the script passes LN/linear/gate weights at call time.)
4. The first call for each shape triggers `triton.autotune`; the result is persisted to
   `$CACHE/{kernel}.{major}.{minor}.json` keyed by GPU fingerprint.
5. Subsequent processes (training ranks) load the persisted cache via the pre-hook
   injection and skip the benchmark phase.

Total warmup cost on B200: order of minutes (one-time per GPU model, per shape set). The
candidate config space is the `_GEMM_CONFIGS` list in `triton/layernorm_linear.py`.

### Fused Pair Bias Design (cuEQ pattern)

The pair bias kernel fuses LayerNorm + Linear projection + layout permute into a single
kernel using NVIDIA cuEquivariance's 3D grid pattern:

```
Grid: (cdiv(V, TILE_V), U, cdiv(H, HEADS_PER_BLK) × B)

Phase 1: LN statistics (mean, rstd) over C in TILE_K tiles
Phase 2: Normalize + project via tl.dot([TILE_V, TILE_K] × [TILE_K, H_BLK])
Phase 3: Optional mask fusion + write to [B, H, U, V] layout
```

Key improvement: The old kernel used a scalar dot-product loop per head
(`tl.sum(xn * w_h)` × H). The new kernel uses `tl.dot` tensor core GEMM,
which is 5-10x faster for C=128, H=4. Supports both training (autograd with
PyTorch backward) and inference (fully fused, no intermediates).

### Fused Gated Dual GEMM Design (cuEQ pattern)

The dual GEMM kernel computes `sigmoid(x₁ @ W₁ᵀ + b₁) × (x₂ @ W₂ᵀ + b₂) × mask`
in a single kernel pass:

- **TWO_INPUTS**: Separate x₁, x₂ for gate and projection paths
- **Bias**: Optional b₁, b₂ added after GEMM accumulation
- **Mask fusion**: Per-row mask (AdaLN-zero gate) applied in epilogue
- **L2 grouping**: GROUP_SIZE_M linearized grid for cache locality (our advantage over cuEQ)
- **Backward**: Pre-GEMM kernel recomputes sigmoid + derives grad factors, then cuBLAS for weight/input gradients

## TiledLinear (Memory-Efficient Pair Transitions)

Pair transition blocks (`TransitionBlock` with C=64/128) operate on `[N, N, C]` tensors.
At N=384 with 4x expansion, each stores `[384, 384, 512]` intermediates for backward —
~150 MB per transition in bf16. With ~52 pair transitions in the Evoformer/PairFormer
trunk, this totals ~8 GB in activation memory.

`TiledLinear` wraps any transition block and chunks the input along `dim=-2` (pair rows).
Each chunk is processed independently, and intermediates are recomputed during backward
instead of being stored.

**How it works:**

- **Training**: Custom `torch.autograd.Function` runs forward under `torch.no_grad()`,
  saves only the input. During backward, recomputes each chunk's forward with
  `torch.enable_grad()` and accumulates parameter gradients across chunks.
- **Inference**: Optionally chunks the forward pass for peak memory reduction on very
  long sequences (enabled via `tiled_linear_inference=True`).

**Why it's exact:** All operations in `TransitionBlock` (LayerNorm, Linear, SwiGLU) act
on `dim=-1` (channels) independently per spatial position. Chunking along `dim=-2` produces
bitwise identical forward results. Backward input gradients are also bitwise identical.
Parameter gradients may differ at ULP level due to fp32 accumulation order (~1e-6 max diff).

**Memory savings:** At N=384 with 3 shards (auto heuristic): ~67% reduction in pair
transition activation memory (~5 GB saved). Cost: ~2x FLOPs for pair transitions only.

**Config:**

```python
OptimConfig(
    use_tiled_linear=True,        # Enable for pair-scale transitions (C=64/128)
    tiled_linear_shards=0,        # 0 = auto: ceil(spatial_dim / channel_dim)
    tiled_linear_inference=False,  # Also tile during inference (long sequences)
)
```

No preset enables it. `make_transition_block` only wraps blocks it did not fuse, at C in (64, 128).

## Attention Tuning Notes

### GridSelfAttention (EvoAttention kernel)

The Triton EvoAttention kernel operates on 5D tensors: `[B, N_seq, N_res, H, D]`.
For GridSelfAttention, `N_seq = N_rows` (row or column dimension) and `N_res = N_cols`
(the attention dimension). The kernel fuses the pair bias addition and avoids
materializing the N^2 attention matrix.

- **Row attention**: `N_seq = N` rows, each attending over `N_res = N` columns
- **Column attention**: transposed — `N_seq = N` columns, each attending over `N_res = N` rows
- **Pair bias**: `[B, 1, H, N, N]` — shared across all rows/columns within a batch

### SingleAttention (torch.compile only)

SingleAttention previously used EvoAttention with `N_seq = 1`, but benchmarks show
this is 0.87-1.03x (slower or neutral) due to kernel launch overhead dominating the
single-sequence case. `torch.compile` on the baseline path gives 1.3-1.4x speedup and
is now the only optimization applied.

---

## Benchmark Manual

### Prerequisites

- CUDA GPU required (all benchmarks check `torch.cuda.is_available()`)
- Install optimization backends you want to benchmark:

```bash
uv sync --group optim   # triton + matplotlib
```

- Check which backends are detected:

```bash
uv run python -c "from deepfold3.model.optim import print_availability; print_availability()"
```

### Quick Start

```bash
# 1. Verify numerical correctness of all kernels (always run first)
uv run python benchmarks/verify.py --verbose

# 2. Run all benchmarks + generate plots
uv run python benchmarks/run_all.py

# 3. Smoke test (tiny shapes, fast — good for CI or checking setup)
uv run python benchmarks/run_all.py --dry-run
```

### Running Individual Benchmarks

Each benchmark script can be run standalone. All accept `--dry-run`, `--warmup`, and `--rep` flags.

```bash
# Primitives: LayerNorm, SwiGLU, LinearNoBias
uv run python benchmarks/bench_primitives.py

# Transition blocks: TransitionBlock, AdaTransitionBlock
uv run python benchmarks/bench_transition.py

# Attention modules: GridSelfAttention, MSAAttention, SingleAttention
uv run python benchmarks/bench_attention.py

# Pair operations: TriangleMul (outgoing/incoming), OuterProductMean
uv run python benchmarks/bench_pair_ops.py
```

### Numerical Verification

```bash
uv run python benchmarks/verify.py --verbose
```

### Auto-Tuner

```bash
# Tune a single shape
uv run python benchmarks/tuner.py --op transition_block --N 512 --C 384

# Sweep multiple common shapes
uv run python benchmarks/tuner.py --op transition_block --sweep

# Tune Triton block sizes for all DeepFold3 common shapes
uv run python benchmarks/tuner.py --triton-tune

# Smoke test
uv run python benchmarks/tuner.py --dry-run
```

### Result Format

Benchmark results are saved as JSON arrays in `benchmarks/results/`. Each entry:

```json
{
  "op_name": "GridSelfAttention",
  "provider": "triton",
  "shape": { "N": 256, "C": 128, "H": 4 },
  "mode": "fwd",
  "time_ms": 0.123,
  "peak_memory_mb": 45.6,
  "dtype": "bfloat16",
  "extra": {}
}
```

### Benchmark File Reference

```
benchmarks/
├── __init__.py            # Package marker
├── utils.py               # gpu_timer(), peak_memory_mb(), BenchResult, save/load JSON
├── optim_config.py        # OptimConfig, is_available(), get_optimized_module()
├── verify.py              # Numerical verification (fwd+bwd allclose)
├── tuner.py               # Auto-tuner with --sweep and --triton-tune modes
├── bench_primitives.py    # LayerNorm, SwiGLU, LinearNoBias
├── bench_transition.py    # TransitionBlock, AdaTransitionBlock
├── bench_attention.py     # GridSelfAttention, MSAAttention, SingleAttention
├── bench_pair_ops.py      # TriangleMul, OuterProductMean
├── plot_results.py        # Matplotlib grouped bar charts (per-suite PNGs)
├── optimization_report.py # Slide-deck PDF: summary, verification, per-op charts, table
├── run_all.py             # Orchestrator: verify -> bench all -> plot + report
└── results/               # Output directory (JSON + PNG + PDF)
```

## Kernel census

`OptimConfig` records what was requested, not what ran. `model/kernel_census.py` tallies the path each dispatch site took, and `inference` / `inference-batch` write it to `run_manifest.json` as `kernels` (`{site: {path: count}}`) with `kernels_shadowed`, the levers that were enabled but never reached. Read those fields, not the preset name, to establish which kernels produced a result. `use_fused_ln_transpose` is the standing example: `use_fused_pair_bias` is tested first and every preset that enables the former enables the latter, so `inference()` builds that kernel and never runs it. Adding a dispatch branch means adding a `kernel_census.record` call to it.

## Inference-only kernels refuse

An inference-only kernel carries `@inference_only(...)` from `model/inference_guard.py`: with grad enabled it raises `InferenceOnlyKernelError` instead of falling back, so training cannot reach it through a flag. A kernel that cannot serve a call for another reason says so through the census (`refused:<reason>`), so the reason reaches `run_manifest.json`. `use_fused_msa_transition_inference` follows this; `OuterProductMean._project_rows` and `map_row_chunks` still take their plain path under grad.
