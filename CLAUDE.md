# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepFold3 is a PyTorch biomolecular complex structure prediction system, architecturally equivalent to AlphaFold 3.

The project has a working inference pipeline and a core training pipeline — model, data, I/O, training, and scripts are implemented in `src/`.

**Status:** Training-time template date-cutoff filter is implemented (`dataset.py:_load_monomer_templates` applies `min(max_template_date, record.released - cutoff_days)` per record). Symmetry alignment for AA / chain / ligand permutation is implemented (`data/symmetry/`). Cluster-balanced sampling is per-(chain|interface) via a two-stage decomposition — the sampler draws records by the cluster marginal and each weighted dataset draws the crop anchor at load time (`dataset.py:_draw_crop_anchor`), reproducing the AF3 §2.5.1 flat per-anchor joint exactly (chain rule). See `docs/training.md`.

- **Package manager**: `uv` (lockfile present as `uv.lock`)
- **Python version**: 3.11+ (pinned in `.python-version`)
- **Source layout**: `src/deepfold3/` (packages found under `src/` via setuptools)

## Commands

```bash
# Install dependencies (including dev)
uv sync --dev

# Run tests
uv run pytest

# Run a single test file or test
uv run pytest tests/path/to/test_file.py
uv run pytest tests/path/to/test_file.py::test_name

# Lint / format
uv run ruff check src/
uv run ruff format src/

# CLI (unified entry point via click — registered in pyproject.toml as deepfold3)
deepfold3 msa       --sequence MKTII --output msa_out/ --json
deepfold3 msa       --sequence SEQ1:SEQ2:SEQ1 --output msa_out/ --use-pairing --json
deepfold3 msa       --fasta query.fasta --output msa_out/ --mode local --db-dir /data/mmseqs_dbs/ --json
deepfold3 msa       --fasta query.fasta --output msa_out/ --mode local --db-dir /data/mmseqs_dbs/ --gpu --json
deepfold3 msa       --fasta query.fasta --output msa_out/ --mode local --db-dir /data/mmseqs_dbs/ --gpu --gpu-server
deepfold3 msa       --sequence MKTII --output msa_out/ --mode local --db-dir /data/mmseqs_dbs/ --use-templates --json
deepfold3 msa       --sequence MKTII --output msa_out/ --link-templates --pdb-mmcif-dir /data/pdb/mmcif/
deepfold3 inference --msa query.a3m --output out.cif --params weights.safetensors
deepfold3 inference --json input.json --output output_dir/ --params weights.safetensors --template-cache /tmp/cache
deepfold3 inference --json input.json --output output_dir/ --params weights.safetensors --pdb-mmcif-dir /data/pdb/mmcif/
deepfold3 inference --json input.json --output output_dir/ --params weights.safetensors --msa-crop-size 4096
deepfold3 inference --json input.json --output output_dir/ --params stage2.safetensors --guidance 2.0@H:A,L:A
deepfold3 inference --json input.json --output output_dir/ --params stage2.safetensors --guidance 1.5@all@1..160 --guidance 2.0@H:A
deepfold3 inference --json input.json --output output_dir/ --params stage2.safetensors --restraint A/25-40+88:B@6
deepfold3 inference --json input.json --output output_dir/ --params stage2.safetensors --restraint-pair A/59+33+55:C/97+94+94@5 --restraint-weight 0.3
deepfold3 inference --json input.json --output output_dir/ --params stage2.safetensors --repel A/12+15:B@8 --restraint-after-step --restraint-after-sigma 0.001
deepfold3 inference --json input.json --output output_dir/ --params weights.safetensors --no-hoist-sampler
deepfold3 inference --json input.json --output output_dir/ --params weights.safetensors --det
deepfold3 convert   --params af3.bin.zst --af3-ref references/alphafold3 --output weights.safetensors
deepfold3 export    --checkpoint runs/exp1/checkpoint_final.pt --output weights.safetensors --snapshot average
deepfold3 plot      --input out_confidences.json --output confidences.png
deepfold3 plot      --input out/seed-42_sample-0/            # dir, logits.npz, or *_distogram.npz
deepfold3 plot      --input out/seed-42_sample-0/logits.npz --panels pae,contact --contact-threshold 12
deepfold3 prepare   --mmcif-dir /scratch/pdb/mmCIF --output /data/training --pdb-ids 1pfe,8qy0
deepfold3 prepare   --mmcif-dir /scratch/pdb/mmCIF --output /data/training --pdb-id-file pdb_ids.txt --workers 16
deepfold3 prepare   --mmcif-dir /scratch/pdb/mmCIF --output /data/training --pdb-id-file pdb_ids.txt --no-assembly --workers 16
deepfold3 prepare   --mmcif-dir /scratch/pdb/mmCIF --output /data/training --max-resolution 4.0 --workers 8
deepfold3 prepare   --mmcif-dir /scratch/pdb/mmCIF --output /data/training --cluster-file clusters-by-entity-40.txt
deepfold3 fetch-ccd        --output /data/ccd/
deepfold3 build-ccd-cache  --components-cif /data/ccd/components.cif.gz --output /data/ccd/ccd_cache.json --ref-pos --automorphisms --workers 16
deepfold3 cluster   --data-dir /data/training --output clusters.txt
deepfold3 cluster   --data-dir /data/training --output clusters.txt --min-seq-id 0.3
deepfold3 apply-clusters --manifest /data/training/manifest.json --cluster-file clusters-by-entity-40.txt --output /data/training/manifest.json
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ --gpu
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ --gpu --use-pairing
deepfold3 msa-batch --data-dir /data/training --convert-only
deepfold3 import-msa --data-dir /data/training --msa-dir /data/afdb/a3ms --workers 32
deepfold3 import-paired-msa --data-dir /data/training --paired-dir /data/msa/paired --workers 32
deepfold3 import-of3 --of3-dir /data/of3/structures --output /data/training
deepfold3 import-of3 --of3-dir /data/of3/structures --output /data/training --cache /data/of3/dataset_cache.json
deepfold3 build-of3-cache --cache-file /data/pdb_training_set/dataset_caches/training_cache_with_templates.json --output /data/pdb_training_set/df3_cache
deepfold3 prepare-templates --data-dir /data/training --m8-dir /data/msa/pdb/merged/m8
deepfold3 prepare-templates --data-dir /data/training --m8-dir /data/msa/pdb/merged/m8 --max-templates 4 --workers 16
deepfold3 stamp-template-index-space --data-dir /data/training --space struct --workers 32
deepfold3 build-pdb-cache --data-dir /data/training --mmcif-dir /data/pdb/divided --output /data/pdb/cache --workers 64
deepfold3 scan-mmcif --mmcif-dir ~/deepfold3/pdb/divided --data-dir /data/training --workers 32
deepfold3 enrich-manifest --data-dir /data/training --idx entries.idx
deepfold3 enrich-manifest --data-dir /data/training --idx entries.idx --mmcif-cache data/mmcif_meta.json
deepfold3 drop-obsolete --data-dir /data/training
deepfold3 drop-obsolete --data-dir /data/training --delete-files
deepfold3 refilter      --data-dir /data/training --max-unk-fraction 0.10 --max-tight-clashes 5
deepfold3 export-complexes --data-dir /data/training --output complexes.json
deepfold3 build-masks   --data-dir /data/training --json-dir /data/patchr/inpainting --workers 32
deepfold3 pack          --src /data/distillation/afdb_nvda --include structures --include msa --include monomer_templates --exclude '*.tar.gz' --out /packed/distillation/afdb_nvda/_pack --workers 32
deepfold3 pack          --src /data/distillation/long_monomers/preprocessed --ids-file ids.txt --member structure.npz --member alignment.npz --member template.npz --out /packed/distillation/long_monomers/preprocessed/_pack
deepfold3 pack-verify   --pack /packed/distillation/afdb_nvda/_pack --src /data/distillation/afdb_nvda --sample 2000 --checksums
deepfold3 inspect-example --data-dir /data/training --id 7jyb
deepfold3 inspect-example --data-dir /data/training --id 7jyb,8qy0,1pfe --json
deepfold3 train     --data-dir /data/training --output runs/exp1 --params weights.safetensors
deepfold3 train     --data-dir /data/training --output runs/exp1 --params weights.safetensors --mmcif-dir /data/pdb/mmcif
deepfold3 train     --data-dir /data/training --output runs/exp1 --resume runs/exp1/checkpoint_1000.pt
deepfold3 train     --data-dir /data/training --output runs/exp1 --config configs/finetune1_structure_640.json --params runs/exp1/checkpoint_final.pt
deepfold3 train     --data-dir /data/training --output runs/exp1 --override alpha_pae=1.0 --override train_confidence_only=true
deepfold3 train     --data-dir /data/training --output runs/exp1 --wandb deepfold3
deepfold3 train     --data-dir /data/pdb --output runs/exp1 --dataset afdb:/data/afdb:0.25:uniform:true
deepfold3 train     --data-dir /data/pdb --output runs/exp1 --dataset afdb:/data/afdb:0.5:uniform:true --dataset pdb2:/data/pdb2:1.0:weighted
# OpenFold 3 `pdb_training_set` as the weighted PDB set (`format: of3pdb`) — configured
# via a `datasets:` entry, not a flag; see configs/initial.yaml and docs/training.md.
# Complex language model (separate model — see docs/complex-lm.md)
deepfold3 plm-prepare structure --data-dir /data/distillation/afdb_nvda --output /data/plm/store/afdb --source afdb --workers 64
deepfold3 plm-prepare pairs     --fasta /data/plm/raw/STRING_seqs.fasta --pairs /data/plm/raw/trainset.STRING.txt --output /data/plm/store/string --source string
deepfold3 plm-prepare cluster   --store /data/plm/store/pdb --store /data/plm/store/afdb --work-dir /data/plm/clust
deepfold3 plm-prepare stats     --store /data/plm/store/pdb
deepfold3 plm-train --config configs/plm-base.yaml --output runs/plm-r1
torchrun --nproc_per_node=8 -m deepfold3.scripts.plm_train --config configs/plm-base.yaml --output runs/plm-r1
# Multi-GPU training via torchrun
torchrun --nproc_per_node=4 -m deepfold3.scripts.train --data-dir /data/training --output runs/exp1
# Pre-warm the persistent Triton autotune cache (run once per GPU model on a single rank
# before launching DDP training; B200 ships with hardcoded configs and skips this step).
deepfold3 warmup-triton --config configs/initial.json
deepfold3 warmup-triton --config configs/stage2.yaml --inference-only
```

## Ruff Configuration

Line length 120; rules: E, F, I (isort), B (bugbear), UP (pyupgrade). `E501` is ignored (handled by `ruff format`). Double quotes, space indent. Google-style docstrings (`Args:`, `Returns:`, `Raises:`, `Attributes:`).

## Documentation Maintenance

When modifying source code, update the corresponding developer documentation in `docs/`:

- **Architecture changes** (new modules, changed forward pass, new heads) → update `docs/architecture.md`
- **Data pipeline changes** (new features, changed shapes, new parsing formats) → update `docs/data-pipeline.md`
- **New patterns or conventions** (testing, code style, extension points) → update `docs/contributing.md`
- **Complex-LM changes** (architecture, store format, masking, sampling) → update `docs/complex-lm.md`, and record measurements in `export/complex-lm/README.md`
- **Feature table** in `docs/data-pipeline.md` must stay in sync with `data/feature/featurizer.py` output

Keep the docs concise — update existing sections rather than appending.

## Architecture

The model is a **diffusion-based generative model** with three major heads and a recycled trunk:

```
Input features (tokens + atoms)
    → Per-atom conditioning (AtomCrossAttEncoder)
    → Evoformer trunk  [10 recycles]
        ├─ Template embedding (optional, T templates → pair)
        ├─ MSA processing (EvoformerBlock × 4)
        └─ PairFormer stack (48 layers)
               → single [N, 384]  +  pair [N, N, 128]
    → Diffusion Head  (200-step denoising)
        └─ atom_positions [S, N, M, 3]
    → Confidence Head (pLDDT, PAE, PDE, exp_resolved, pTM, ipTM)
    → Distogram Head
```

**Key dimensions**: `seq_channel=384`, `pair_channel=128`, `msa_channel=64`, `num_recycles=10`, `diffusion_steps=200`, `num_samples=5`.

**Atom representation**: Two layouts — _all-tokens_ (one representative atom per token) and _all-token-atoms_ `[N_tokens, max_atoms_per_token]` (all heavy atoms). Proteins/RNA/DNA use one token per residue; ligands use one token per atom.

## Source Layout

```
src/deepfold3/
├── model/           # Neural network modules
│   ├── primitives.py     # LinearNoBias, LayerNorm, SwiGLU, TransitionBlock, AdaLN
│   ├── config.py         # EvoformerConfig, DiffusionConfig, ConfidenceConfig, ModelConfig, TrainingConfig
│   ├── embedders.py      # create_relative_encoding
│   ├── attention.py      # GridSelfAttention, MSAAttention, SingleAttention
│   ├── ops.py            # TriangleMul, OuterProductMean
│   ├── evoformer.py      # EvoformerBlock, PairFormerBlock, Evoformer
│   ├── template_embedding.py  # TemplateEmbedding, SingleTemplateEmbedding, TemplatePairFormerBlock
│   ├── atom_cross_attn.py # AtomCrossAttEncoder/Decoder, EvoformerConditioningEncoder
│   ├── diffusion.py      # DiffusionTransformer, DiffusionHead (sampling + training noise/denoise)
│   ├── guidance.py       # `--guidance` terms: syntax, checkpoint check, chain names → GuidanceTerm
│   ├── confidence.py     # ConfidenceHead (pLDDT, PAE, PDE, exp_resolved)
│   ├── bucket.py         # Bucket-based sequence length padding for training
│   ├── model.py          # DeepFold3 (top-level, forward + forward_training), DeepFold3Output, DistogramHead
│   └── triton/           # Triton fused kernels (optional GPU acceleration)
│       ├── __init__.py            # Precision enum, cvt_tf32_rn, precision_dot
│       ├── evo_attention.py       # Fused evoformer attention with pair bias
│       ├── fused_pair_bias.py     # Fused LN + tl.dot projection + permute + mask
│       ├── fused_gated_dual_gemm.py # Dual GEMM + sigmoid (TWO_INPUTS, bias, mask)
│       ├── layernorm_linear.py    # Fused LayerNorm + Linear
│       ├── ada_layernorm_linear.py # Fused AdaLN + Linear
│       ├── layernorm_transpose.py # Fused LayerNorm + dim transpose
│       ├── triangle_mul_gate.py   # Fused TriangleMul gate (LN + packed GEMM)
│       ├── fused_ada_transition.py # Fused AdaTransition (diffusion FFN)
│       ├── fused_transition.py    # Fused TransitionBlock
│       ├── swiglu.py              # Fused SwiGLU
│       ├── tuning.py              # Pre-tuned block size configs (CUDA/HIP)
│       └── cache.py               # Persistent autotuning cache (GPU fingerprint)
├── data/            # Data pipeline
│   ├── const.py          # RES_TYPE_ORDER, ELEMENT_TO_INT, REF_ATOMS, element_name_to_int(), get_res_type()
│   ├── types.py          # Structure, MSA, Input, Tokenized, Record dataclasses
│   ├── masking.py        # TokenMask (per-token mask channels for loss control: fixed, region)
│   ├── ccd.py            # CcdMolecule, load_ccd_molecule(), smiles_to_molecule()
│   ├── paths.py          # shard(), structure_path, mask_path, template_path — sharded layout helpers
│   ├── pack.py           # packed data roots: tar shards + sorted index, read transparently (register/read_bytes/exists/is_dir)
│   ├── msa/             # mmseqs2.py (search) + paths.py (unpaired_msa_path, paired_msa_path)
│   ├── parse/            # mmcif.py (mol_type_from_gemmi), a3m.py (parse_a3m + convert_a3m_file_to_npz), templates.py (find_pdb_mmcif_path), template_realign.py, hhr.py, json_input.py (4 dialects), pdb_chain_cache.py (PdbChainCache: per-chain template cache, mem LRU + sharded disk)
│   ├── tokenize/         # tokenizer.py (Input → Tokenized)
│   ├── crop/             # cropper.py (contiguous + spatial + interface-seeded)
│   ├── sample/           # weighted.py (WeightedClusterSampler — AF3-style cluster-balanced)
│   ├── feature/          # ref_atoms.py (CCD), featurizer.py (Tokenized → Tensor dict)
│   ├── prepare/          # structure.py (mmCIF→NPZ + AF3-aligned metadata extract), filters.py, manifest.py (atomic save), clustering.py, template_search.py, materialize.py (parallel A3M→NPZ runner), entries_idx.py (wwPDB TSV parser), openfold3.py (OF3 AtomArray NPZ → Structure), of3_pdb_cache.py (OF3 dataset-cache JSON → mmap'd columnar cache; no LMDB), pack_build.py (`deepfold3 pack` / `pack-verify`)
│   └── module/           # dataset.py (TrainingDataset, InferenceInput), of3_dataset.py (format: of3), of3_pdb_dataset.py (format: of3pdb — OF3 pdb_training_set in place), collate.py (batch)
├── complexlm/       # Complex language model (MLM over multi-chain complexes)
│   ├── config.py         # ComplexLMConfig (layer_pattern = the architecture ablation), Data/Train/Source/Masking
│   ├── vocab.py          # ESM2-compatible alphabet
│   ├── store.py          # ComplexStore / ComplexStoreWriter (mmap'd flat-npy complex store)
│   ├── prepare.py        # structures + paired FASTA -> store; Cβ 8 Å interface flags
│   ├── cluster.py        # mmseqs linclust over all stores + source-overlap report
│   ├── sampler.py        # hierarchical p(D) -> p(T|D) -> cluster -> member
│   ├── crop.py           # interface-biased crop, entity-consistent windows
│   ├── masking.py        # uniform / interface-biased / homomer-paired MLM
│   ├── batch.py          # PackedBatch (one token axis, chain_cu + complex_cu)
│   ├── dataset.py        # packing stream, evaluation plan
│   ├── model.py          # ComplexLM (chain-RoPE / complex-NoPE varlen attention)
│   ├── diagnostics.py    # partner utilisation, permutation sensitivity, attention sink
│   └── trainer.py        # DDP MLM training loop
├── convert/         # Weight conversion
│   └── af3.py            # Haiku → PyTorch state_dict converter
├── io/              # Output serialization
│   ├── background_writer.py # Bounded background thread for logit writes; a failed write fails the run
│   ├── mmcif.py          # save_mmcif() (gemmi-based, pLDDT in B-factors)
│   ├── run_manifest.py   # run_manifest.json: code, versions, GPU, checkpoint digest, run settings
│   └── confidences.py    # Confidence metric computation + save_confidences_json()
├── training/        # Training pipeline
│   ├── loss.py           # smooth_lddt, diffusion_mse, bond, distogram, plddt, pae, pde, exp_resolved, training_loss
│   ├── ema.py            # EMA (constant decay, params-only; matches OF3 ExponentialMovingAverage)
│   ├── scheduler.py      # AF3LRScheduler + CosineAnnealingScheduler
│   └── trainer.py        # DDP Trainer (training loop, checkpointing, eval, WandB)
├── scripts/         # CLI (click group: `deepfold3 <command>`)
│   ├── __init__.py       # `_LazyGroup` click.Group `cli` — unified entry point (lazy submodule import)
│   ├── _common.py        # setup_cli_logging(), require_manifest()
│   ├── build_masks.py    # `deepfold3 build-masks`
│   ├── build_pdb_cache.py # `deepfold3 build-pdb-cache` (parallel per-chain template cache pre-build)
│   ├── convert_weights.py  # `deepfold3 convert`
│   ├── inference.py      # `deepfold3 inference`
│   ├── inspect_example.py # `deepfold3 inspect-example`
│   ├── msa_batch.py      # `deepfold3 msa-batch`
│   ├── msa_search.py     # `deepfold3 msa`
│   ├── pack.py           # `deepfold3 pack`, `pack-verify`
│   ├── plm_prepare.py    # `deepfold3 plm-prepare {structure,pairs,cluster,stats}`
│   ├── plm_train.py      # `deepfold3 plm-train`
│   ├── plot_confidences.py # `deepfold3 plot`
│   ├── prepare.py        # `deepfold3 prepare`, `apply-clusters`, `cluster`, `fetch-ccd`, `build-ccd-cache`, `export-complexes`, `import-msa`, `import-paired-msa`, `import-of3`, `build-of3-cache`, `stamp-template-index-space`, `enrich-manifest`, `drop-obsolete`
│   ├── prepare_templates.py # `deepfold3 prepare-templates`
│   ├── refilter.py       # `deepfold3 refilter`
│   ├── scan_mmcif.py     # `deepfold3 scan-mmcif` (parallel mmCIF metadata cache)
│   ├── train.py          # `deepfold3 train`
│   └── warmup_triton.py  # `deepfold3 warmup-triton`
└── params.py        # read_af3_params() (zstd decompression)

hangang/                     # KISTI Hangang: pack_all.sh, transfer.sh, env.sh, setup_env.sh, check.sbatch, submit.sh + train.sbatch (see hangang/README.md)

scripts/                     # Shell scripts for database setup and GPU server
├── setup_databases.sh       # Download and index UniRef30, ColabFoldDB, PDB100
├── mmseqs_gpuserver.sh      # Start persistent MMseqs2 GPU server processes
├── mmseqs_msa.sh            # Verbatim ColabFold server-side MSA search
└── mmseqs_pair.sh           # Verbatim ColabFold server-side paired MSA search
```

## Key Dependencies

| Package        | Purpose                                              |
| -------------- | ---------------------------------------------------- |
| `torch`        | Core deep learning framework                         |
| `gemmi`        | mmCIF/PDB structure file parsing                     |
| `rdkit`        | Small molecule chemistry (ligand handling, symmetry) |
| `pdbeccdutils` | CCD (Chemical Component Dictionary) utilities        |
| `safetensors`  | Model weight serialization                           |
| `zstandard`    | Compression (model weights are `.zst`)               |
| `click`        | CLI interface                                        |
| `numpy`        | Numerical arrays / structured array storage          |

## Training Pipeline

### Overview

Training uses native PyTorch DDP. The core training loop, losses, EMA, and LR scheduler are implemented in `src/deepfold3/training/`. See `docs/training.md` for full details.

### Forward Pass (per batch)

```
1. Evoformer trunk (random 1–max_recycles recycles, gradient on last only)
2. Diffusion training step (noise sampling + single denoise, with gradient)
3. Mini-rollout (20-step denoise, no gradient → predicted positions for confidence / chain-perm)
4. Confidence head (on mini-rollout positions; gated on `alpha_confidence > 0`, skipped entirely in stages 1–3)
5. Distogram head (cross-entropy on distance bins; all samples incl. distillation)
```

### Loss Weights (AF3 §5.3)

| Loss                        | Weight                                            | Notes                                                                                                                 |
| --------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Diffusion MSE + smooth LDDT | `alpha_diffusion` (4.0)                           | AF3 §5.3 eq. 3: `(1/3)·Σ‖Δ‖²·w·mask / Σ mask` (per-coord avg, atom-count denom). EDM-weighted; mol-type reweight is AF3 eq. 4 additive over base 1 → DNA/RNA 6× (1+5), ligand 11× (1+10). All samples incl. distillation. |
| Bond violation              | `alpha_diffusion × alpha_bond` (0)                | MSE on bond distances vs CCD ideal; enable for stage 2                                                                |
| Shape complementarity       | `alpha_shape_comp` (0)                            | SE(3)-invariant cross-chain protein interface loss (OpenDDE `arXiv:2607.03787` §A.2.1): Huber regression of the predicted surface-complementarity field onto the GT field, GT-only masking, ~8 Å range of action. `shape_comp_huber_delta` is `smooth_l1`'s beta, not `huber`'s delta. OpenDDE's own weight is 3e-2. See `export/shape-complementarity/`. |
| Distogram                   | `alpha_distogram` (0.03)                          | 64-bin cross-entropy. All samples incl. distillation (AF3 §5.2 Table 6).                                              |
| pLDDT, PDE, exp_resolved    | `alpha_confidence` (0 in stages 1–3, 1e-4 in FT3) | PDB-only (distillation skipped, AF3 §4.3); gated on resolution ∈ [0.1, 4.0]. Head forward disabled entirely when α=0. pLDDT j-axis restricted to representative atoms (CA / C1') with 15 Å (protein-rep j) / 30 Å (nucleic-rep j) radius (AF3 §4.3.1). |
| PAE                         | `alpha_confidence × alpha_pae` (0 until FT3)      | Set `alpha_pae=1.0` only in finetune3 (confidence-only stage). Frames within 25° of co-linearity dropped per-sample on both pred and GT (AF3 §4.3.2). |

Confidence head forward is gated on `alpha_confidence > 0` ([model.py:557](src/deepfold3/model/model.py#L557)) — skipped entirely when disabled to avoid `0 × NaN = NaN` contamination of the total loss. Mini-rollout still runs (needed for chain permutation determinism). Distogram loss lives above the `is_distillation` early-return in `training_loss` so distillation samples contribute distogram gradient.

### Optimizer

AdamW or Schedule-Free AdamW with decoupled weight decay split by parameter role ([optim.py:build_optimizer](src/deepfold3/training/optim.py)). **No-decay** group: 1D params (LN scale/shift, biases), per-channel scale-like 2D weights (`min(shape)==1`), gates (`to_g`, `out_gate`, `.gate.`, AdaLN-zero `cond_scale` / `cond_bias` / `zero_cond` incl. atom-transformer `ModuleList` variants), and input embeddings (`single_activations`, `msa_activations`, `embed_*`, `{left,right}_single`, `position_activations`, `bond_embedding`, `prev_*`). **Decay** group: all remaining 2D+ matmul weights (attention QKV / output, FFN, triangle multiplication output, OPM `out_w`, …). See `_NO_DECAY_NAME_SUBSTRINGS` for the substring patterns.

### Configuration Persistence

`CheckpointManager.save` pickles `model_cfg` and `train_cfg` into the `.pt` blob, and `convert_weights.py` embeds the architectural `ModelConfig` as a JSON metadata entry (`SAFETENSORS_MODEL_CFG_KEY = "model_cfg_json"`) in `.safetensors`. `deepfold3 inference` reconstructs the dataclass at load time so flags whose default differs from the trained run (notably `pairformer.deepnet_residual_scale` and `confidence.deepnet_residual_scale` — both default `False` but enabled in `configs/schedule_free.yaml`) propagate correctly. `--optim` flag overrides only `OptimConfig`. Legacy `.safetensors` without metadata and raw AF3 `.bin/.bin.zst` fall back to `ModelConfig()` defaults with a warning. See `docs/training.md` § _Configuration persistence_.

### Token Masks (Loss Control)

Per-token mask channels stored in `data_dir/masks/{pdb_id}.npz`, loaded automatically by `TrainingDataset`. Managed via `TokenMask` (`data/masking.py`). Two reserved channels:

| Channel    | Dtype         | Effect                                                                                                                         |
| ---------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `"fixed"`  | float32 `[N]` | 1.0 = experimental, 0.0 = model-generated. Scales `exp_resolved` loss target (soft CE). For PATCHR ATLAS distillation.         |
| `"region"` | int16 `[N]`   | Region label (0=unset, 1+=CDR-H1/H2/H3 etc.). Consumed by `atomize_source="region"` only — it drives **no** loss weighting (the `region_loss_weights` path was removed as unreachable). |

Generate PATCHR ATLAS masks from inpainting metadata JSON:

```python
from deepfold3.data.masking import TokenMask, MASK_FIXED
fixed = TokenMask.fixed_from_patchr(json_path, tokens, chains, chain_infos=record.chains)
np.savez(masks_dir / f"{pdb_id}.npz", **{MASK_FIXED: fixed})
```

### Multi-Stage Training

Stages are controlled by config overrides, not hardcoded presets. Use `--config` with stage JSON files or `--override key=value`:

```bash
# Stage 1: Initial training (default config)
deepfold3 train --data-dir /data/training --output runs/exp1

# Stage 2: Fine-tune structure (bond loss on, cosine LR)
deepfold3 train --data-dir /data/training --output runs/exp2 \
    --config configs/finetune1_structure_640.json --params runs/exp1/checkpoint_final.pt

# Stage 3: Fine-tune confidence (freeze trunk, enable PAE)
deepfold3 train --data-dir /data/training --output runs/exp3 \
    --config configs/finetune3_confidence.json --params runs/exp2/checkpoint_final.pt
```

### Training Data Format

Every persistent file is **sharded** by the first two characters of its id (256 buckets) so listing, rsync, and inode walks stay fast at 100k+ files. Path construction goes through the helpers in `data/paths.py` (`structure_path`, `mask_path`, `template_path`) and `data/msa/paths.py` (`unpaired_msa_path`, `paired_msa_path`) — never inline.

```
data_dir/
├── manifest.json                                # list of Record objects (atomic save with .old backup)
├── structures/{ab}/{pdb_id}.npz                 # Structure (atoms, bonds, residues, chains)
├── masks/{ab}/{pdb_id}.npz                      # TokenMask channels (optional)
├── templates/{ab}/{pdb_id}.json                 # Multimer template hits (optional)
└── msa/
    ├── unpaired/{ab}/{msa_id}.npz               # per-sequence (SHA256 hash)
    └── paired/{ab}/{record_id}_paired_{chain_hash}.npz  # per-complex
```

Records are loaded via `discover_records()` from `data/prepare/manifest.py` (reads `manifest.json` or recursively scans `structures/`). Each `ChainInfo` has a `msa_id` that maps to the unpaired MSA file name (SHA256 of the chain sequence). `ChainInfo` also stores `auth_chain_name` (original chain name from input CIF, before assembly renaming) for accurate chain mapping in mask generation.

`Record.structure_info` (`resolution`, `method`, `deposited`, `released`) can be populated either at prepare time (from mmCIF metadata) or after the fact via `deepfold3 enrich-manifest --idx entries.idx [--mmcif-cache mmcif_meta.json]`. Use `deepfold3 scan-mmcif` first to generate the cache from a local PDB mirror — that's the only canonical source for `released` (= earliest `_pdbx_audit_revision_history.revision_date`, matching AlphaFold 3's template date-cutoff convention).

### Data Preprocessing Pipeline

Implemented: `deepfold3 prepare` (steps 1 + 2 + 7), `deepfold3 msa` (step 3 for inference), `deepfold3 cluster` (step 2 via MMseqs2), `deepfold3 apply-clusters` (apply pre-computed clusters). Remaining steps (3 batch, 4, 5) need implementation for training.

```bash
# Process specific PDB IDs
deepfold3 prepare --mmcif-dir /scratch/pdb/mmCIF --output /data/training --pdb-ids 1pfe,8qy0,4hhb

# Batch process with filters
deepfold3 prepare --mmcif-dir /scratch/pdb/mmCIF --output /data/training \
    --max-resolution 9.0 --methods "X-RAY DIFFRACTION,ELECTRON MICROSCOPY" --workers 8

# With date cutoff for train/val split
deepfold3 prepare --mmcif-dir /scratch/pdb/mmCIF --output /data/training --max-date 2023-09-30

# Sample N structures
deepfold3 prepare --mmcif-dir /scratch/pdb/mmCIF --output /data/training --limit 100
```

Steps:

1. **Structure processing** (implemented): PDB mmCIF → `gemmi.make_assembly()` (assembly 1) → structure-level filters (resolution, method, date, chain count, residue count, polymer count) → chain-level filters (min/max residue count, all-unknown, consecutive CA distance, chain clashing) → Structure NPZ.
2. **Sequence clustering** (implemented): `deepfold3 cluster` (MMseqs2 at 40% identity) or download pre-computed wwPDB `clusters-by-entity-40.txt`. Apply via `deepfold3 apply-clusters` or `--cluster-file` during prepare.
3. **MSA generation** (implemented): `deepfold3 msa` for inference, `deepfold3 msa-batch` for training. Local MMseqs2 (GPU optional) against UniRef30/ColabFoldDB → A3M → NPZ per unique sequence. Resume support, batch splitting.
4. **Template discovery** (implemented): `deepfold3 prepare-templates` discovers multimer templates from per-chain m8 files (MMseqs2 SEQRES hits). Uses Hungarian algorithm for optimal chain pairing. Writes lightweight JSON index to `{data_dir}/templates/`. Note the pairing only selects *which* template chain is aligned to each query chain — since `34326d5` the cross-chain block is masked out of the template embedding, so the assignment does not feed inter-chain geometry to the model.
5. **ID filtering** (implemented): `--include-list` / `--exclude-list` text files for `deepfold3 train`
6. **Manifest creation** (implemented): collect records/ → `manifest.json`

Source: `src/deepfold3/data/prepare/` (structure.py, filters.py, manifest.py, clustering.py, template_search.py).

PDB mmCIF mirror: `/scratch/pdb/mmCIF/` (layout: `{mid2}/{pdb_id}.cif.gz`, ~250k structures).

### Template Pipeline

Training-time template support uses pre-computed per-chain m8 files (MMseqs2 hits against PDB SEQRES):

1. **Offline discovery**: `deepfold3 prepare-templates` reads m8 files, groups hits by template PDB ID, finds PDB entries covering ≥2 query chains, uses bipartite matching (Hungarian algorithm) for chain pairing, writes `{data_dir}/templates/{pdb_id}.json`. The ≥2-chain requirement and the pairing are a *discovery* convention; the model consumes the result per chain (see the intra-chain mask in step 5).
2. **Training-time loading**: `TrainingDataset` loads template JSON index, reads mmCIF from PDB mirror (`--mmcif-dir`), extracts `TemplateFeatures` via `extract_from_seqres_alignment()` (with seqres→structure realignment), passes `template_feats_per_chain` to featurizer.
3. **Seqres realignment**: `template_realign.py` handles the mismatch between PDB SEQRES (used by search tools) and the actual mmCIF structure (which may have missing/different residues). Applied automatically in both the HHR inference path (`extract_template_features_from_hhr`) and the training CIGAR path (`extract_from_seqres_alignment`).
4. **Template augmentation**: each sample randomly uses `k = min(Uniform[0, n_avail], max_templates)` of the available hits, so **`k = 0` is drawn with probability `1/(n_avail+1)`** — template-free inference is in-distribution, not a train/serve mismatch. Per-chain dropout additionally drops chain alignments from multimer templates (50% per chain, always keeping ≥1), so a query chain may be templated or bare.
5. **Template pair mask is intra-chain only** (`34326d5`): the template pair features are masked by `pb_mask[i] × pb_mask[j] × multichain_mask`, so only the block-diagonal (same-`asym_id`) signal reaches the trunk. Template features are extracted **per chain, each in its own coordinate frame**, so their cross-chain pseudo-beta distances and backbone-frame orientations are not real interface geometry — feeding them unmasked injects noise. Verified: with the mask the template output is invariant to a rigid translation of one chain; without it the output shifts by 2.9. **Consequence: a multimer template index is consumed as a bundle of per-chain templates** — the chain pairing that produced it does not change what the model sees.
6. **Date cutoff filtering**: `dataset.py:_effective_template_cutoff` computes `min(max_template_date, record.released - cutoff_days)` per record and is applied on **both** the monomer and multimer load paths (`ba646ba` re-enabled multimer templates behind this runtime cutoff after they were found leaking ~50% future hits). **Fail-closed**: when no cutoff can be established — the query has no release date and no `default_template_cutoff_date` is configured — templates are dropped rather than used unfiltered. AF3 §2.4: template release ≥ 60 days before example release; distillation max 2018-04-30.
7. **Template index frame**: hit files store template positions either in SEQRES space or in **structure space** (0-based resolved-residue). The two differ by the template's unresolved N-terminal count, which varies per template, so reading one as the other misaligns every hit and raises nothing — templates just become noise. `prepare-templates` therefore **stamps `"index_space"` into each hit**, and the loader uses that; the per-dataset `template_struct_space` flag is only a fallback for index files written before the field existed, and taking it logs a one-shot warning per dataset. `deepfold3 stamp-template-index-space` back-fills the field into an existing index (idempotent, atomic per file, safe to run against a directory a training job is reading) — the stamp is not on every copy: the `/store/deepfold3/data` and Hangang copies of `afdb_nvda` and `teddymer` carry none, so `template_struct_space: true` is what makes them correct, and the `index_space` warning at training start is expected. Verify a new source by re-deriving each hit's sequence identity and comparing it to the identity the search recorded: under the correct frame the medians agree to ~1 pt. Frames in use — `afdb_nvda` and `teddymer` are **structure**-space, `patchr` is mixed per hit (mostly SEQRES, recorded in each hit; measured 2026-10-31); of3 sets (`long_monomers`, `short_monomers`) carry `template.npz` `idx_map` in 1-based **SEQRES** and `OF3Dataset` handles them, ignoring `template_struct_space`. `format: of3pdb` needs no frame flag at all: OF3's `template_structure_arrays` render each template chain over its full SEQRES with `res_id` **as** the SEQRES index, so a hit's `idx_map` (1-based query, template `res_id`) addresses it directly with no mmCIF mirror and no realignment (verified: 99.3% identity on a close homolog, 0.45 pooled on remote validation hits vs 0.07–0.11 for every off-by-one reading). `monomer_template_struct_space` is the deprecated spelling of the flag.
8. **Per-chain cache (`PdbChainCache`)**: training extracts each template chain's query-independent intermediate (`seqres`, `struct_seq`, residue→atom `lookup`) at most once via a two-tier cache — per-worker memory LRU + hierarchical sharded disk (`{cache}/{shard}/{pdb}/{chain}.npz` + `_done` sentinel) — instead of re-parsing the full template mmCIF with gemmi per hit (the dominant, query-size-independent dataloader-stall cost). Pre-build with `deepfold3 build-pdb-cache --output /data/pdb/cache`; enable at train time via `training.pdb_chain_cache_dir`. `mmcif_dir` is the cold-parse fallback for chains absent from the cache.

**Crop-then-load ordering** (`dataset.py:_load_sample`): tokenization reads only the structure (MSA content is never needed to build `msa_col_idx`), so MSA files and templates are loaded **after** cropping, restricted to the chains that survive the crop. A 228-chain assembly cropped to `max_tokens` loads MSA/templates for the few surviving chains, not all 228 — turning O(all chains) I/O into O(cropped chains). `data/crop/cropper.py:compute_chain_msa_cols` derives the per-chain MSA columns to load from the cropped `msa_col_idx`. Tier-4 diagnostics: per-`__getitem__` phase timing (`slow_sample_log_threshold_s`) + per-step `data_wait_s` (dataloader-wait) logged by the trainer / WandB.

The inference template pipeline (`data/parse/templates.py`) supports per-chain HHR templates and whole-complex JSON templates.

## Confidence Metrics

The confidence head produces four raw outputs per diffusion sample:

- **pLDDT** `[S, N, M*50]` — per-atom confidence (50 bins → expected value in [0, 100])
- **PAE** `[S, N, N, 64]` — predicted aligned error (64 bins, max 31 Å)
- **PDE** `[S, N, N, 64]` — predicted distance error (symmetrised, same bins as PAE)
- **exp_resolved** `[S, N, M*2]` — per-atom 2-class (not-resolved, resolved)

Derived metrics computed in `io/confidences.py`:

- **pTM**: max_i Σ_j (tm_adjusted_pae[i,j] × normed_mask[i,j]), d_0 = 1.24×(max(N,19)-15)^(1/3)−1.8
- **ipTM**: same as pTM but restricted to cross-chain pairs
- **Ranking score**: 0.8×ipTM + 0.2×pTM (single-chain: just pTM)
- **Contact probs**: from distogram logits, P(d < 8 Å)
- **gPDE**: global average PDE over all token pairs
