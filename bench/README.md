# bench/ — FoldBench evaluation of df3 checkpoint series

Tracks DeepFold3 structure quality on **FoldBench-small** (`num_tokens <= 1024`,
a subset of [FoldBench](https://github.com/BEAM-Labs/FoldBench)) across a
training trajectory, scored with OpenStructure (OST) and stratified by DSSP
secondary-structure fold class.

Categories: monomer (`monomer_protein`, `monomer_rna`, `monomer_dna`) and
interface (`interface_protein_protein`, `interface_antibody_antigen`,
`interface_protein_dna`, `interface_protein_rna`, `interface_protein_peptide`).
Each target is folded with **4 seeds** (FoldBench convention) × `--num-samples`
diffusion samples; aggregates are best-of-N over all seed×sample structures.
`interface_protein_ligand` builds but is not swept by default (no ligand-aware
OST scorer yet). Nucleic chains carry no MSA, so interface inputs recover their
DNA/RNA partners straight from the ground-truth CIF.

## Model series

A *series* is one snapshot lineage. The current one is **`df3v1`** (the
FAPE/`fapeK5` run). Future retrains are `df3v2`, `df3v3`, … — each gets its
own `ckpts/<series>/` and `runs/<series>/`; nothing else changes.

`df3v1` is sampled on a **5k-step grid, 10k–90k** (17 checkpoints). The
checkpoints live in `runs/` and are symlinked here, latest first in the sweep
so good results surface early.

> **df3v1 has no trained confidence head** (`alpha_confidence = 0`). pTM /
> pLDDT / `ranking_score` are therefore meaningless and must NOT be used to
> rank or judge samples. Eval reports confidence-free aggregates:
> `mean` (over samples) and `oracle` (best-of-N by the metric itself).
> It is a mid-training model — modest LDDT is expected; the trajectory is
> the signal.

## Layout

```
bench/
├── ckpts/<series>/step_<N>.pt        # symlinks -> real runs/.../checkpoint_<N>.pt
├── data/                             # symlinks to read-only sources
│   ├── targets        -> /data/FoldBench/targets
│   ├── ground_truths  -> /data/FoldBench/examples/ground_truths
│   └── msa            -> /data/msa/foldbench/merged   ({pdb}__{chain}.a3m, m8, paired/)
├── inputs/
│   ├── <category>/<pdb>.json         # generated deepfold3 inference inputs
│   ├── <category>_manifest.csv       # pdb_id, num_tokens, in_small (<=1024)
│   └── <category>_dssp.csv           # pdb_id, pct_H/E/C, fold_class
├── runs/<series>/
│   ├── pred/step_<N>/<pdb>/seed-<s>_sample-<i>/model.cif
│   ├── eval/step_<N>/{prediction_reference.csv, raw/, detail/, summary.csv}
│   ├── trajectory_<category>.csv     # step x mean LDDT/TM/GDT
│   ├── byclass_<category>.csv        # step x fold-class LDDT
│   └── trajectory_<category>.png
└── scripts/
```

## Pipeline (all categories; monomer + interface)

```bash
cd /home/jovyan/work/deepfold3-dev
PY=.venv/bin/python

# 1. inputs + FoldBench-small filter  (once per category; <=1024 tokens default)
$PY bench/scripts/build_inputs.py  --category monomer_protein
# 2. DSSP fold classes                (monomer_protein only)
$PY bench/scripts/dssp_classify.py --category monomer_protein
# 3. inference — 4 GPUs, 4 seeds, latest first ('all' = full grid)
$PY bench/scripts/run_sweep.py --series df3v1 --steps 90000 --category monomer_protein --seed 42,43,44,45

# serve fewer MSA rows than the pool (reproduce a training-crop run, or bound memory)
$PY bench/scripts/run_sweep.py --series df3v1 --steps 90000 --category monomer_protein --msa-crop-size 1024
# 4. score with OST (per step)
$PY bench/scripts/eval_ost.py  --series df3v1 --step 90000 --category monomer_protein
# 5. trajectory table + plot (after several steps scored)
$PY bench/scripts/summarize.py --series df3v1 --category monomer_protein
```

**End-to-end automation** — `auto.sh` polls the live training dir and, for each
new 1k-grid checkpoint, sweeps every configured category (build → sweep → OST →
summarize; monomer_protein also gets the DSSP suite):

```bash
nohup bash bench/scripts/auto.sh runs/df3-stage0-r5 3600 \
      > bench/runs/df3v1/logs/auto.log 2>&1 &
# Env: SERIES, CATEGORIES, SEEDS (default 42,43,44,45), NUM_SAMPLES, MAX_TOKENS (1024)
```

## Scripts

| script | role |
|--------|------|
| `build_inputs.py`  | targets CSV + MSA → per-target deepfold3 JSON + token-count manifest |
| `dssp_classify.py` | mkdssp on GT CIF → `fold_class` ∈ {alpha, beta, alpha_beta, coil} |
| `run_sweep.py`     | length-balanced target shards × N GPUs, one `inference-batch` per GPU (loads weights once, `--snapshot average`); resumable |

> **MSA rows.** `inference` / `inference-batch` default `msa_crop_size` to the pool cap
> (`const.MAX_MSA_ROWS` = 16384), **not** the checkpoint's trained crop — training uses a small crop
> (r14: 1024) plus AF3 §2.3's random-`k` mask precisely so serving can use everything. `run_sweep.py`
> and `slurm_sweep.py` take `--msa-crop-size N` and pass it through. Predictions made before this
> default landed were served at the training crop; **do not mix them with new ones inside a step** —
> `--skip-existing` will happily leave the old ones in place, so purge the step first (see the
> per-step purge note) if you change the setting mid-sweep.
| `make_pred_ref.py` | prediction tree → FoldBench `prediction_reference.csv` |
| `eval_ost.py`      | `ost compare-structures` per sample (FoldBench-faithful flags) → mean/oracle LDDT, TM, GDT-TS |
| `external_sets.py` | 2026ARK-AB / PXM-2025 (`ark_antibody_antigen`, `px_antibody_antigen`, `px_protein_protein`) for `eval_ost.py`: chain names, interfaces, trimmed ground truths |
| `summarize.py`     | per-step + per-fold-class trajectory CSVs and plot |
| `kfold_bench.py`   | external MSA-free baseline: bench inputs → AtlasFold / K-Fold queries → `runs/kfold/pred` tree (series `kfold`, step 0); see `docs/kfold-foldbench.md` |
| `kfold_run.sh`     | resumable end-to-end K-Fold run: prepare → validate queries → predict → OST → compare |
| `compare_series.py`| per-category AF3 vs a df3 checkpoint vs K-Fold on the common target set, seed-matched; `--after` / `--exclude-ids` controls |
| `disto_probe.py`   | per-(target,step) **trunk** distogram recall / distoMed vs **structure** realization — separates "trunk knows the contact" from "sampler builds it" |
| `disto_traj.py`    | **distoMed trajectory** over checkpoints (2 arms side-by-side) + plot — the trunk β-distance *consolidation gauge* |

## Distogram / trunk-distance probe (`distoMed`)

These two scripts read the **distogram head straight off `confidences.json`** —
i.e. the trunk (PairFormer) pair-representation's belief about pairwise
distances, **bypassing the diffusion sampler**. Use them to ask *"does the
trunk encode the right inter-residue distances?"* independently of whether the
diffusion head realizes them in 3D.

**Data each sample writes** (`.../seed-<s>_sample-<i>/confidences.json`):
`distogram` `[N,N]` (expected distance, 64 bins 2.31–21.69 Å → caps ~21.7 Å) and
`contact_probs` `[N,N]` (P(d < 8 Å)). Both are trunk readouts (one per
forward, identical across diffusion samples). GT comes from
`tmp/bench/<target>/_ref.cif`.

**`distoMed`** = median, over a target's true long-range contacts
(GT **CB–CB** < 8 Å, |seq i−j| ≥ 6 — dominated by β-sheet register / tertiary
contacts), of the trunk's predicted distance.
`distoMed ≈ 6–8 Å` → trunk *knows* the sheet; `≈ 13–18 Å` → trunk thinks they're
far (no opinion). It is **deterministic** (no diffusion noise) so far less
noisy than TM/GDT; it is a per-target **consolidation gauge**.

> **Rep atom = CB (Gly → CA).** The distogram is trained on CB–CB distances
> (`featurizer.py` `token_to_rep_atom`, `loss.py` `gt_rep`). Both scripts use CB;
> **do not use CA** (it silently shifts every number).

```bash
PY=.venv/bin/python
# per-checkpoint trunk-vs-sampler breakdown (by seq-sep band lives in disto_probe internals)
$PY bench/scripts/disto_probe.py <arm> 3000            # ARM, comma-steps optional
$PY bench/scripts/disto_probe.py df3-stage1-crop640-average 2500,2750,3000

# distoMed trajectory across all step_* checkpoints of two arms → PNG
$PY bench/scripts/disto_traj.py                         # default: base df3v1 vs crop640
$PY bench/scripts/disto_traj.py <arm_left> <arm_right>  # → tmp/bench/ss_analysis_full/distoMed_trajectory.png
```

**Reading it.** `disto_probe` prints `TRUNK recall` (contact_probs > 0.5 on GT
contacts) and `distoMed` next to `STRUCT recall` (fraction realized in the
generated CIF): trunk-high / struct-low ⇒ sampler bottleneck; both low ⇒ the
**trunk** lost the distances. `disto_traj` is the **plateau watch** — while
distoMed is still descending the trunk β representation is still consolidating;
flat slope = converged. (This probe is how the stage1-crop640 β-regression was
diagnosed as a *trunk* distance-rep collapse, band-localized to seq-sep 6–23;
see `memory/project_stage1_crop640`.)

> Targets are hard-coded to three mainly-β cases (`9EDF`, `9OT9`, `29MW`);
> edit the `targets=[…]` list in-script to probe others. Arms/steps are CLI args.

## Conventions / gotchas

- **External sets (2026ARK-AB, PXM-2025).** Inputs and ground truths are read from the
  shared benchmark folder (`DF3_GW_BENCH`, default `/store/deepfold/users/gunwoong/Benchmark`,
  never written). The model holds only the chains of the scored interfaces, so `eval_ost.py`
  scores against a copy of the ground truth trimmed to the `chain_map.csv` chains
  (`data/ground_truths_trimmed/<category>/`, with a sha256 manifest) and passes the
  model→reference chain correspondence to OST explicitly (`-c`). CSV names such as `B.1`
  resolve to the RCSB copy label `B-2` by the same rule as the set's `verify_mapping.py`.
  Against the full assembly the interface DockQ is unchanged but global lDDT/TM fall
  (9qrv sample: lDDT 0.85 → 0.64).
- **8 PXM-2025 PPI targets never score** (13 of 882 interfaces; same at every checkpoint, so
  cross-checkpoint comparisons are unaffected). OST cannot score them under the explicit mapping:
  - `8u8e`, `9dlp`, `9dlv`: the scored partner chain holds `A1AMM`, which OST's compound library
    does not know (SEQRES mismatch). The FoldBench rescue would drop exactly that chain.
  - `9kkq`, `9kmn`: the partner chain is all `UNK` in the ground truth, so OST cannot chem-group it.
  - `9bj5`, `9cd2`, `9r2z`: two copies of one protein that the PDB splits into two entities
    (a methyl-lysine at different positions, or one extra N-terminal Met). The model chains are
    identical, so OST puts both in one chem group and rejects the mapping. Automatic mapping
    "succeeds" but maps only one chain, so the interface has no DockQ either.
- **Re-deriving a result.** Every eval dir gets `provenance.json` (commit, dirty flag, OST
  command and version, flags, and the external-set CSV hashes). `DF3_BENCH=<bench tree>` runs a
  clean checkout's scripts against an existing `runs/` and `data/`, e.g.
  `DF3_BENCH=/store/deepfold/users/vv137/foldbench/deepfold3-dev/bench python bench/scripts/eval_ost.py ...`.
- **OST binary**: `/home/jovyan/.conda/envs/foldbench/bin/ost` (2.8.0). The
  conda env's bundled `python` can't import its own pip deps, so `eval_ost.py`
  calls the `ost` CLI directly from our venv and parses the JSON.
- **Metric**: monomer = LDDT (primary), TM-score, GDT-TS. FoldBench's
  ranking-score "rank" metric is unavailable here (no confidence head).
- **FoldBench-small**: `num_tokens <= 384`. monomer_protein = 261 / 334 targets.
- **Resumable**: inference skips targets that already have `seed-*` dirs;
  OST skips comparisons whose detail JSON exists. Re-run to fill gaps.
- **Sources are read-only** (`/data/...`); everything written stays under `bench/`.
- **5 targets never score** — `8c5h`, `8dtn`, `8ts0` (antibody_antigen) and
  `8ows`, `8q00` (protein_protein) return OST `status=FAILURE` with
  `IndexError: arrays used as indices must be of integer (or boolean) type`
  (a numpy-2 incompatibility inside OST, not a model or structure problem).
  Verified 0 successes in 5,970 comparisons across r14/r15/r16 × 79 steps ×
  every seed/sample, so it is independent of prediction quality. Effective
  denominators are therefore **209/211 for protein_protein and 92/95 for
  antibody_antigen**, constant across every step and series — cross-arm
  comparisons are unaffected, but the target set differs from FoldBench's
  official rows by these 5 when comparing against published numbers.
- **`n_ost_failed` counts comparisons, not targets** — one target contributes
  `seeds x samples` failures (20 for the default 2x5 x2 grid). A 3.2% failure
  rate in `summary.csv` is 3 targets, not 3% of targets.
- **Partial steps get a completion marker.** `auto.sh` treats
  `eval/<cat>/step_<N>/summary.csv` as "done", and it is written even when the
  sweep lost shards (e.g. a CUDA OOM killed 3 of 8 GPUs at step 14000, leaving
  134/211 protein_protein targets). `plot_lddt_grid.py` then averages whatever
  exists, silently. **Always check `n_targets` in `lddt_by_category.csv`
  against the previous step before reading the newest column**; if it dropped,
  the step is partial. To force a redo, delete the `summary.csv` marker (and
  the `pred/` dirs too if the inputs themselves changed — the sweep is
  `--skip-existing`).
```
