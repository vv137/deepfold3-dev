# Training on Hangang (KISTI 6th supercomputer)

Hangang's Lustre `/scratch` gives a group **80 TB and 5M files**, purges files
not read for **15 days**, and its GPU nodes are **aarch64 GH200 ×4**. The
unpacked training data is ~63M files, so it moves as **packs**: each bulky
subtree becomes `_pack/` (a few thousand tar shards + a sorted index, see
[`src/deepfold3/data/pack.py`](../src/deepfold3/data/pack.py)). The loaders read
packs transparently — configs and code paths are unchanged, only `/data` →
`/scratch/$USER/deepfold3/data`.

```
local (source host)                          Hangang
───────────────────                          ───────
/store/deepfold3/data   ──pack_all.sh──▶  /store/deepfold3/packed  ──transfer.sh──▶  /scratch/$USER/deepfold3/data
(63M files, symlinks)                     (~6k files, ~40 TB)       (data mover)     (same tree)
```

| Script | Where | What |
| --- | --- | --- |
| `pack_all.sh` | source host | build the packed staging tree (resumable, verifies each pack) |
| `transfer.sh` | source host | `connect` (one OTP) → `push` (parallel resumable rsync) → `verify` (sha256 on the data mover); `pull` / `runs` bring the training output back |
| `env.sh` | Hangang | paths, uv, Slurm comment, NCCL/Slingshot settings (`df3_slingshot_env`: aws-ofi-nccl module, hsn0 rendezvous; multi-node only) — `DF3_SLURM_*` defaults to `field=sciai;appl=pytorch-ddp` |
| `setup_env.sh` | Hangang GPU login node | aarch64 venv via `uv sync --frozen`, then torch swapped to the `cu126` build (driver is CUDA 12.8); afterwards never `uv run`/`uv sync` there |
| `check.sbatch` | 1 GPU node | GPUs, NCCL all-reduce, pack integrity, file count, training DataLoader smoke test |
| `warmup.sbatch` | 1 GPU node | fills the Triton autotune cache (`$HOME/.cache/deepfold3-triton`) for a training config, once before `submit.sh` |
| `submit.sh` + `train.sbatch` | Hangang | training as a chain of 24 h jobs with auto-resume |
| `keepalive.sh` | Hangang | report files close to the 15-day purge (optionally refresh atime) |
| `smoke_data.py` | (called by `check.sbatch`) | builds the real DataLoader from a config and times it |

## 1. Build the packed tree (source host)

```bash
cd /store/deepfold3/deepfold3-dev
bash hangang/pack_all.sh                     # all steps; re-run to resume
ONLY="patchr_training afdb_nvda" bash hangang/pack_all.sh
```

`$DEST` (default `/store/deepfold3/packed`) mirrors `/data`:

```
packed/
├── components.cif.gz  ccd_cache.json  mmcif_meta.json
├── pdb/cache/_pack                       template chain cache (mmCIF + assemblies)
├── pdb/divided/_pack                     mmCIF mirror (the of3 sets parse it for templates)
├── pdb_training_set/{_pack, df3_cache/, df3_cache_val/}
└── distillation/
    ├── long_monomers/preprocessed/{_pack, ids.txt}
    ├── short_monomers/preprocessed/{_pack, ids.txt}
    ├── afdb_nvda/{_pack, manifest.json}
    ├── teddymer/{_pack, manifest.json}
    └── patchr_training/{_pack, manifest.json, ...}
```

Each `_pack/` holds `meta.json`, `keys.npy`, `loc.npy`, `SHA256SUMS`,
`shards/*.tar` and a local-only `_build/` (resume state, not transferred).
Shards are plain tar files: `tar tf shards/00000.tar` lists the original paths.

## 2. Transfer (source host → Hangang data mover)

```bash
export HANGANG_USER=<kisti-id>
bash hangang/transfer.sh connect       # OTP + password once; the master connection lives 24 h
bash hangang/transfer.sh push          # STREAMS=8 parallel rsync; re-run to resume
bash hangang/transfer.sh verify        # every shard's sha256, every other file's sha256
bash hangang/transfer.sh status        # anything missing or short
```

Push a small set first (`ONLY="distillation/patchr_training" bash hangang/transfer.sh push`)
to measure the line rate; at 1 GB/s the full ~40 TB takes ~11 h, at 100 MB/s ~5 days.
The data mover (`hangang-dm.ksc.re.kr`) has no CPU-time limit, unlike the login nodes.

### Bringing the checkpoints back

```bash
bash hangang/transfer.sh runs                        # remote sizes, and what is missing here
ONLY="df3-stage2-001" bash hangang/transfer.sh pull  # one run
bash hangang/transfer.sh pull                        # every run
```

`pull` goes `$RUNS_REMOTE` (`/scratch/$USER/deepfold3/runs`) → `$RUNS_LOCAL`
(`/store/deepfold3/runs`) and takes the same `ONLY` as `push`, so a single run
comes back without the smoke tests' checkpoints. `PULL_EXCLUDE` defaults to
`wandb .wandb _logs *.tmp`; WandB writes to `$REPO/wandb`, not under `runs/`, so
that excludes nothing today and is there for a run that writes its own.

Nothing is deleted locally, so a checkpoint kept here after the 15-day purge
took the remote copy (`keepalive.sh`) stays. A checkpoint still being written
has the wrong size, is skipped, and arrives on the next `pull` rather than
landing truncated. One stream measured 480 MB/s on 5 GB checkpoints — the line,
not rsync, so `pull` does not parallelise.

## 3. Environment and pre-flight (Hangang)

```bash
ssh <kisti-id>@hangang.ksc.re.kr                       # GPU login node (aarch64)
cd /scratch/$USER/deepfold3/deepfold3-dev               # repo copied/cloned here
vi hangang/env.sh                                       # optional: DF3_SLURM_FIELD/APPL (check showappl)
bash hangang/setup_env.sh
sbatch hangang/check.sbatch                             # carries --comment in its #SBATCH header
```

Then warm the Triton autotune cache for the config you will train, so the
training ranks do not each benchmark every kernel shape on their first steps:

```bash
CONFIG_SRC=configs/stage2.yaml sbatch --comment="$(source hangang/env.sh; df3_slurm_comment)" hangang/warmup.sbatch
```

It must end with `=== warmup done`. Re-run it after changing the model widths,
`max_tokens` or `batch_size_per_rank`; already-cached shapes are skipped.

`check.sbatch` must end with `=== check passed`. In its smoke-test section
every dataset should report `pack` (not `plain`/`MISSING`), and the per-dataset
sample counts should show `msa>1` and `templates` for most samples.

## 4. Train

```bash
bash hangang/submit.sh df3-hangang-001                  # 64 nodes = 256 GH200, 5 × 24 h links
NODES=16 EXTRA_ARGS="--wandb deepfold3-stage0" bash hangang/submit.sh df3-hangang-002
```

`train.sbatch` writes `$RUNS/<run>/config.yaml` once (from `configs/initial.yaml`
with `/data/` → `$DATA/`) and every link reuses it. Ten minutes before the
walltime the job asks the ranks to checkpoint (SIGTERM → `ShutdownCoordinator`);
the next link resumes from the newest checkpoint (`train.sbatch` forces
`auto_resume=true`; the configs default it to false).

Starting from another run's weights, with a different seed:

```bash
CONFIG_SRC=configs/stage2.yaml \
EXTRA_ARGS="--params ${RUNS}/<stage1-run>/checkpoint_<step>.pt --seed 202 --wandb deepfold3-stage2" \
    bash hangang/submit.sh df3-stage2-001
```

- `--params` loads model weights only (`ckpt["model"]` from a `.pt`, or a
  `.safetensors`); step, optimizer and LR schedule start fresh. Every link
  passes it, but once the run directory has a checkpoint, auto-resume
  overwrites those weights together with the optimizer state and step, so
  only the first link actually starts from `--params`.
- `--seed` sets `training.seed` (data sampling order and init RNG); keep it
  the same on every link of a chain, which `EXTRA_ARGS` does automatically.
- `EXTRA_ARGS` travels through `sbatch --export`, so it must not contain a
  comma.
- `CONFIG_SRC` is read only by the first link, which freezes it into
  `$RUNS/<run>/config.yaml`; to change a started run's config, edit that file.
  `EXTRA_ARGS` is passed to every link, and re-running `submit.sh` to extend a
  chain uses that invocation's `EXTRA_ARGS` for the new links.

## Things to confirm on the machine

- **Slingshot NCCL plugin.** Multi-node jobs load `aws-ofi-nccl/1.20.0` +
  `libfabric/2.3.1` and KISTI's NCCL settings (`df3_slingshot_env` in
  `env.sh`). The job log must show `NCCL INFO Using network AWS Libfabric`;
  `Using network Socket` means the plugin did not load. The plugin runs
  against the venv's NCCL (cu12 wheel); KISTI validated it with their own
  NCCL, so a version mismatch would show up here first.
- **Rendezvous.** `MASTER_ADDR` is the first node's hostname (KISTI's
  example), which resolves to the management network (`bond0`). If init times
  out with `TCP client failed to connect ... <node>:<port>`, resubmit with
  `DF3_RDZV_HSN=1` in the environment to rendezvous over the first node's
  `hsn0` address instead.
- **PyPI from the login node.** `setup_env.sh` needs it once; if blocked, build
  the venv inside the KISTI PyTorch container instead.
- **Purge policy.** Training keeps touched shards fresh; `keepalive.sh` reports
  what is ageing. Ask KISTI before automating `--touch`. Anything purged can be
  re-sent with `transfer.sh status` + `push`.
- **Quota.** `check.sbatch` prints the file count under `$DF3_ROOT` and
  `lfs quota`; the data is ~6k files, the venv ~30–50k.

## Known pitfalls (found during the first stage2 launch, 2026-10-29)

Each of these failed a 64-node submission or cost a debugging round. The fix
is already in the scripts unless the item says otherwise.

| Symptom | Cause | Fix |
|---|---|---|
| `sbatch: error: --gres-flags=allow-task-sharing is only used with srun.` | Slurm 26.05 accepts the flag only on `srun` | on the `srun` line in `train.sbatch` / `check.sbatch` |
| `FileNotFoundError: Config file not found: …/runs/<run>/stage1.yaml` | the frozen per-run config was a plain copy, so a relative `base:` was looked up in the run dir | `df3_make_config` resolves the `base:` chain into one self-contained YAML |
| `The NVIDIA driver on your system is too old (found version 12080)` | driver 570.124.06 is CUDA 12.8; the lockfile's PyPI torch is the CUDA 13 build | `setup_env.sh` reinstalls the same torch version as `+cu126` (`DF3_TORCH_CUDA`). **Never `uv run` / `uv sync` on Hangang**: both revert it |
| Every chain link would restart from step 0 | configs inherit `auto_resume: false` | `train.sbatch` passes `--override auto_resume=true` |
| Job `.out` stops after `No primary data_dir specified …` and looks hung | Python block-buffers stdout under `srun` | `PYTHONUNBUFFERED=1` in `env.sh`; `$RUNS/<run>/train.log` (rank 0) was always current |
| `TCP client failed to connect … <node>:<port>` after 600 s, `Connection refused` on that port | Slurm does not put task 0 on the first host of `SLURM_JOB_NODELIST`: rank 0's TCPStore listened on node0228 while `MASTER_ADDR` was node0061 | `srun_rank.sh` derives `RANK` from the node's position in the node list, so rank 0 is always on `MASTER_ADDR`. KISTI's `torchrun --rdzv-backend=c10d` example avoids this by electing the store host by name |
| Multi-node NCCL silently on TCP sockets | the venv's NCCL wheel has no Slingshot plugin | `df3_slingshot_env` (multi-node only): `aws-ofi-nccl/1.20.0` + `libfabric/2.3.1`, `NCCL_PROTO=^LL128` (LL128 gave wrong AllReduce results in KISTI's tests). Check for `Using network AWS Libfabric` |

Facts worth knowing:

- **Network:** node hostnames resolve to the management network (`bond0`, `10.161.64.x`, `*.hostmgmt`); Slingshot is `hsn0..3` (`10.169.x`). Rendezvous over the hostname works once rank 0 is on the right node; `DF3_RDZV_HSN=1` switches it to the first node's `hsn0` address.
- **Not yet verified (as of 2026-10-29):** whether the venv's NCCL (2.29.3, cu12 wheel) works with KISTI's `aws-ofi-nccl/1.20.0`; KISTI validated the plugin with the NCCL in their containers. If the log shows `Using network Socket` or `Failed to initialize any NET plugin`, the fallback is the Slingshot image `/apps/common/kisti-container/images/slingshot/kisti-pytorch-25.03-gh200-r1-aarch64.sif` via `kisti-container --runtime singularity --nccl ofi` (torch 2.7-era: check API / Triton compatibility first).
- **Slurm limits:** partition `normal`, 24 h walltime, 5 active jobs per user (a full chain uses all 5, so run `check` / `warmup` before `submit.sh`), jobs submitted only from `/scratch` or `/appsdata`, `--comment="field=…;appl=…"` mandatory.
- **Filesystem:** 4 KiB pages; `/scratch` (Lustre) honours `posix_fadvise(DONTNEED)` and `fincore` reports it correctly; files untouched for 15 days are purged.
- **`EXTRA_ARGS`** travels through `sbatch --export`, so no commas.
