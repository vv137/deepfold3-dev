"""Batch inference — load the model once, run many JSON inputs back-to-back.

Use case: trajectory / benchmark sweeps that would otherwise re-load the
5.8 GB checkpoint for every query. Model load is ~12-15 s on B200; for a
sweep of 40 queries that's ~10 minutes purely in setup. This command
amortises it across the full input list.

Input format — text file, one query per line:

    {json_path} [TAB {output_dir}]

If only ``json_path`` is given on a line, the output directory is
``{output_template}`` rendered with ``{json_dir}`` (parent of the JSON
file), ``{json_stem}`` (file stem) and ``{params_stem}`` (parameter file
stem; useful for putting all sweep outputs under a per-checkpoint dir).
Blank lines and lines starting with ``#`` are ignored.

Usage::

    uv run deepfold3 inference-batch \\
        --inputs-file inputs.txt \\
        --params runs/df3-fapeS3/checkpoint_10000.pt \\
        --num-samples 30 \\
        --seed 42 --seed 43 --seed 44 --seed 45 \\
        --snapshot average \\
        --output-template '{json_dir}/inference/{params_stem}'

``--seed`` is repeatable: each input is folded once per seed into its own
``seed-{seed}_sample-{s}`` subdir (S seeds x num_samples structures). With
``--skip-existing`` a job is skipped only when every requested (seed, sample)
is already on disk, and any individually-complete seed is skipped, so an
interrupted multi-seed sweep resumes seed-by-seed.

Per-job options (num-recycles, num-samples, seed, plot-confidences,
template-cache, pdb-mmcif-dir) apply uniformly to every line.
"""

from __future__ import annotations

import dataclasses
import json
import time
import traceback
from pathlib import Path

import click
import torch

from deepfold3.data.msa.pairing import MSA_PAIRING_MODES
from deepfold3.io.run_manifest import checkpoint_record, gpu_name, new_manifest, sha256_file, write_manifest
from deepfold3.model import kernel_census
from deepfold3.model.config import OPTIM_PRESETS
from deepfold3.model.guidance import check_guidance_training, has_residue_ranges
from deepfold3.model.restraints import RestraintSpec, build_restraints, describe_restraint
from deepfold3.scripts import cli
from deepfold3.scripts._common import apply_det_recipe
from deepfold3.scripts.inference import (
    _describe_guidance,
    _parse_guidance,
    _parse_repels,
    _parse_restraint_pairs,
    _parse_restraints,
)


def _load_model_once(
    params: Path,
    snapshot: str,
    dev: torch.device,
    optim: str,
    msa_crop_size: int | None = None,
    matmul_precision: str = "auto",
    allow_missing_keys: bool = False,
):
    """Load weights + build model. Returns ``(model, model_cfg)``.

    Same as the top half of ``inference``. Extracted so batch loops can
    amortise the ~12 s checkpoint-read cost.
    """
    from deepfold3.model.config import ModelConfig, OptimConfig
    from deepfold3.model.model import DeepFold3
    from deepfold3.scripts._common import check_loaded_keys, load_pt_snapshot

    click.echo(f"[batch] loading parameters from {params} …")
    t0 = time.perf_counter()
    ckpt_model_cfg = None
    if params.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        from deepfold3.model.config import model_cfg_from_metadata

        with safe_open(str(params), framework="pt") as f:
            ckpt_model_cfg = model_cfg_from_metadata(f.metadata())
        state_dict = load_file(str(params))
    elif params.suffix == ".pt":
        state_dict, ckpt_model_cfg = load_pt_snapshot(params, snapshot)
    else:
        import warnings

        from deepfold3.convert.af3 import haiku_to_torch
        from deepfold3.params import read_af3_params

        warnings.warn(
            "Loading raw AF3 params — Fourier weights will be missing.",
            stacklevel=1,
        )
        state_dict = haiku_to_torch(read_af3_params(params))

    model_cfg = ckpt_model_cfg if ckpt_model_cfg is not None else ModelConfig()
    if ckpt_model_cfg is None and params.suffix == ".pt":
        click.echo("[batch] checkpoint has no embedded model_cfg — using ModelConfig() defaults")

    try:
        model_cfg.optim = OptimConfig.from_preset(optim, ckpt_model_cfg.optim if ckpt_model_cfg is not None else None)
    except ValueError as e:
        raise click.UsageError(str(e)) from e

    # See ``inference``: default follows the MSA pool cap, not the training crop.
    from deepfold3.data.const import MAX_MSA_ROWS

    want = msa_crop_size if msa_crop_size is not None else MAX_MSA_ROWS
    if want != model_cfg.msa_crop_size:
        prev = model_cfg.msa_crop_size
        model_cfg.msa_crop_size = want
        src = "flag" if msa_crop_size is not None else "default (MSA pool cap)"
        click.echo(f"[batch] msa_crop_size {prev} -> {want} per recycle [{src}]")

    if matmul_precision != "auto":
        torch.set_float32_matmul_precision(matmul_precision)
        click.echo(f"[batch] float32 matmul precision: {matmul_precision}")
    elif optim and optim != "none":
        torch.set_float32_matmul_precision("high")
    if optim and optim != "none":
        click.echo(f"[batch] optimizations: {optim}")

    model = DeepFold3(model_cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    check_loaded_keys(missing, unexpected, allow_missing=allow_missing_keys)
    model.to(dev).eval()
    click.echo(f"  loaded in {time.perf_counter() - t0:.1f}s")
    return model, model_cfg


def _run_one_job(
    model,
    model_cfg,
    dev: torch.device,
    json_input: Path,
    output_dir: Path,
    *,
    num_recycles: int,
    num_samples: int,
    seeds: list[int],
    template_cache: Path | None,
    sample_chunk: int | None = None,
    pdb_mmcif_dir: Path | None,
    plot_confidences: bool,
    msa_pairing: str = "auto",
    save_distogram: bool = False,
    logits_scheme: str = "int8_softmax",
    skip_existing: bool,
    bucket: bool = False,
    precision: str = "fp32",
    guidance_specs: tuple = (),
    allow_overlap: bool = False,
    restraint_specs: tuple = (),
    restraint_weight: float = 0.0,
    restraint_rigid: bool = True,
    restraint_max_step: float = 2.0,
    restraint_after_step: bool = False,
    restraint_after_sigma: float = 16.0,
    manifest_fields: dict | None = None,
    sweep: list[tuple[str, Path, tuple]] | None = None,
) -> None:
    """Featurize + forward + save for a single JSON job. Mirrors the JSON
    branch of ``inference``.

    ``sweep`` lists ``(name, output_dir, guidance_specs)`` entries that replace
    ``output_dir`` / ``guidance_specs``: the trunk runs once per seed and every
    entry samples from it, re-seeded, so each entry is what a separate run with
    that guidance would give (the trunk does not depend on guidance).

    ``seeds`` is the CLI-provided seed list; when empty each JSON's own
    ``modelSeeds`` are used. Every seed is folded independently and written to
    its own ``seed-{seed}_sample-{s}`` subdir, so a job with S seeds and
    ``num_samples`` samples yields S x num_samples structures. When
    ``skip_existing`` is set, any seed whose full set of sample CIFs already
    exists is skipped (per-seed resume)."""
    from deepfold3.data.module.dataset import InferenceInput
    from deepfold3.data.parse.json_input import describe_msa_pairing, parse_json_input
    from deepfold3.io.confidences import save_ranking_scores_csv, save_summary_confidences_json

    click.echo(f"[batch] featurizing {json_input} …")
    t0 = time.perf_counter()
    jobs = parse_json_input(
        json_input, template_cache_dir=template_cache, pdb_mmcif_dir=pdb_mmcif_dir, msa_pairing=msa_pairing
    )
    if not jobs:
        click.echo(f"[batch] skipping {json_input}: no jobs in JSON")
        return
    job = jobs[0]
    job_name = job.name
    click.echo(f"[batch] MSA pairing: {describe_msa_pairing(job)} (--msa-pairing {msa_pairing})")
    job_title = job.description or job.name

    # Chain names as written in the input JSON, in asym_id order — otherwise the
    # output is renamed A, B, C, … and name-keyed consumers (bench per-interface
    # ids, chain_pair_iptm) silently mismatch.
    from deepfold3.data.module.dataset import job_chain_ids
    from deepfold3.io.mmcif import resolve_chain_name

    chain_names = job_chain_ids(job)

    chain_idx = 0
    desc_map: dict[str, str] = {}
    for chain_list in [job.protein_chains, job.dna_chains, job.rna_chains, job.ligand_chains]:
        for chain in chain_list:
            n_expanded = len(chain.ccd_ids) if getattr(chain, "ccd_ids", None) else 1
            for _ in range(n_expanded):
                if chain.description:
                    desc_map[resolve_chain_name(chain_idx, chain_names)] = chain.description
                chain_idx += 1
    entity_descriptions = desc_map or None

    feats = InferenceInput.from_job(job)
    N = feats["res_type"].shape[0]
    S = feats["msa"].shape[0]
    T = feats.get("template_aatype", torch.zeros(0)).shape[0]
    click.echo(f"  {N} tokens, {S} MSA sequences, {T} templates  ({time.perf_counter() - t0:.1f}s)")

    original_n = N
    # Bucket-pad N up to the next BUCKET_SIZES entry. torch.compile *requires*
    # this (static shapes), but the fused Triton kernels benefit just as much:
    # they autotune on M = batch·N², so every distinct token count is a fresh
    # ~60 s autotune. Quantizing N to ~13 buckets collapses hundreds of shapes
    # to a handful, so a sweep autotunes each bucket once (and warmup-triton can
    # pre-warm them) instead of recompiling per target. Enabled by --bucket or
    # whenever torch.compile is on.
    use_bucket = bucket or model_cfg.optim.use_torch_compile
    if use_bucket:
        from deepfold3.model.bucket import next_bucket, pad_features

        bucket_n = next_bucket(N)
        if bucket_n > N:
            feats, original_n = pad_features(feats, bucket_n)
            click.echo(f"  bucket padding: {N} → {bucket_n} tokens")

    feats = {k: v.to(dev) for k, v in feats.items()}

    entries = sweep if sweep is not None else [("", output_dir, guidance_specs)]

    def _terms(specs: tuple):
        if not specs:
            return None
        from deepfold3.io.mmcif import resolve_chain_name as _resolve
        from deepfold3.model.guidance import build_guidance_terms

        chain_index = {_resolve(a, chain_names): a for a in feats["asym_id"][:original_n].unique().tolist()}
        try:
            return build_guidance_terms(
                specs,
                feats["asym_id"],
                chain_index,
                original_n,
                residue_index=feats.get("residue_index"),
                allow_overlap=allow_overlap,
            )
        except ValueError as e:
            raise click.ClickException(f"--guidance on {json_input.name}: {e}") from e

    entry_terms = [_terms(specs) for _, _, specs in entries]

    restraints = ()
    if restraint_specs:  # already merged by the command below
        from deepfold3.io.mmcif import resolve_chain_name as _resolve2

        chain_index = {_resolve2(a, chain_names): a for a in feats["asym_id"][:original_n].unique().tolist()}
        try:
            restraints = build_restraints(
                restraint_specs,
                feats["asym_id"],
                chain_index,
                original_n,
                residue_index=feats.get("residue_index"),
            )
        except ValueError as e:
            raise click.ClickException(f"--restraint on {json_input.name}: {e}") from e

    json_seeds = list(job.model_seeds) if job.model_seeds else [42]
    # CLI seeds (when provided) are authoritative — replace the JSON's own
    # seeds so a benchmark sweep folds exactly the requested set. Dedup while
    # preserving order.
    seed_list = list(dict.fromkeys(seeds)) if seeds else json_seeds

    # One manifest per entry: a sweep's outputs then say for themselves which
    # weights, code and flags produced them. A job restricted to some seeds
    # (an inputs-file ``seeds=`` column) shares its directory with jobs running
    # the other seeds, so its manifest is per seed.
    manifest_name = "run_manifest.json"
    if seeds and sweep is not None and len(seed_list) == 1:
        manifest_name = f"run_manifest.seed-{int(seed_list[0])}.json"
    manifests = []
    for _, od, specs in entries:
        od.mkdir(parents=True, exist_ok=True)
        m = new_manifest(
            json_input=str(json_input),
            seeds=[int(x) for x in seed_list],
            num_tokens=original_n,
            padded_tokens=feats["res_type"].shape[0],
            msa_rows=S,
            msa_pairing={"requested": msa_pairing, "resolved": job.msa_pairing},
            guidance=[{**dataclasses.asdict(g), "description": _describe_guidance(g)} for g in specs],
            restraints=[
                {**dataclasses.asdict(r), "description": describe_restraint(r), "weight": restraint_weight}
                for r in restraint_specs
            ],
            precision=precision,
            **(manifest_fields or {}),
        )
        write_manifest(od / manifest_name, m)
        manifests.append(m)
    kernel_census.reset_counts()

    summaries: list[list[dict]] = [[] for _ in entries]
    total = num_samples  # samples per seed on disk
    chunks = sample_chunks(num_samples, sample_chunk)
    chunk = chunks[0][2]

    for current_seed in seed_list:
        # Judge a seed on every file it owes, not just model.cif: a run killed
        # between the structure and its confidences would otherwise be "done"
        # with the rest missing for good.
        todo = []  # (entry, batch) pairs still owed
        for k, (name, od, _) in enumerate(entries):
            if skip_existing and all(p.exists() for p in _seed_outputs(od, current_seed, total, save_distogram)):
                click.echo(f"[batch]   {name + ' ' if name else ''}seed {current_seed}: complete — skip")
                summaries[k] += _existing_summaries(od, current_seed, total)
                continue
            for c, start, size in chunks:
                done = skip_existing and all(
                    p.exists() for p in _seed_outputs(od, current_seed, size, save_distogram and c == 0, start)
                )
                if done:
                    summaries[k] += _existing_summaries(od, current_seed, size, start)
                else:
                    todo.append((k, c, start, size))
        if not todo:
            continue
        passes = f" in {len(chunks)} passes of {chunk}" if len(chunks) > 1 else ""
        click.echo(
            f"[batch] {json_input.name}  seed={current_seed} samples={num_samples}{passes} recycles={num_recycles}"
        )
        t0 = time.perf_counter()
        with (
            torch.inference_mode(),
            torch.autocast(feats["res_type"].device.type, dtype=torch.bfloat16, enabled=precision == "bf16"),
        ):
            orig_num_samples = model.diffusion_head.cfg.num_samples
            model.diffusion_head.cfg.num_samples = num_samples
            try:
                trunk = model.run_evoformer(feats, num_recycles, seed=current_seed)
            finally:
                model.diffusion_head.cfg.num_samples = orig_num_samples
        if len(entries) > 1:
            click.echo(f"  trunk {time.perf_counter() - t0:.1f}s")
        distogram = None
        for k, c, start, size in todo:
            name, od, _ = entries[k]
            t1 = time.perf_counter()
            with (
                torch.inference_mode(),
                torch.autocast(feats["res_type"].device.type, dtype=torch.bfloat16, enabled=precision == "bf16"),
            ):
                model_output = model.sample_from_trunk(
                    feats,
                    trunk,
                    seed=sample_chunk_seed(current_seed, c),
                    num_samples=size,
                    guidance_terms=entry_terms[k],
                    restraints=restraints,
                    restraint_weight=restraint_weight,
                    restraint_rigid=restraint_rigid,
                    restraint_max_step=restraint_max_step,
                    restraint_after_step=restraint_after_step,
                    restraint_after_sigma=restraint_after_sigma,
                    distogram=distogram,
                )
            distogram = model_output.distogram

            iter_feats = feats
            if use_bucket and original_n < feats["res_type"].shape[0]:
                from deepfold3.model.bucket import unpad_features, unpad_output

                model_output = unpad_output(model_output, original_n)
                iter_feats = unpad_features(feats, original_n)
            label = (f"{name} " if name else "") + (f"chunk {c} " if len(chunks) > 1 else "")
            click.echo(f"  {label}done in {time.perf_counter() - (t1 if len(entries) > 1 else t0):.1f}s")
            summaries[k] += _save_seed(
                od,
                model_output,
                iter_feats,
                current_seed,
                size,
                sample_offset=start,
                save_distogram=save_distogram and c == 0,
                logits_scheme=logits_scheme,
                plot_confidences=plot_confidences,
                entity_descriptions=entity_descriptions,
                job_title=job_title,
                chain_names=chain_names,
            )
            # Free this entry's outputs before the next sample. Otherwise
            # ``model_output`` (PAE/PDE [S, N, N, 64] fp32, ~16 GiB at N=2560)
            # stays alive until the next one returns: at N=2560 the first fits
            # and the second OOMs in the confidence head.
            del model_output, iter_feats
            if feats["res_type"].is_cuda:
                torch.cuda.empty_cache()
        del trunk, distogram
        if feats["res_type"].is_cuda:
            torch.cuda.empty_cache()

    for k, (_, od, _) in enumerate(entries):
        if sweep is not None and seeds and len(seed_list) == 1:
            # Other jobs hold the other seeds of this directory: list every seed on disk.
            summaries[k] = _all_summaries_on_disk(od, total)
        if summaries[k]:
            save_ranking_scores_csv(od / "ranking_scores.csv", summaries[k])
            summary_name = f"{job_name}_summary_confidences.json" if job_name else "summary_confidences.json"
            save_summary_confidences_json(od / summary_name, summaries[k], job_name=job_name)
        manifests[k]["status"] = "ok"
        manifests[k]["kernels"] = kernel_census.snapshot()
        manifests[k]["kernels_shadowed"] = kernel_census.shadowed()
        manifests[k]["kernels_inactive"] = kernel_census.inactive()
        write_manifest(od / manifest_name, manifests[k])


def _save_seed(
    output_dir: Path,
    model_output,
    iter_feats: dict,
    seed: int,
    num_samples: int,
    *,
    sample_offset: int = 0,
    save_distogram: bool,
    logits_scheme: str,
    plot_confidences: bool,
    entity_descriptions,
    job_title: str,
    chain_names,
) -> list[dict]:
    """Write one seed's structures and confidences; returns the per-sample summaries."""
    from deepfold3.io.confidences import save_confidences_json, save_distogram_logits
    from deepfold3.io.mmcif import save_mmcif

    # Trunk distogram is shared across this seed's samples (no sample axis),
    # so write it once per seed rather than into every sample dir.
    if save_distogram:
        distogram_npz = output_dir / f"seed-{int(seed)}_distogram.npz"
        if save_distogram_logits(distogram_npz, model_output, scheme=logits_scheme, seed=int(seed)):
            click.echo(f"  saved  {distogram_npz.name}")

    out = []
    for s in range(num_samples):
        sample_dir = output_dir / f"seed-{int(seed)}_sample-{int(s + sample_offset)}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_mmcif(
            sample_dir / "model.cif",
            model_output,
            iter_feats,
            entity_descriptions=entity_descriptions,
            title=job_title,
            sample_idx=s,
            chain_names=chain_names,
        )
        conf_path = sample_dir / "confidences.json"
        summary = save_confidences_json(
            conf_path,
            model_output,
            iter_feats,
            sample_idx=s,
            seed=int(seed),
            chain_names=chain_names,
        )
        out.append({**summary, "seed": int(seed), "sample": int(s + sample_offset)})
        if plot_confidences:
            from deepfold3.scripts.plot_confidences import plot_confidences_file

            plot_confidences_file(conf_path, conf_path.with_suffix(".png"))
    return out


def sample_chunks(num_samples: int, chunk: int | None) -> list[tuple[int, int, int]]:
    """``(chunk index, first sample index, size)`` of each diffusion pass; ``chunk=None`` is one pass."""
    size = min(chunk or num_samples, num_samples)
    return [(c, c * size, min(size, num_samples - c * size)) for c in range(-(-num_samples // size))]


def sample_chunk_seed(seed: int, chunk: int) -> int:
    """Diffusion-noise seed of sample chunk ``chunk`` of model seed ``seed``; chunk 0 is the seed itself.

    Later chunks step by a large prime so they do not land on another model seed's chunk 0 in the usual
    seed ranges, and stay well below 2**63 for ``torch.Generator.manual_seed``.
    """
    return int(seed) + 1_000_003 * int(chunk)


def _all_summaries_on_disk(output_dir: Path, num_samples: int) -> list[dict]:
    """Summaries of every seed of ``output_dir`` whose samples are all written."""
    seeds = sorted({int(p.name.split("_")[0][5:]) for p in output_dir.glob("seed-*_sample-*") if p.is_dir()})
    out = []
    for seed in seeds:
        if all((output_dir / f"seed-{seed}_sample-{s}" / "confidences.json").exists() for s in range(num_samples)):
            try:
                out += _existing_summaries(output_dir, seed, num_samples)
            except click.ClickException:
                continue  # being written by another job right now
    return out


def _seed_outputs(output_dir: Path, seed: int, num_samples: int, save_distogram: bool, start: int = 0) -> list[Path]:
    """Every file one seed of a job must leave behind (samples ``start .. start+num_samples-1``), in write order."""
    out: list[Path] = []
    if save_distogram:
        out.append(output_dir / f"seed-{int(seed)}_distogram.npz")
    for s in range(start, start + num_samples):
        d = output_dir / f"seed-{int(seed)}_sample-{int(s)}"
        out += [d / "model.cif", d / "confidences.json"]
    return out


def _existing_summaries(output_dir: Path, seed: int, num_samples: int, start: int = 0) -> list[dict]:
    """The per-sample summaries of a seed already on disk.

    ``ranking_scores.csv`` and ``summary_confidences.json`` cover the whole job,
    so a resumed run has to re-read the seeds it skips or it would publish only
    the seeds of this run.

    Raises:
        click.ClickException: A ``confidences.json`` is unreadable or has no
            summary, which means the skip decision was wrong.
    """
    out = []
    for s in range(start, start + num_samples):
        path = output_dir / f"seed-{int(seed)}_sample-{int(s)}" / "confidences.json"
        try:
            summary = json.loads(path.read_text())["summary"]
        except (OSError, ValueError, KeyError) as e:
            raise click.ClickException(f"{path} is present but unusable ({e}); delete the seed and re-run") from e
        out.append({**summary, "seed": int(seed), "sample": int(s)})
    return out


def _parse_inputs_file(path: Path, output_template: str, params: Path) -> list[tuple[Path, Path, dict[str, str]]]:
    """Read a TSV of ``json_path [TAB output_dir [TAB key=value ...]]`` lines.

    The ``key=value`` columns are per-line options: ``seeds=42,66`` folds only
    those seeds (overriding ``--seed``), and with ``--guidance-sweep``
    ``group=NAME`` picks the sweep group and ``pairs=H:A,L:A`` fills its
    ``{pairs}`` placeholder. Columns are tab-separated when the line has a tab,
    else the line is ``json_path [output_dir]`` split on whitespace.

    Returns ``[(json_path, output_dir, extras), ...]``.
    """
    out = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("\t")] if "\t" in line else line.split(None, 1)
        json_path = Path(parts[0])
        if len(parts) >= 2 and parts[1]:
            output_dir = Path(parts[1])
        else:
            output_dir = Path(
                output_template.format(
                    json_dir=str(json_path.parent),
                    json_stem=json_path.stem,
                    json_name=json_path.name,
                    params_stem=params.stem,
                )
            )
        extras = {}
        for col in parts[2:]:
            key, eq, value = col.partition("=")
            if not eq:
                raise click.UsageError(f"{path}: expected key=value, got {col!r} in line {raw!r}")
            extras[key] = value
        out.append((json_path, output_dir, extras))
    return out


def _load_sweep(path: Path) -> dict[str, list[tuple[str, tuple[str, ...]]]]:
    """``--guidance-sweep`` file: ``{"groups": {GROUP: [{"name": N, "guidance": [W@TARGET, ...]}, ...]}}``.

    Each entry is one guidance setting; its outputs go to the line's output
    directory with ``{cfg}`` replaced by the entry name. ``{pairs}`` in a
    guidance value is replaced by the line's ``pairs=`` column.
    """
    data = json.loads(path.read_text())
    groups = {}
    for group, items in data["groups"].items():
        entries = []
        for it in items:
            entries.append((str(it["name"]), tuple(str(g) for g in it.get("guidance", ()))))
        names = [n for n, _ in entries]
        if len(set(names)) != len(names):
            raise click.UsageError(f"--guidance-sweep: duplicate entry names in group {group!r}")
        groups[group] = entries
    return groups


def _sweep_entries(
    groups: dict[str, list[tuple[str, tuple[str, ...]]]], output_dir: Path, extras: dict[str, str]
) -> list[tuple[str, Path, tuple]]:
    """The ``(name, output_dir, guidance_specs)`` entries one inputs-file line runs."""
    from deepfold3.model.guidance import parse_guidance_spec

    group = extras.get("group")
    if group not in groups:
        raise click.UsageError(f"--guidance-sweep: line {output_dir} has group={group!r}; known: {sorted(groups)}")
    if "{cfg}" not in str(output_dir):
        raise click.UsageError(f"--guidance-sweep: output dir {output_dir} has no {{cfg}} placeholder")
    out = []
    for name, texts in groups[group]:
        if any("{pairs}" in t for t in texts) and not extras.get("pairs"):
            raise click.UsageError(f"--guidance-sweep: entry {name!r} needs pairs= on line {output_dir}")
        specs = tuple(parse_guidance_spec(t.replace("{pairs}", extras.get("pairs", ""))) for t in texts)
        out.append((name, Path(str(output_dir).replace("{cfg}", name)), specs))
    return out


@cli.command("inference-batch")
@click.option(
    "--inputs-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Text file: 'json_path [TAB output_dir]' per line, '#' comments allowed",
)
@click.option(
    "--json",
    "json_paths",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Repeatable JSON input path. Output dir from --output-template.",
)
@click.option(
    "--output-template",
    default="{json_dir}/inference",
    help="Output dir template when only a JSON path is provided. "
    "Variables: {json_dir}, {json_stem}, {json_name}, {params_stem}.",
)
@click.option(
    "--params",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Model parameters (.safetensors / .pt / .bin / .bin.zst)",
)
@click.option(
    "--msa-crop-size",
    type=int,
    default=None,
    help=(
        "Per-recycle MSA rows for the Evoformer. Default: follow the MSA pool cap (``const.MAX_MSA_ROWS`` = 16384) rather than the checkpoint's value — training deliberately uses a small crop for throughput and randomises the effective row count (AF3 s2.3 k ~ Uniform[1, crop]) so the model is row-count invariant, and serving is meant to use the whole pool. Pass a smaller number to reproduce a training-crop run or to bound memory."
    ),
)
@click.option("--num-recycles", default=10, show_default=True)
@click.option("--num-samples", default=5, show_default=True)
@click.option(
    "--sample-chunk",
    default=5,
    show_default=True,
    type=click.IntRange(min=1),
    help="Samples per diffusion pass. The trunk runs once per seed and the --num-samples samples are drawn "
    "in passes of this many, so a large --num-samples fits in memory (5 fits on the largest FoldBench targets "
    "on a 95 GB GH200, 25 does not). Pass 0 is seeded "
    "with the model seed, pass c > 0 with sample_chunk_seed(seed, c): the first chunk matches a run of "
    "--num-samples equal to the chunk size, but the full set depends on the chunk size.",
)
@click.option("--snapshot", type=click.Choice(["auto", "shadow", "average", "fast"]), default="auto", show_default=True)
@click.option(
    "--seed",
    "seeds",
    multiple=True,
    type=int,
    help="Repeatable model seed (e.g. --seed 42 --seed 43). Each input is folded "
    "once per seed into its own seed-*_sample-* subdir. When given, replaces "
    "each JSON's modelSeeds; defaults to the JSON's own seeds.",
)
@click.option("--device", default=None)
@click.option("--template-cache", default=None, type=click.Path(path_type=Path))
@click.option("--pdb-mmcif-dir", default=None, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--msa-pairing",
    default="auto",
    type=click.Choice(MSA_PAIRING_MODES),
    show_default=True,
    help="How each chain's pairedMsa is used (JSON input). precomputed: rows are already paired across chains (row i of every chain is one paired row; ColabFold/MMseqs2 pairaln output). species: pairedMsa is each chain's raw UniProt hits (AF3 semantics); rows are paired by species as AF3/OpenFold3 do, and unpaired rows repeating a paired row are dropped. auto: species only when >=2 distinct protein chains have paired MSAs whose headers carry UniProt species ids and whose row counts differ, else precomputed.",
)
@click.option(
    "--optim",
    default="inference",
    type=click.Choice(OPTIM_PRESETS),
    show_default=True,
)
@click.option(
    "--matmul-precision",
    default="auto",
    type=click.Choice(["auto", "highest", "high", "medium"]),
    show_default=True,
    help=(
        "float32 matmul precision. 'auto' = TF32 whenever --optim is not 'none' "
        "(historical behaviour); 'highest' forces true fp32 matmuls."
    ),
)
@click.option("--plot-confidences", is_flag=True, default=False)
@click.option(
    "--save-distogram/--no-save-distogram",
    default=False,
    show_default=True,
    help="Persist the per-seed trunk distogram (seed-*_distogram.npz, shared "
    "across that seed's samples). Cheap ([N,N,64], one per seed); needed for "
    "distogram post-analysis (e.g. cross-chain interface-contact prediction).",
)
@click.option(
    "--logits-scheme",
    default="int8_softmax",
    type=click.Choice(["float32", "float16", "int8", "int8_softmax", "int4_softmax"]),
    show_default=True,
    help="Compression scheme for --save-distogram npz. int8_softmax preserves "
    "softmax(logits) within ~1e-4 at ~1 byte/value.",
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
    help="Skip jobs whose output dir already has any seed-* subdir.",
)
@click.option(
    "--bucket/--no-bucket",
    default=True,
    show_default=True,
    help="Pad token count N up to the next bucket size before the forward. The "
    "fused Triton kernels autotune on M=batch*N^2, so bucketing collapses "
    "hundreds of distinct N to ~13 shapes — each autotuned once (and "
    "pre-warmable via warmup-triton) instead of a ~60s re-autotune per target. "
    "Recommended for multi-target sweeps; auto-on when torch.compile is used.",
)
@click.option(
    "--precision",
    default="fp32",
    type=click.Choice(["fp32", "bf16"]),
    show_default=True,
    help="Compute dtype of the trunk, as ``inference --precision``.",
)
@click.option(
    "--hoist-sampler/--no-hoist-sampler",
    default=True,
    show_default=True,
    help="Hoist the diffusion sampler's step-invariant conditioning, as ``inference --hoist-sampler``.",
)
@click.option(
    "--guidance",
    "guidance_specs",
    multiple=True,
    metavar="W[@TARGET[@LO..HI]]",
    callback=_parse_guidance,
    help="Classifier-free guidance term, as ``inference --guidance``; applied to every job.",
)
@click.option(
    "--guidance-sweep",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "JSON of guidance settings to run from one trunk per seed (see ``_load_sweep``). Each inputs-file "
        "line names its group (group=…), its output dir carries a {cfg} placeholder, and {pairs} in a "
        "guidance value takes the line's pairs=… column. Not combinable with --guidance."
    ),
)
@click.option(
    "--allow-guidance-overlap",
    is_flag=True,
    default=False,
    help=(
        "Accept --guidance selections (as ``inference``) that drop the same pair block twice: a union inside one "
        "term, and summed weights across terms (otherwise an error and a warning)."
    ),
)
@click.option(
    "--restraint",
    "restraint_specs",
    multiple=True,
    metavar="SEL:SEL[@DIST[@LO..HI]]",
    callback=_parse_restraints,
    help="Distance restraint steered into the sampler, as ``inference --restraint``; applied to every job.",
)
@click.option(
    "--restraint-rigid/--restraint-residues",
    default=True,
    show_default=True,
    help=(
        "Rigid: project the restraint force onto each chain's six rigid degrees of freedom, "
        "so the chain is translated and rotated and its internal geometry is untouched. "
        "--restraint-residues pulls the named residues alone, which stretches the chain "
        "where the selection ends."
    ),
)
@click.option(
    "--restraint-after-step/--restraint-before-step",
    default=False,
    show_default=True,
    help=(
        "Where the restraint acts. Before (default): it moves the predicted clean structure, "
        "so the Euler step carries ~70 % of it and the next denoiser call may put the chain "
        "back — the restraint competes with the model and the final structure is a model "
        "output. After: it moves the coordinates themselves, which the model cannot undo, and "
        "which nothing supervises — a rigid push knows nothing about clashes."
    ),
)
@click.option(
    "--restraint-after-sigma",
    default=16.0,
    show_default=True,
    help=(
        "With --restraint-after-step, stop pushing below this σ̂ so the rest of the schedule "
        "is pure denoising and repairs what the push broke; σ_data by default. The last step "
        "never pushes whatever this is set to, so the structure returned is always a denoiser "
        "output. Set it low to hold the placement to the end: letting go at σ̂ = 16 left 68 "
        "steps in which the model walked 7uvs all the way back (fnat 0.42 → 0.000)."
    ),
)
@click.option(
    "--restraint-max-step",
    default=2.0,
    show_default=True,
    help="Å cap on the displacement one sampler step may apply per restraint (steering needs the clip).",
)
@click.option(
    "--restraint-pair",
    "restraint_pair_specs",
    multiple=True,
    metavar="SEL:SEL[@DIST[@LO..HI]]",
    callback=_parse_restraint_pairs,
    help=(
        "Residue-to-residue restraint: the selectors are read in the order written and "
        "matched position by position (C/118+120+145:A/31+52+101 is three pairs), so both "
        "sides must name the same number of residues. Unlike --restraint this fixes WHICH "
        "residue touches which, which is what orients a chain on a site: three well-separated "
        "pairs pin all six rigid freedoms. Default bottom 6 Å; native contacts run 3.6 Å median, "
        "so 4-5 is the useful range."
    ),
)
@click.option(
    "--repel",
    "repel_specs",
    multiple=True,
    metavar="SEL:SEL[@DIST[@LO..HI]]",
    callback=_parse_repels,
    help=(
        "The opposite bound: every residue of the left selector must stay at least DIST Å "
        "from the right group. Use it to push a prediction off a site it keeps choosing."
    ),
)
@click.option(
    "--restraint-weight",
    default=0.2,
    show_default=True,
    help="Fraction of each restraint's remaining violation removed per sampler step.",
)
@click.option(
    "--det",
    is_flag=True,
    default=False,
    help="Deterministic inference, as ``inference --det``.",
)
@click.option(
    "--allow-missing-keys",
    is_flag=True,
    default=False,
    help="Run even if the weights leave model parameters at random initialisation.",
)
def inference_batch(
    inputs_file: Path | None,
    json_paths: tuple[Path, ...],
    output_template: str,
    params: Path,
    msa_crop_size: int | None,
    num_recycles: int,
    num_samples: int,
    sample_chunk: int | None,
    snapshot: str,
    seeds: tuple[int, ...],
    device: str | None,
    template_cache: Path | None,
    pdb_mmcif_dir: Path | None,
    msa_pairing: str,
    optim: str,
    matmul_precision: str,
    precision: str,
    hoist_sampler: bool,
    guidance_specs: tuple,
    guidance_sweep: Path | None,
    allow_guidance_overlap: bool,
    restraint_specs: tuple[RestraintSpec, ...],
    restraint_pair_specs: tuple[RestraintSpec, ...],
    repel_specs: tuple[RestraintSpec, ...],
    restraint_weight: float,
    restraint_rigid: bool,
    restraint_max_step: float,
    restraint_after_step: bool,
    restraint_after_sigma: float,
    det: bool,
    plot_confidences: bool,
    save_distogram: bool,
    logits_scheme: str,
    skip_existing: bool,
    bucket: bool,
    allow_missing_keys: bool,
) -> None:
    """Run inference on many JSON inputs without reloading the model."""
    if not inputs_file and not json_paths:
        raise click.UsageError("Provide --inputs-file or one or more --json")

    jobs: list[tuple[Path, Path, dict[str, str]]] = []
    if inputs_file:
        jobs += _parse_inputs_file(inputs_file, output_template, params)
    for jp in json_paths:
        out_dir = Path(
            output_template.format(
                json_dir=str(jp.parent),
                json_stem=jp.stem,
                json_name=jp.name,
                params_stem=params.stem,
            )
        )
        jobs.append((jp, out_dir, {}))

    sweep_groups = None
    if guidance_sweep is not None:
        if guidance_specs:
            raise click.UsageError("--guidance-sweep replaces --guidance; give one or the other")
        sweep_groups = _load_sweep(guidance_sweep)
        job_entries = [_sweep_entries(sweep_groups, od, extras) for _, od, extras in jobs]
    else:
        job_entries = [None for _ in jobs]

    click.echo(f"[batch] {len(jobs)} jobs queued")

    if det:
        apply_det_recipe()

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # MSA depth varies by input even with --bucket: see ``set_nearest_shape_lookup``.
    from deepfold3.model.triton.cache import set_nearest_shape_lookup

    set_nearest_shape_lookup(True)
    click.echo(f"[batch] device: {dev}")

    model, model_cfg = _load_model_once(
        params, snapshot, dev, optim, msa_crop_size, matmul_precision, allow_missing_keys=allow_missing_keys
    )
    model_cfg.diffusion.hoist_sampler = hoist_sampler
    if guidance_specs:
        try:
            check_guidance_training(guidance_specs, model_cfg.diffusion)
        except ValueError as e:
            raise click.UsageError(f"--guidance: {e}") from e
        click.echo(f"[batch] guidance: {' + '.join(_describe_guidance(g) for g in guidance_specs)}")
    if sweep_groups is not None:
        seen = set()
        for entries in job_entries:
            for name, _, specs in entries:
                if name in seen:
                    continue
                seen.add(name)
                try:
                    check_guidance_training(specs, model_cfg.diffusion)
                except ValueError as e:
                    raise click.UsageError(f"--guidance-sweep {name}: {e}") from e
        for group, items in sweep_groups.items():
            click.echo(f"[batch] sweep group {group}: " + ", ".join(n for n, _ in items))
    all_restraint_specs = tuple(restraint_specs) + tuple(restraint_pair_specs) + tuple(repel_specs)
    if all_restraint_specs:
        click.echo(
            f"[batch] restraints (weight {restraint_weight:g}): "
            + " + ".join(describe_restraint(r) for r in all_restraint_specs)
        )
        if has_residue_ranges(guidance_specs):
            click.echo("[batch] note: a residue-subset drop is finer than the training drop (whole chain pairs)")

    checkpoint = checkpoint_record(params)
    checkpoint["sha256"] = sha256_file(params)
    manifest_fields = {
        "device": str(dev),
        "gpu": gpu_name(dev),
        "checkpoint": checkpoint,
        "num_samples": num_samples,
        "sample_chunk": sample_chunk or num_samples,
        "num_recycles": num_recycles,
        "msa_crop_size": model_cfg.msa_crop_size,
        "bucket": bucket or model_cfg.optim.use_torch_compile,
        "hoist_sampler": hoist_sampler,
        "optim": optim,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "logits_scheme": logits_scheme,
        "det": det,
    }

    seed_list = list(dict.fromkeys(seeds))

    def _job_complete(od: Path, job_seeds: list[int]) -> bool:
        """A job is complete when every requested (seed, sample) CIF exists.

        With CLI seeds we know the exact set; without them we can't cheaply
        know the JSON's own seeds, so fall back to the any-seed-dir heuristic.
        """
        if not od.exists():
            return False
        if job_seeds:
            return all(p.exists() for s in job_seeds for p in _seed_outputs(od, s, num_samples, save_distogram))
        return any(p.name.startswith("seed-") for p in od.iterdir() if p.is_dir())

    n_done = 0
    n_skipped = 0
    oom: list[str] = []
    t_total = time.perf_counter()
    for idx, ((jp, od, extras), entries) in enumerate(zip(jobs, job_entries, strict=True), 1):
        job_seeds = [int(x) for x in extras["seeds"].split(",")] if extras.get("seeds") else seed_list
        dirs = [e[1] for e in entries] if entries is not None else [od]
        if skip_existing and all(_job_complete(d, job_seeds) for d in dirs):
            click.echo(f"[batch] [{idx}/{len(jobs)}] SKIP {jp.name} → {od} (all seeds/samples present)")
            n_skipped += 1
            continue
        if not jp.exists():
            click.echo(f"[batch] [{idx}/{len(jobs)}] MISS {jp} — not found")
            continue
        click.echo(f"[batch] [{idx}/{len(jobs)}] {jp.name} → {od}")
        try:
            _run_one_job(
                model,
                model_cfg,
                dev,
                jp,
                od,
                num_recycles=num_recycles,
                num_samples=num_samples,
                sample_chunk=sample_chunk,
                seeds=job_seeds,
                template_cache=template_cache,
                pdb_mmcif_dir=pdb_mmcif_dir,
                msa_pairing=msa_pairing,
                plot_confidences=plot_confidences,
                save_distogram=save_distogram,
                logits_scheme=logits_scheme,
                skip_existing=skip_existing,
                bucket=bucket,
                precision=precision,
                guidance_specs=guidance_specs,
                allow_overlap=allow_guidance_overlap,
                restraint_specs=tuple(restraint_specs) + tuple(restraint_pair_specs) + tuple(repel_specs),
                restraint_weight=restraint_weight,
                restraint_rigid=restraint_rigid,
                restraint_max_step=restraint_max_step,
                restraint_after_step=restraint_after_step,
                restraint_after_sigma=restraint_after_sigma,
                manifest_fields=manifest_fields,
                sweep=entries,
            )
            n_done += 1
        except torch.OutOfMemoryError as e:
            # One target too large for the GPU must not take the rest of the
            # shard down with it. Seeds it finished stay on disk for --skip-existing.
            click.echo(f"[batch] [{idx}/{len(jobs)}] OOM {jp} — {str(e).splitlines()[0]}")
            # Where it ran out: the model frames, innermost last (no locals kept).
            for fr in [f for f in traceback.extract_tb(e.__traceback__) if "site-packages" not in f.filename][-10:]:
                click.echo(f"[batch]     at {fr.filename}:{fr.lineno} in {fr.name}: {fr.line}")
            oom.append(str(jp))
        # Release this job's cached blocks back to the driver before the next
        # target. Bucketing folds varied N (=> varied allocation sizes), and
        # per-shape Triton autotune allocates scratch, so cached blocks
        # fragment across a long shard and intermittently OOM the next (large)
        # target. The job's tensors are already freed by refcounting on return
        # (inference_mode => no autograd graph / cycles), so empty_cache alone
        # suffices — no gc.collect needed.
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t_total
    click.echo(f"[batch] finished: {n_done} done, {n_skipped} skipped, {len(oom)} OOM  ({elapsed:.1f}s total)")
    if oom:
        raise click.ClickException(f"{len(oom)} job(s) ran out of GPU memory: " + ", ".join(oom))


if __name__ == "__main__":
    inference_batch()
