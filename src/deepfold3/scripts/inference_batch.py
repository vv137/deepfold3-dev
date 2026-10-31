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
from pathlib import Path

import click
import torch

from deepfold3.io.run_manifest import checkpoint_record, gpu_name, new_manifest, sha256_file, write_manifest
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

    if optim == "all":
        model_cfg.optim = OptimConfig.all_enabled()
    elif optim == "recommended":
        model_cfg.optim = OptimConfig.recommended()
    elif optim == "inference":
        model_cfg.optim = OptimConfig.inference()
    elif optim == "triton":
        model_cfg.optim = OptimConfig.triton_only()

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
    pdb_mmcif_dir: Path | None,
    plot_confidences: bool,
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
) -> None:
    """Featurize + forward + save for a single JSON job. Mirrors the JSON
    branch of ``inference``.

    ``seeds`` is the CLI-provided seed list; when empty each JSON's own
    ``modelSeeds`` are used. Every seed is folded independently and written to
    its own ``seed-{seed}_sample-{s}`` subdir, so a job with S seeds and
    ``num_samples`` samples yields S x num_samples structures. When
    ``skip_existing`` is set, any seed whose full set of sample CIFs already
    exists is skipped (per-seed resume)."""
    from deepfold3.data.module.dataset import InferenceInput
    from deepfold3.data.parse.json_input import parse_json_input
    from deepfold3.io.confidences import (
        save_confidences_json,
        save_distogram_logits,
        save_ranking_scores_csv,
        save_summary_confidences_json,
    )
    from deepfold3.io.mmcif import save_mmcif

    click.echo(f"[batch] featurizing {json_input} …")
    t0 = time.perf_counter()
    jobs = parse_json_input(json_input, template_cache_dir=template_cache, pdb_mmcif_dir=pdb_mmcif_dir)
    if not jobs:
        click.echo(f"[batch] skipping {json_input}: no jobs in JSON")
        return
    job = jobs[0]
    job_name = job.name
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

    guidance_terms = None
    if guidance_specs:
        from deepfold3.io.mmcif import resolve_chain_name as _resolve
        from deepfold3.model.guidance import build_guidance_terms

        chain_index = {_resolve(a, chain_names): a for a in feats["asym_id"][:original_n].unique().tolist()}
        try:
            guidance_terms = build_guidance_terms(
                guidance_specs,
                feats["asym_id"],
                chain_index,
                original_n,
                residue_index=feats.get("residue_index"),
                allow_overlap=allow_overlap,
            )
        except ValueError as e:
            raise click.ClickException(f"--guidance on {json_input.name}: {e}") from e

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

    output_dir.mkdir(parents=True, exist_ok=True)
    # One manifest per job: a sweep's outputs then say for themselves which
    # weights, code and flags produced them.
    manifest_path = output_dir / "run_manifest.json"
    manifest = new_manifest(
        json_input=str(json_input),
        seeds=[int(x) for x in seed_list],
        num_tokens=original_n,
        padded_tokens=feats["res_type"].shape[0],
        msa_rows=S,
        guidance=[{**dataclasses.asdict(g), "description": _describe_guidance(g)} for g in guidance_specs],
        restraints=[
            {**dataclasses.asdict(r), "description": describe_restraint(r), "weight": restraint_weight}
            for r in restraint_specs
        ],
        precision=precision,
        **(manifest_fields or {}),
    )
    write_manifest(manifest_path, manifest)

    summaries: list[dict] = []

    for current_seed in seed_list:
        # Judge a seed on every file it owes, not just model.cif: a run killed
        # between the structure and its confidences would otherwise be "done"
        # with the rest missing for good.
        if skip_existing and all(
            p.exists() for p in _seed_outputs(output_dir, current_seed, num_samples, save_distogram)
        ):
            click.echo(f"[batch]   seed {current_seed}: complete — skip")
            summaries += _existing_summaries(output_dir, current_seed, num_samples)
            continue
        click.echo(f"[batch] {json_input.name}  seed={current_seed} samples={num_samples} recycles={num_recycles}")
        t0 = time.perf_counter()
        with (
            torch.inference_mode(),
            torch.autocast(feats["res_type"].device.type, dtype=torch.bfloat16, enabled=precision == "bf16"),
        ):
            model_output = model(
                feats,
                seed=current_seed,
                num_recycles=num_recycles,
                num_samples=num_samples,
                guidance_terms=guidance_terms,
                restraints=restraints,
                restraint_weight=restraint_weight,
                restraint_rigid=restraint_rigid,
                restraint_max_step=restraint_max_step,
                restraint_after_step=restraint_after_step,
                restraint_after_sigma=restraint_after_sigma,
            )

        iter_feats = feats
        if use_bucket and original_n < feats["res_type"].shape[0]:
            from deepfold3.model.bucket import unpad_features, unpad_output

            model_output = unpad_output(model_output, original_n)
            iter_feats = unpad_features(feats, original_n)
        click.echo(f"  done in {time.perf_counter() - t0:.1f}s")

        # Trunk distogram is shared across this seed's samples (no sample axis),
        # so write it once per seed rather than into every sample dir.
        if save_distogram:
            distogram_npz = output_dir / f"seed-{int(current_seed)}_distogram.npz"
            if save_distogram_logits(distogram_npz, model_output, scheme=logits_scheme, seed=int(current_seed)):
                click.echo(f"  saved  {distogram_npz.name}")

        for s in range(num_samples):
            sample_dir = output_dir / f"seed-{int(current_seed)}_sample-{int(s)}"
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
                seed=int(current_seed),
                chain_names=chain_names,
            )
            summary = {**summary, "seed": int(current_seed), "sample": int(s)}
            summaries.append(summary)
            if plot_confidences:
                from deepfold3.scripts.plot_confidences import plot_confidences_file

                plot_confidences_file(conf_path, conf_path.with_suffix(".png"))

    if summaries:
        save_ranking_scores_csv(output_dir / "ranking_scores.csv", summaries)
        summary_name = f"{job_name}_summary_confidences.json" if job_name else "summary_confidences.json"
        save_summary_confidences_json(output_dir / summary_name, summaries, job_name=job_name)
    manifest["status"] = "ok"
    write_manifest(manifest_path, manifest)


def _seed_outputs(output_dir: Path, seed: int, num_samples: int, save_distogram: bool) -> list[Path]:
    """Every file one seed of a job must leave behind, in write order."""
    out: list[Path] = []
    if save_distogram:
        out.append(output_dir / f"seed-{int(seed)}_distogram.npz")
    for s in range(num_samples):
        d = output_dir / f"seed-{int(seed)}_sample-{int(s)}"
        out += [d / "model.cif", d / "confidences.json"]
    return out


def _existing_summaries(output_dir: Path, seed: int, num_samples: int) -> list[dict]:
    """The per-sample summaries of a seed already on disk.

    ``ranking_scores.csv`` and ``summary_confidences.json`` cover the whole job,
    so a resumed run has to re-read the seeds it skips or it would publish only
    the seeds of this run.

    Raises:
        click.ClickException: A ``confidences.json`` is unreadable or has no
            summary, which means the skip decision was wrong.
    """
    out = []
    for s in range(num_samples):
        path = output_dir / f"seed-{int(seed)}_sample-{int(s)}" / "confidences.json"
        try:
            summary = json.loads(path.read_text())["summary"]
        except (OSError, ValueError, KeyError) as e:
            raise click.ClickException(f"{path} is present but unusable ({e}); delete the seed and re-run") from e
        out.append({**summary, "seed": int(seed), "sample": int(s)})
    return out


def _parse_inputs_file(path: Path, output_template: str, params: Path) -> list[tuple[Path, Path]]:
    """Read a TSV of ``json_path [TAB output_dir]`` lines. Returns
    ``[(json_path, output_dir), ...]``."""
    out = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        json_path = Path(parts[0])
        if len(parts) == 2:
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
        out.append((json_path, output_dir))
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
    "--optim",
    default="inference",
    type=click.Choice(["none", "recommended", "inference", "all", "triton"]),
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
        "is pure denoising: it repairs what the push broke. The default is σ_data."
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
    snapshot: str,
    seeds: tuple[int, ...],
    device: str | None,
    template_cache: Path | None,
    pdb_mmcif_dir: Path | None,
    optim: str,
    matmul_precision: str,
    precision: str,
    hoist_sampler: bool,
    guidance_specs: tuple,
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

    jobs: list[tuple[Path, Path]] = []
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
        jobs.append((jp, out_dir))

    click.echo(f"[batch] {len(jobs)} jobs queued")

    if det:
        apply_det_recipe()
        if optim in ("recommended", "all"):
            raise click.UsageError(f"--det cannot fix torch.compile's kernels; --optim {optim} enables it")

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

    def _job_complete(od: Path) -> bool:
        """A job is complete when every requested (seed, sample) CIF exists.

        With CLI seeds we know the exact set; without them we can't cheaply
        know the JSON's own seeds, so fall back to the any-seed-dir heuristic.
        """
        if not od.exists():
            return False
        if seed_list:
            return all(p.exists() for s in seed_list for p in _seed_outputs(od, s, num_samples, save_distogram))
        return any(p.name.startswith("seed-") for p in od.iterdir() if p.is_dir())

    n_done = 0
    n_skipped = 0
    t_total = time.perf_counter()
    for idx, (jp, od) in enumerate(jobs, 1):
        if skip_existing and _job_complete(od):
            click.echo(f"[batch] [{idx}/{len(jobs)}] SKIP {jp.name} → {od} (all seeds/samples present)")
            n_skipped += 1
            continue
        if not jp.exists():
            click.echo(f"[batch] [{idx}/{len(jobs)}] MISS {jp} — not found")
            continue
        click.echo(f"[batch] [{idx}/{len(jobs)}] {jp.name} → {od}")
        _run_one_job(
            model,
            model_cfg,
            dev,
            jp,
            od,
            num_recycles=num_recycles,
            num_samples=num_samples,
            seeds=seed_list,
            template_cache=template_cache,
            pdb_mmcif_dir=pdb_mmcif_dir,
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
        )
        n_done += 1
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
    click.echo(f"[batch] finished: {n_done} done, {n_skipped} skipped  ({elapsed:.1f}s total)")


if __name__ == "__main__":
    inference_batch()
