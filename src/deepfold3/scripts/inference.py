"""Inference script: run DeepFold3 from an A3M MSA file or JSON input.

Usage
-----
uv run python -m deepfold3.scripts.inference \\
    --msa      path/to/query.a3m \\
    --output   output/prediction.cif \\
    --params   resources/deepfold3.safetensors \\
    --num-samples  5 \\
    --seed     42

# Or from an AF3/Protenix-format JSON input:
uv run python -m deepfold3.scripts.inference \\
    --json     path/to/input.json \\
    --output   output_dir/ \\
    --params   resources/deepfold3.safetensors \\
    --template-cache /tmp/template_cache

Supported parameter formats:
  - ``.safetensors`` — native format produced by ``convert_weights.py``
  - ``.bin`` / ``.bin.zst`` — raw AF3 params (on-the-fly conversion;
    Fourier weights will be missing — run ``convert_weights.py`` first)

For ``--json`` input, the output directory is populated with one
sub-directory per (seed, sample) pair named ``seed-{seed}_sample-{idx}/``
containing ``model.cif``, ``confidences.json`` (scalars + per-atom pLDDT),
and ``logits.npz`` (compressed raw PAE/PDE logits), plus a per-seed
``seed-{seed}_distogram.npz`` (compressed raw distogram logits, shared across
that seed's samples) and top-level ``ranking_scores.csv`` /
``{job_name}_summary_confidences.json``. The dense PAE/PDE/distogram/contact
matrices are no longer inlined into JSON — recover them with
``deepfold3.io.confidences.load_logit_maps`` (or ``logit_codec.load_logits``).

Classifier-free guidance needs a checkpoint trained with the conditioning drop
(``configs/stage2.yaml``). ``--guidance`` is repeatable and the terms add up
(syntax in ``deepfold3.model.guidance``):

    --guidance 2.0                      # full drop, D_u + 2·(D_c − D_u)
    --guidance 2.0@H:A,L:A              # only these interfaces
    --guidance 1.5@all@1..160 \\
    --guidance 2.0@H:A                  # D_c + 0.5·(D_c − D_u1) + 1·(D_c − D_u2)

For ``--msa`` / ``--sequence`` input, a single mmCIF file with all
samples stacked as MODEL blocks is written to ``--output``, with
``{stem}[_sample{s}]_logits.npz`` per sample and ``{stem}_distogram.npz``
per seed alongside it. Per-atom pLDDT is stored in the B-factor column.

Every run also writes ``run_manifest.json`` (``{stem}_run_manifest.json`` for
``--msa`` / ``--sequence``): code, versions, GPU, checkpoint digest and the
settings that change a result. See ``deepfold3.io.run_manifest``.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import click
import torch

from deepfold3.model.guidance import (
    GuidanceSpec,
    build_guidance_terms,
    check_guidance_training,
    has_residue_ranges,
    parse_guidance_spec,
)
from deepfold3.model.restraints import RestraintSpec, build_restraints, describe_restraint, parse_restraint_spec
from deepfold3.scripts import cli
from deepfold3.scripts._common import apply_det_recipe, check_loaded_keys, load_pt_snapshot, pin_cuda_device


def _parse_guidance(_ctx, _param, values: tuple[str, ...]) -> tuple[GuidanceSpec, ...]:
    try:
        return tuple(parse_guidance_spec(v) for v in values)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e


def _parse_restraints(_ctx, _param, values: tuple[str, ...]) -> tuple[RestraintSpec, ...]:
    try:
        return tuple(parse_restraint_spec(v) for v in values)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e


def _parse_restraint_pairs(_ctx, _param, values: tuple[str, ...]) -> tuple[RestraintSpec, ...]:
    try:
        return tuple(parse_restraint_spec(v, paired=True) for v in values)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e


def _parse_repels(_ctx, _param, values: tuple[str, ...]) -> tuple[RestraintSpec, ...]:
    try:
        return tuple(parse_restraint_spec(v, repel=True) for v in values)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e


def _describe_guidance(spec: GuidanceSpec) -> str:
    target = ",".join(f"{a}:{b}" for a, b in spec.pairs) or spec.target
    sigma = f" σ∈[{spec.sigma_range[0]:g}, {spec.sigma_range[1]:g}]" if spec.sigma_range else ""
    return f"w={spec.scale:g} {target}{sigma}"


def _save_logits(save, path: Path, *args, **kwargs) -> None:
    """Run a logit saver on the writer thread and log the file it wrote."""
    if save(path, *args, **kwargs):
        click.echo(f"  saved  {path.resolve()}")


@cli.command("inference")
@click.option(
    "--msa",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Input A3M file (first sequence is query)",
)
@click.option(
    "--json",
    "json_input",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Protenix-format JSON input file (alternative to --msa)",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output path: directory for --json (filenames from job name), or .cif file for --msa",
)
@click.option(
    "--params",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Model parameters (.safetensors converted weights, .pt training checkpoint, or .bin/.bin.zst raw AF3 params)",
)
@click.option(
    "--mol-type",
    default="protein",
    type=click.Choice(["protein", "rna", "dna"]),
    show_default=True,
    help="Molecule type",
)
@click.option(
    "--msa-crop-size",
    type=int,
    default=None,
    help=(
        "Per-recycle MSA rows for the Evoformer. Default: follow the MSA pool cap (``const.MAX_MSA_ROWS`` = 16384) rather than the checkpoint's value — training deliberately uses a small crop for throughput and randomises the effective row count (AF3 s2.3 k ~ Uniform[1, crop]) so the model is row-count invariant, and serving is meant to use the whole pool. Pass a smaller number to reproduce a training-crop run or to bound memory."
    ),
)
@click.option("--num-recycles", default=10, show_default=True, help="Evoformer recycling iterations")
@click.option("--num-samples", default=5, show_default=True, help="Number of diffusion samples per seed")
@click.option(
    "--snapshot",
    type=click.Choice(["auto", "shadow", "average", "fast"]),
    default="auto",
    show_default=True,
    help=(
        "Which weight snapshot to load from a .pt training checkpoint. "
        "'auto' = shadow when present (overlaid on ckpt['model']), else ckpt['model']. "
        "'shadow' forces the external EMA shadow (errors if absent). "
        "'average' forces ckpt['model'] (= SF's averaged iterate x post eval-mode flip; "
        "= live params for AdamW). 'fast' forces the SF z buffer reconstructed from "
        "ckpt['optimizer']['state'] (SF only; ablation use). For .safetensors / raw "
        "AF3 params this flag is ignored."
    ),
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help=(
        "Random seed for diffusion sampling. For --json input, this is "
        "prepended to the JSON's modelSeeds list (all seeds are run). "
        "Defaults to 42 for --msa/--sequence input, and to the JSON's "
        "modelSeeds otherwise."
    ),
)
@click.option("--device", default=None, help="Torch device (e.g. 'cuda', 'cpu'). Auto-detected if omitted.")
@click.option(
    "--templates",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Directory of template mmCIF files (optional, used with --msa)",
)
@click.option(
    "--template-cache",
    default=None,
    type=click.Path(path_type=Path),
    help="Directory to cache downloaded template mmCIF files (used with --json)",
)
@click.option(
    "--pdb-mmcif-dir",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Local PDB mmCIF mirror directory (layout: {dir}/{mid2}/{id}.cif.gz)",
)
@click.option(
    "--optim",
    default="inference",
    type=click.Choice(["none", "recommended", "inference", "all", "triton"]),
    show_default=True,
    help="Optimization preset: inference (default, Triton+tiling), recommended (training), triton (no tiling), none, all (testing)",
)
@click.option(
    "--matmul-precision",
    default="auto",
    type=click.Choice(["auto", "highest", "high", "medium"]),
    show_default=True,
    help=(
        "float32 matmul precision (torch.set_float32_matmul_precision). 'auto' "
        "keeps the historical behaviour: TF32 ('high') whenever --optim is not "
        "'none', untouched otherwise. 'highest' forces true fp32 matmuls — "
        "slower, but the reference for whether TF32 costs coordinate accuracy."
    ),
)
@click.option(
    "--plot-confidences",
    is_flag=True,
    default=False,
    help="Also write a PAE/PDE/contact-probability PNG next to each confidences JSON",
)
@click.option(
    "--logits-scheme",
    default="int8_softmax",
    type=click.Choice(["float32", "float16", "int8", "int8_softmax", "int4_softmax"]),
    show_default=True,
    help=(
        "Compression scheme for the raw head logits stored alongside each "
        "prediction: per-sample logits.npz (PAE, PDE) + per-seed distogram.npz. "
        "int8_softmax preserves softmax(logits) within ~1e-4 at ~1 byte/value."
    ),
)
@click.option(
    "--precision",
    default="fp32",
    type=click.Choice(["fp32", "bf16"]),
    show_default=True,
    help=(
        "Compute dtype of the trunk, as training and eval run it: the MSA stack, "
        "Pairformer and confidence blocks. The diffusion sampler, the atom conditioning "
        "encoder and the final confidence layers stay fp32 either way, as the checkpoint's "
        "own precision config asks. fp32 is the historical inference behaviour."
    ),
)
@click.option(
    "--bucket/--no-bucket",
    default=True,
    show_default=True,
    help=(
        "Pad tokens to the next bucket size (masked, removed from the outputs), so "
        "shapes stay few and warmed kernels hit. The sampler's noise then differs "
        "from an unpadded run of the same seed."
    ),
)
@click.option(
    "--guidance",
    "guidance_specs",
    multiple=True,
    metavar="W[@TARGET[@LO..HI]]",
    callback=_parse_guidance,
    help=(
        "Classifier-free guidance term; repeat for a linear combination "
        "D_c + Σ (W−1)·(D_c − D_u). TARGET is what the term's unconditional branch "
        "drops: full (default), all (every interface) or chain pairs A:B,AA:B. "
        "Either side of a pair may name residues of that chain instead of all of "
        "it, in the input's own numbering, with + between ranges: A/25-40+88:B. "
        "A residue subset is a finer drop than training ever did (whole chain-pair "
        "blocks), so W far from 1 there is extrapolation. "
        "LO..HI limits the term to that σ̂ range (absolute, 0.0077..4608 is the "
        "whole schedule). Each active term adds one denoiser pass per step."
    ),
)
@click.option(
    "--allow-guidance-overlap",
    is_flag=True,
    default=False,
    help=(
        "Accept --guidance selections that drop the same pair block twice: a union inside one "
        "term, and summed weights across terms (otherwise an error and a warning)."
    ),
)
@click.option(
    "--restraint",
    "restraint_specs",
    multiple=True,
    metavar="SEL:SEL[@DIST[@LO..HI]]",
    callback=_parse_restraints,
    help=(
        "Distance restraint steered into the sampler: every residue of the left selector "
        "must come within DIST Å (default 6, range 2–30) of the right one. Selectors are "
        "``--guidance``'s (A/25-40+88:B). LO..HI limits it to that σ̂ range. Unlike "
        "--guidance this acts on coordinates, not on the conditioning, and nothing was "
        "trained for it; repeat for several restraints."
    ),
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
    "--hoist-sampler/--no-hoist-sampler",
    default=True,
    show_default=True,
    help=(
        "Drop empty atom windows and compute step-invariant conditioning once in the "
        "diffusion sampler. Same noise draws; coordinates can move within the TF32 "
        "tolerance, ~0.1 Å per denoiser call. --no-hoist-sampler runs the plain sampler."
    ),
)
@click.option(
    "--allow-missing-keys",
    is_flag=True,
    default=False,
    help=(
        "Run even if the weights leave model parameters at random initialisation. Without it a "
        "missing key is an error. Needed for raw AF3 params, which lack the Fourier weights."
    ),
)
@click.option(
    "--det",
    is_flag=True,
    default=False,
    help=(
        "Deterministic inference: the same input, weights, seed and flags give the same bytes on the "
        "same GPU model, driver, software stack and CPU thread count. Fixes cuBLAS's workspace, uses "
        "torch's deterministic algorithms and serves each Triton kernel one fixed config instead of "
        "the tuned one, so runs may be slower and differ from a run without it in the low bits. Not "
        "reproducible across GPU models or torch/Triton versions. Refuses torch.compile presets."
    ),
)
def inference(
    msa: Path | None,
    json_input: Path | None,
    output: Path,
    params: Path,
    mol_type: str,
    msa_crop_size: int | None,
    num_recycles: int,
    num_samples: int,
    snapshot: str,
    seed: int | None,
    device: str | None,
    templates: Path | None,
    template_cache: Path | None,
    pdb_mmcif_dir: Path | None,
    optim: str | None,
    matmul_precision: str,
    plot_confidences: bool,
    logits_scheme: str,
    precision: str,
    bucket: bool,
    guidance_specs: tuple[GuidanceSpec, ...],
    allow_guidance_overlap: bool,
    restraint_specs: tuple[RestraintSpec, ...],
    restraint_pair_specs: tuple[RestraintSpec, ...],
    repel_specs: tuple[RestraintSpec, ...],
    restraint_weight: float,
    restraint_rigid: bool,
    restraint_max_step: float,
    restraint_after_step: bool,
    restraint_after_sigma: float,
    hoist_sampler: bool,
    allow_missing_keys: bool,
    det: bool,
) -> None:
    """Run structure prediction from an A3M MSA file or JSON input."""
    # Validate input: exactly one of --msa or --json
    if msa is None and json_input is None:
        raise click.UsageError("Provide either --msa or --json")
    if msa is not None and json_input is not None:
        raise click.UsageError("Provide only one of --msa or --json, not both")
    if det:
        apply_det_recipe()  # before pin_cuda_device: cuBLAS reads its workspace at first use

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    # pin_cuda_device also sets the current CUDA device so the fused Triton
    # kernels launch on the same device as the model/features — a non-default
    # --device (e.g. cuda:1) otherwise trips "cannot be accessed from Triton".
    dev = pin_cuda_device(device)
    click.echo(f"[inference] device: {dev}")

    # Each input is a new shape: reuse the closest warmed Triton config rather
    # than stall on an autotune benchmark (``warmup-triton --inference-only``).
    from deepfold3.model.triton.cache import set_nearest_shape_lookup

    set_nearest_shape_lookup(True)  # ignored under --det, which serves fixed configs

    # ------------------------------------------------------------------
    # Load model weights
    # ------------------------------------------------------------------
    click.echo(f"[inference] loading parameters from {params} …")
    t0 = time.perf_counter()
    from deepfold3.model.model import DeepFold3

    # Architectural ModelConfig embedded in training checkpoints — preserved
    # so the inference build matches whatever flags (e.g.
    # ``pairformer.residual_output_scaling``) were active at training time.
    # Stays ``None`` only for raw-AF3 ``.bin`` / ``.bin.zst`` paths and for
    # legacy ``.safetensors`` files exported before metadata embedding was
    # introduced — both fall back to ``ModelConfig()`` defaults.
    ckpt_model_cfg = None

    if params.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        from deepfold3.model.config import model_cfg_from_metadata

        # safetensors metadata read is cheap (header-only); pull ModelConfig
        # JSON before loading tensors so we can build the model with the
        # correct architectural flags.
        with safe_open(str(params), framework="pt") as f:
            ckpt_model_cfg = model_cfg_from_metadata(f.metadata())
        state_dict = load_file(str(params))
    elif params.suffix == ".pt":
        # Also captures the embedded model_cfg — critical for runs that override
        # architectural flags (residual scaling, layer counts, etc.) whose
        # dataclass defaults differ from the values used during training.
        state_dict, ckpt_model_cfg = load_pt_snapshot(params, snapshot)
    else:
        import warnings

        from deepfold3.convert.af3 import haiku_to_torch
        from deepfold3.params import read_af3_params

        warnings.warn(
            "Loading raw AF3 params — Fourier weights will be missing. "
            "Run convert_weights.py to produce a .safetensors file.",
            stacklevel=1,
        )
        raw_params = read_af3_params(params)
        state_dict = haiku_to_torch(raw_params)

    # Build model config with optimization settings
    from deepfold3.model.config import ModelConfig, OptimConfig

    if ckpt_model_cfg is not None:
        model_cfg = ckpt_model_cfg
        click.echo("[inference] using model_cfg embedded in checkpoint")
    else:
        model_cfg = ModelConfig()
        if params.suffix == ".pt":
            click.echo(
                "[inference] checkpoint has no embedded model_cfg — falling "
                "back to ModelConfig() defaults. If the checkpoint was trained "
                "with non-default architectural flags (e.g. residual_output_scaling), "
                "the inference build will silently disagree."
            )
    # ``--optim`` always overrides the OptimConfig sub-config so inference
    # picks fused / SDPA / tiled kernels regardless of how the checkpoint
    # was trained (training and inference often differ here).
    if optim == "all":
        model_cfg.optim = OptimConfig.all_enabled()
    elif optim == "recommended":
        model_cfg.optim = OptimConfig.recommended()
    elif optim == "inference":
        model_cfg.optim = OptimConfig.inference()
    elif optim == "triton":
        model_cfg.optim = OptimConfig.triton_only()
    # else: keep checkpoint's optim (or default OptimConfig if no checkpoint cfg)
    model_cfg.diffusion.hoist_sampler = hoist_sampler

    if det and model_cfg.optim.use_torch_compile:
        raise click.UsageError("--det does not cover torch.compile; use --optim inference")

    if guidance_specs:
        if ckpt_model_cfg is None:
            click.echo("  [WARN] no embedded model_cfg: cannot check that --guidance matches the training drop")
        else:
            try:
                check_guidance_training(guidance_specs, ckpt_model_cfg.diffusion)
            except ValueError as e:
                raise click.UsageError(f"--guidance: {e}") from e

    # Default to the MSA pool cap, NOT the checkpoint's training crop: the crop
    # is a training-throughput choice and the random-k mask makes the model
    # row-count invariant, so serving the trained crop would throw away MSA the
    # model can use. Explicit --msa-crop-size still wins.
    from deepfold3.data.const import MAX_MSA_ROWS

    want = msa_crop_size if msa_crop_size is not None else MAX_MSA_ROWS
    if want != model_cfg.msa_crop_size:
        prev = model_cfg.msa_crop_size
        model_cfg.msa_crop_size = want
        src = "flag" if msa_crop_size is not None else "default (MSA pool cap)"
        click.echo(f"[inference] msa_crop_size {prev} -> {want} per recycle [{src}]")

    if matmul_precision != "auto":
        torch.set_float32_matmul_precision(matmul_precision)
        click.echo(f"[inference] float32 matmul precision: {matmul_precision}")
    elif optim and optim != "none":
        torch.set_float32_matmul_precision("high")  # TF32 for better perf
    if optim and optim != "none":
        click.echo(f"[inference] optimizations: {optim}")

    model = DeepFold3(model_cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Filter legacy fused-kernel key names (no-op for current checkpoints).
    real_missing = check_loaded_keys(missing, unexpected, allow_missing=allow_missing_keys)
    model.to(dev).eval()
    click.echo(f"  loaded in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Featurize
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    from deepfold3.data.module.dataset import InferenceInput

    job_name: str | None = None
    entity_descriptions: dict[str, str] | None = None
    job_title: str | None = None
    chain_names: list[str | None] | None = None
    if json_input is not None:
        click.echo(f"[inference] featurizing {json_input} …")
        from deepfold3.data.parse.json_input import parse_json_input

        jobs = parse_json_input(json_input, template_cache_dir=template_cache, pdb_mmcif_dir=pdb_mmcif_dir)
        if not jobs:
            raise click.UsageError("No jobs found in JSON input")
        job = jobs[0]
        job_name = job.name
        job_title = job.description or job.name

        # Chain names as written in the input JSON, in asym_id order. Without
        # this the output is renamed A, B, C, … so a job asking for A/B/D/E gets
        # A/B/C/D back and every name-keyed consumer silently mismatches.
        from deepfold3.data.module.dataset import job_chain_ids
        from deepfold3.io.mmcif import resolve_chain_name

        chain_names = job_chain_ids(job)

        # Build entity descriptions from chain descriptions, keyed by the same
        # names the writers will use. A multi-CCD ligand expands to several
        # chains, so walk asym_id and let job_chain_ids define the ordering.
        desc_map: dict[str, str] = {}
        chain_idx = 0
        for chain_list in [job.protein_chains, job.dna_chains, job.rna_chains, job.ligand_chains]:
            for chain in chain_list:
                n_expanded = len(chain.ccd_ids) if getattr(chain, "ccd_ids", None) else 1
                for _ in range(n_expanded):
                    if chain.description:
                        desc_map[resolve_chain_name(chain_idx, chain_names)] = chain.description
                    chain_idx += 1
        if desc_map:
            entity_descriptions = desc_map

        feats = InferenceInput.from_job(job)
    else:
        click.echo(f"[inference] featurizing {msa} …")
        a3m_text = msa.read_text()

        # Parse templates if directory provided
        template_feats = None
        if templates is not None:
            import numpy as np

            from deepfold3.data import const as data_const
            from deepfold3.data.parse.a3m import parse_a3m
            from deepfold3.data.parse.templates import parse_template_mmcifs

            mt = data_const.MOL_TYPE_IDS.get(mol_type.upper(), data_const.MOL_PROTEIN)
            msa_obj = parse_a3m(a3m_text, mt)
            n_tokens = msa_obj.sequences.shape[1]
            query_residue_index = np.arange(n_tokens, dtype=np.int64) + 1  # 1-based
            template_feats = parse_template_mmcifs(templates, query_residue_index)
            if template_feats:
                click.echo(f"  loaded {len(template_feats)} template(s) from {templates}")
            else:
                click.echo("  no valid templates found")
                template_feats = None

        feats = InferenceInput.from_a3m(a3m_text, mol_type=mol_type, templates=template_feats)

    N = feats["res_type"].shape[0]
    S = feats["msa"].shape[0]
    T = feats.get("template_aatype", torch.zeros(0)).shape[0]
    click.echo(f"  {N} tokens, {S} MSA sequences, {T} templates  ({time.perf_counter() - t0:.1f}s)")

    # Bucket padding: a small set of token counts keeps the Triton autotune cache
    # (exact-shape keys) and torch.compile from redoing their work per input.
    original_n = N
    use_bucket = bucket or model_cfg.optim.use_torch_compile
    if use_bucket:
        from deepfold3.model.bucket import next_bucket, pad_features

        bucket_n = next_bucket(N)
        if bucket_n > N:
            feats, original_n = pad_features(feats, bucket_n)
            click.echo(f"  bucket padding: {N} → {bucket_n} tokens")

    # Move to device
    feats = {k: v.to(dev) for k, v in feats.items()}

    guidance_terms = None
    if guidance_specs:
        from deepfold3.io.mmcif import resolve_chain_name

        chain_index = {resolve_chain_name(a, chain_names): a for a in feats["asym_id"][:original_n].unique().tolist()}
        try:
            guidance_terms = build_guidance_terms(
                guidance_specs,
                feats["asym_id"],
                chain_index,
                original_n,
                residue_index=feats.get("residue_index"),
                allow_overlap=allow_guidance_overlap,
            )
        except ValueError as e:
            raise click.UsageError(f"--guidance: {e}") from e
        click.echo(f"[inference] guidance: {' + '.join(_describe_guidance(g) for g in guidance_specs)}")
        if has_residue_ranges(guidance_specs):
            click.echo("[inference] note: a residue-subset drop is finer than the training drop (whole chain pairs)")

    all_restraint_specs = tuple(restraint_specs) + tuple(restraint_pair_specs) + tuple(repel_specs)
    restraints = ()
    if all_restraint_specs:
        from deepfold3.io.mmcif import resolve_chain_name

        chain_index = {resolve_chain_name(a, chain_names): a for a in feats["asym_id"][:original_n].unique().tolist()}
        try:
            restraints = build_restraints(
                all_restraint_specs,
                feats["asym_id"],
                chain_index,
                original_n,
                residue_index=feats.get("residue_index"),
            )
        except ValueError as e:
            raise click.UsageError(f"--restraint: {e}") from e
        click.echo(
            f"[inference] restraints (weight {restraint_weight:g}): "
            + " + ".join(describe_restraint(r) for r in all_restraint_specs)
        )

    # ------------------------------------------------------------------
    # Determine seed list. JSON inputs honour the JSON's modelSeeds
    # (with --seed prepended if explicitly passed); MSA/sequence inputs
    # use a single seed (default 42).
    # ------------------------------------------------------------------
    if json_input is not None:
        json_seeds = list(job.model_seeds) if job.model_seeds else [42]
        if seed is not None:
            seed_list = [seed] + [s for s in json_seeds if s != seed]
        else:
            seed_list = json_seeds
    else:
        seed_list = [seed if seed is not None else 42]

    from deepfold3.io.background_writer import BackgroundWriter, logits_to_host
    from deepfold3.io.confidences import (
        save_confidences_json,
        save_distogram_logits,
        save_ranking_scores_csv,
        save_sample_logits,
        save_summary_confidences_json,
    )
    from deepfold3.io.mmcif import save_mmcif

    if json_input is not None:
        output_dir = output
    else:
        cif_path = output
        output_dir = cif_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    from deepfold3.io.run_manifest import (
        checkpoint_record,
        gpu_name,
        new_manifest,
        sha256_file,
        write_manifest,
    )

    manifest_path = output_dir / (
        "run_manifest.json" if json_input is not None else f"{cif_path.stem}_run_manifest.json"
    )
    checkpoint = {**checkpoint_record(params), "snapshot": snapshot if params.suffix == ".pt" else None}
    manifest = new_manifest(
        device=str(dev),
        gpu=gpu_name(dev),
        checkpoint=checkpoint,
        missing_keys=real_missing,
        seeds=[int(s) for s in seed_list],
        num_samples=num_samples,
        num_recycles=num_recycles,
        msa_crop_size=model_cfg.msa_crop_size,
        bucket={"enabled": use_bucket, "num_tokens": original_n, "padded_tokens": feats["res_type"].shape[0]},
        guidance=[{**asdict(g), "description": _describe_guidance(g)} for g in guidance_specs],
        restraints=[
            {**asdict(r), "description": describe_restraint(r), "weight": restraint_weight} for r in restraint_specs
        ],
        optim=optim,
        precision=precision,
        matmul_precision=torch.get_float32_matmul_precision(),
        logits_scheme=logits_scheme,
        det=det,
    )
    write_manifest(manifest_path, manifest)

    def _record_digest() -> None:
        checkpoint["sha256"] = sha256_file(params)

    summaries: list[dict] = []  # one entry per (seed, sample), augmented with seed/sample

    # Logit compression is the bulk of the write time and does not depend on the
    # device, so it runs behind the next seed's forward. One seed's worth may
    # queue; everything is on disk, or the run has failed, past this block.
    try:
        with BackgroundWriter(max_pending=num_samples + 2) as writer:
            writer.submit("checkpoint digest", _record_digest)
            for current_seed in seed_list:
                click.echo(
                    f"[inference] running model  (seed={current_seed}, recycles={num_recycles}, samples={num_samples}) …"
                )
                t0 = time.perf_counter()
                with (
                    torch.inference_mode(),
                    torch.autocast(dev.type, dtype=torch.bfloat16, enabled=precision == "bf16"),
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

                elapsed = time.perf_counter() - t0
                click.echo(f"  done in {elapsed:.1f}s  —  positions: {tuple(model_output.atom_positions.shape)}")

                # Trunk distogram logits are shared across all samples of this seed, so
                # they are written once per seed (not duplicated per sample).
                if json_input is not None:
                    distogram_npz = output_dir / f"seed-{int(current_seed)}_distogram.npz"
                else:
                    distogram_npz = output_dir / f"{cif_path.stem}_distogram.npz"
                host_logits = logits_to_host(model_output)
                writer.submit(
                    distogram_npz.name,
                    _save_logits,
                    save_distogram_logits,
                    distogram_npz,
                    host_logits,
                    scheme=logits_scheme,
                    seed=int(current_seed),
                )

                # Non-JSON path writes one multi-MODEL mmCIF; JSON path writes
                # per-sample directories.  Either way, save_confidences_json
                # returns the per-sample summary used for ranking.
                if json_input is None:
                    click.echo(f"[inference] writing {cif_path} …")
                    save_mmcif(
                        cif_path,
                        model_output,
                        iter_feats,
                        entity_descriptions=entity_descriptions,
                        title=job_title,
                        chain_names=chain_names,
                    )
                    click.echo(f"  saved  {cif_path.resolve()}")

                for s in range(num_samples):
                    if json_input is not None:
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
                        logits_npz = sample_dir / "logits.npz"
                    else:
                        stem = cif_path.stem
                        suffix = "" if num_samples == 1 else f"_sample{s}"
                        conf_path = output_dir / f"{stem}{suffix}_confidences.json"
                        logits_npz = output_dir / f"{stem}{suffix}_logits.npz"

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

                    # Raw per-sample PAE/PDE logits (compressed) — replaces the dense
                    # float matrices that used to live in confidences.json.
                    writer.submit(
                        str(logits_npz),
                        _save_logits,
                        save_sample_logits,
                        logits_npz,
                        host_logits,
                        sample_idx=s,
                        scheme=logits_scheme,
                        seed=int(current_seed),
                    )

                    iptm_str = f", ipTM={summary['iptm']:.3f}" if summary.get("iptm") is not None else ""
                    clash_str = ", CLASH" if summary["has_clash"] else ""
                    click.echo(
                        f"  seed {current_seed} sample {s}: "
                        f"pLDDT={summary['mean_plddt']:.1f}, pTM={summary['ptm']:.3f}{iptm_str}, "
                        f"gPDE={summary['average_pde']:.2f}, disorder={summary['fraction_disordered']:.3f}"
                        f"{clash_str}, ranking={summary['ranking_score']:.3f}"
                    )
                    click.echo(f"  saved  {conf_path.resolve()}")

                    if plot_confidences:
                        from deepfold3.scripts.plot_confidences import plot_confidences_file

                        writer.drain()  # the plot reads the logit files just queued
                        plot_path = conf_path.with_suffix(".png")
                        if plot_confidences_file(
                            conf_path, plot_path, logits_npz=logits_npz, distogram_npz=distogram_npz
                        ):
                            click.echo(f"  saved  {plot_path.resolve()}")

        # ------------------------------------------------------------------
        # Aggregate ranking outputs (JSON input only)
        # ------------------------------------------------------------------
        if json_input is not None and summaries:
            ranking_csv = output_dir / "ranking_scores.csv"
            save_ranking_scores_csv(ranking_csv, summaries)
            click.echo(f"  saved  {ranking_csv.resolve()}")

            summary_name = f"{job_name}_summary_confidences.json" if job_name else "summary_confidences.json"
            summary_path = output_dir / summary_name
            save_summary_confidences_json(summary_path, summaries, job_name=job_name)
            click.echo(f"  saved  {summary_path.resolve()}")

            if len(summaries) > 1:
                ranked = sorted(summaries, key=lambda r: r["ranking_score"], reverse=True)
                click.echo("[inference] ranking:")
                for rank, row in enumerate(ranked, 1):
                    click.echo(
                        f"  #{rank}  seed-{row['seed']}_sample-{row['sample']}  ranking_score={row['ranking_score']:.4f}"
                    )
    except BaseException as e:
        write_manifest(manifest_path, {**manifest, "status": "failed", "error": repr(e)})
        raise
    write_manifest(manifest_path, {**manifest, "status": "complete"})
    click.echo(f"  saved  {manifest_path.resolve()}")


if __name__ == "__main__":
    inference()
