"""Dataset and inference input utilities.

TrainingDataset: loads Records, crops, and featurizes for training.
InferenceInput:  builds features directly from a sequence string (no mmCIF needed).
"""

from __future__ import annotations

import json
import logging
import random
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.utils.data

from deepfold3.data import const, pack
from deepfold3.data.crop.cropper import Cropper, compute_chain_msa_cols, remap_msa_col_idx_to_slice
from deepfold3.data.feature.featurizer import Featurizer
from deepfold3.data.feature.ref_atoms import get_ref_atoms
from deepfold3.data.io_cache import read_npz
from deepfold3.data.masking import TokenMask
from deepfold3.data.msa.paths import paired_msa_path, unpaired_msa_path
from deepfold3.data.parse.pdb_chain_cache import PdbChainCache, template_features_from_chain
from deepfold3.data.paths import mask_path, monomer_template_path, structure_path, template_path
from deepfold3.data.tokenize.atomize import maybe_atomize
from deepfold3.data.tokenize.tokenizer import Tokenizer
from deepfold3.data.types import (
    ATOM_DTYPE,
    BOND_DTYPE,
    CHAIN_DTYPE,
    MSA,
    RESIDUE_DTYPE,
    Input,
    Record,
    Structure,
    Tokenized,
)

if TYPE_CHECKING:
    from deepfold3.data.ccd import CcdMolecule
    from deepfold3.data.parse.templates import TemplateFeatures

logger = logging.getLogger(__name__)

# TEMPORARILY DISABLED (2026-07-08): loading disk-stored per-token masks
# (masks/{id}.npz: "fixed" / "region"). They were built under the previous
# tokenization; after nucleotides became single tokens (res_type != 20) the
# per-token count changed, so a stored mask's length no longer matches the
# current tokens — it misaligns and crashes the loss (fixed_mask.expand_as),
# and even where lengths coincide the per-token content maps to the OLD
# token↔residue layout (cropping cannot fix this). Skip loading them until they
# are rebuilt with `deepfold3 build-masks`. The synthesized loss-weight channel
# (derived from the current tokens) is unaffected and still applied.
_DISK_TOKEN_MASKS_ENABLED = False

# Essential feature keys that every sample must contain for the model forward
# pass and loss computation to produce a valid gradient.
#
# This is also the enforcement point for the batch-uniform key contract. A key
# that only *some* samples of a batch carry is not merely absent downstream:
# ``collate_fn`` zero-fills it (wrong for a mask — reads as "everything masked
# out") and drops 0-dim keys batch-wide (taking the value away from the samples
# that did have it). Neither failure raises. Listing a key here turns "one
# dataset class forgot to emit it" into a loud per-sample warning instead, and
# it covers every class that loads through ``_retry_getitem`` — including
# ``OF3Dataset``. Add new features here rather than relying on convention.
_REQUIRED_KEYS = {
    "res_type",
    "seq_mask",
    "atom_gt_positions",
    "atom_gt_mask",
    "msa",
    "msa_mask",
    # Batch-uniform contract (see above); all four are emitted unconditionally.
    # ``is_distillation`` and ``resolution`` are 0-dim, the case collate_fn
    # drops batch-wide — losing the distillation flag would apply the
    # confidence loss to predicted structures.
    "resolution",
    "is_distillation",
    "bond_matrix",
    "template_mask",
}

# Aggregate counters for data that is dropped without failing the run: a
# resampled record, a missing MSA, a template that would not load. Each event
# is individually survivable, which is exactly why a systematic outage (one
# dataset with the wrong msa_id convention, a misconfigured cache dir) can run
# for hours with a healthy-looking loss curve. Counting them per worker and
# reporting on a widening schedule surfaces the outage without flooding the log.
_drop_counts: dict[str, int] = {}


def _count_drop(kind: str, detail: str, *, exc_info: bool = False) -> None:
    """Record a silently-dropped item and warn on a widening schedule."""
    n = _drop_counts[kind] = _drop_counts.get(kind, 0) + 1
    if n in (1, 10, 100) or n % 1000 == 0:
        logger.warning("%s [worker total: %d] — %s", kind, n, detail, exc_info=exc_info)


def _validate_features(feats: dict[str, torch.Tensor]) -> str | None:
    """Check that featurized sample is usable for training.

    Returns ``None`` if valid, or a short reason string if degenerate.
    """
    missing = _REQUIRED_KEYS - feats.keys()
    if missing:
        return f"missing features {missing}"
    if feats["res_type"].shape[0] == 0:
        return "zero tokens after crop"
    if not feats["seq_mask"].any():
        return "seq_mask is all-zero"
    return None


# Exceptions that genuinely indicate bad data on disk: unreadable file, short
# read, corrupt archive, malformed array contents.
_DATA_ERRORS = (OSError, EOFError, ValueError, zipfile.BadZipFile)

# Exceptions that are far more likely to be OUR bug than bad data — a KeyError
# from a tokenizer/CCD lookup, a RuntimeError from the cropper or featurizer, an
# IndexError from a mis-derived index array. They are still caught, because one
# malformed record must not kill a multi-day run, but they are reported at ERROR
# *with a traceback*: logged as bare ``str(e)`` they read as
# ``Sample 7abc (attempt 1): 'ARG'`` and are indistinguishable from data noise,
# which is how a bug that rejects an entire subset of the dataset stays invisible.
#
# IndexError earns its place the hard way: an out-of-range MSA column escaped
# this handler, killed the worker, and took a 112-rank job with it — the exact
# desync this function's docstring promises not to cause.
_SUSPECT_ERRORS = (RuntimeError, KeyError, IndexError)


def _retry_getitem(
    dataset: torch.utils.data.Dataset,
    idx: int,
    load_fn,
    id_fn,
    max_attempts: int = 3,
) -> dict[str, torch.Tensor]:
    """Load a sample with retry on data errors or validation failure.

    On failure, retries with a random different index.  Returns the last
    successfully loaded features (even if validation failed) so the
    DataLoader always gets a tensor dict — never an unhandled exception
    that would desync DDP ranks.

    Note the replacement index is drawn UNIFORMLY, which discards the
    cluster-balanced weighting the sampler just applied. That is a deliberate
    simplification (the dataset has no handle on the sampler), but it means a
    high retry rate skews the training distribution toward uniform-over-records
    — another reason the retry counter below matters.
    """
    orig_id = id_fn(idx)
    last_feats: dict[str, torch.Tensor] | None = None
    for attempt in range(1, max_attempts + 1):
        sid = id_fn(idx)
        try:
            feats = load_fn(idx)
            reason = _validate_features(feats)
            if reason is None:
                return feats
            logger.warning("Sample %s (attempt %d): %s", sid, attempt, reason)
            _count_drop("sample failed validation, resampling", f"{sid}: {reason}")
            last_feats = feats
        except _SUSPECT_ERRORS as e:
            # Almost certainly a bug in our own pipeline — carry the traceback.
            logger.error("Sample %s (attempt %d): %s", sid, attempt, e, exc_info=True)
            _count_drop("sample raised a suspect error, resampling", f"{sid}: {type(e).__name__}: {e}")
        except _DATA_ERRORS as e:
            logger.warning("Sample %s (attempt %d): %s", sid, attempt, e)
            _count_drop("sample raised a data error, resampling", f"{sid}: {type(e).__name__}: {e}")
        idx = random.randint(0, len(dataset) - 1)
    logger.error("Sample %s: all %d retries exhausted, returning degenerate feats", orig_id, max_attempts)
    if last_feats is not None:
        return last_feats
    # Last resort. Guarded: an escaping exception kills this worker, and the
    # peers then block in the next collective until the NCCL watchdog fires —
    # the exact desync the docstring above promises not to cause. Re-raise only
    # if we have nothing at all to hand back.
    try:
        return load_fn(idx)
    except (*_SUSPECT_ERRORS, *_DATA_ERRORS) as e:
        logger.error("Sample %s: final fallback load also failed: %s", orig_id, e, exc_info=True)
        raise


def _per_sample_rng(idx: int, salt: int) -> random.Random:
    """Deterministic per-(worker-epoch, idx) RNG, reproducible under a fixed global seed.

    Mixes the worker's per-epoch ``torch.initial_seed()`` with ``idx`` and a
    stream-specific ``salt`` — distinct salts decorrelate independent draws for
    the same sample (e.g. anchor choice vs. the large-assembly pre-crop). A bare
    ``random.Random()`` would draw OS entropy and never reproduce.
    """
    return random.Random((torch.initial_seed() ^ (idx * salt)) & 0xFFFFFFFF)


def _build_record_anchors(records: list, anchor_sampling: bool) -> dict | None:
    """A dataset's per-record AF3 §2.5.1 anchor conditional, or ``None`` if disabled.

    Weighted datasets (``anchor_sampling=True``) get a cluster-weighted
    chain/interface table so ``_draw_crop_anchor`` can seed the crop at load
    time; uniform datasets pass ``False`` and let the cropper self-seed.
    """
    if not anchor_sampling:
        return None
    from deepfold3.data.sample.weighted import build_record_anchor_tables

    return build_record_anchor_tables(records)


def _draw_crop_anchor(record_anchors: dict | None, idx: int) -> tuple[str, int, int] | None:
    """Second stage of two-stage AF3 §2.5.1 sampling — draw record ``idx``'s crop anchor.

    Draws one of the record's chain/interface anchors with probability
    ``w_a / Σ w_a`` (from ``record_anchors``) to seed the crop; ``None`` (no
    table, or no valid anchor) leaves the cropper to self-seed. The per-idx seed
    means repeated draws of the same record within one ``epoch_size`` chunk
    reuse the anchor — unbiased across chunks (each reseeds). See
    ``data/sample/weighted.py`` for the decomposition.
    """
    if record_anchors is None:
        return None
    table = record_anchors.get(idx)
    if table is None:
        return None
    anchors, weights = table
    rng = _per_sample_rng(idx, 0x85EBCA6B)
    return anchors[rng.choices(range(len(anchors)), weights=weights, k=1)[0]]


class TrainingDataset(torch.utils.data.Dataset):
    """PyTorch dataset for DeepFold3 training.

    Loads Record -> Structure NPZ + MSA NPZ -> tokenize -> crop -> featurize.

    Args:
        records: List of Record objects.
        data_dir: Root directory containing structures/ and msa/ subdirectories.
        max_tokens: Maximum tokens per sample.
        max_atoms: Maximum atoms per sample.
        max_msa_seqs: Maximum MSA sequences.
    """

    def __init__(
        self,
        records: list[Record],
        data_dir: Path,
        max_tokens: int = 384,
        max_atoms: int = 4096,
        max_msa_seqs: int = const.MAX_MSA_ROWS,
        is_distillation: bool = False,
        mmcif_dir: Path | None = None,
        max_templates: int = 4,
        template_cutoff_days: int = 60,
        max_template_date: str | None = None,
        default_template_cutoff_date: str | None = None,
        prefer_monomer_templates: bool = True,
        max_neighborhood: int = 40,
        contiguous_crop_weight: float = 0.2,
        spatial_crop_weight: float = 0.4,
        interface_crop_weight: float = 0.4,
        interface_distance: float = 15.0,
        max_chains: int = 0,
        atomize_prob: float = 0.0,
        atomize_source: str = "random",
        atomize_max_residues: int = 0,
        anchor_sampling: bool = False,
        name: str = "",
        ccd_cache_path: str | Path | None = None,
        exclude_loss_mol_types: tuple[int, ...] | list[int] = (),
        pdb_chain_cache_dir: str | Path | None = None,
        pdb_chain_cache_mem: int = 512,
        slow_sample_log_threshold_s: float = 0.0,
        msa_crop_size: int = 1024,
        n_msa_crops: int = 0,
        template_struct_space: bool = False,
        pad_templates: bool = False,
    ):
        self.records = records
        # Dataset label used to disambiguate record ids in NaN-skip logs
        # and ``deepfold3 inspect-example`` triage. Record ids are NOT
        # guaranteed unique across extra datasets, so the trainer emits
        # ``{name}:{record.id}`` when ``name`` is set.
        self.name = name
        # Two-stage AF3 §2.5.1 anchor sampling (weighted datasets). The
        # sampler draws the record by the cluster marginal; this per-record
        # anchor conditional (chain/interface + ``w_a``) is drawn at load time
        # in ``_load_sample`` to seed the crop. ``None`` → the cropper
        # self-seeds (uniform / DistributedSampler datasets). See
        # ``data/sample/weighted.py`` for the decomposition.
        self._record_anchors = _build_record_anchors(records, anchor_sampling)
        self.data_dir = Path(data_dir)
        self.max_tokens = max_tokens
        self.max_atoms = max_atoms
        self.max_chains = max_chains
        # Stochastic atomization (training-only regularizer). Applied after
        # tokenization and BEFORE cropping so the crop budget accounts for the
        # per-atom expansion. See ``data/tokenize/atomize.py`` and
        # ``DataConfig.atomize_*``.
        self.atomize_prob = float(atomize_prob)
        self.atomize_source = atomize_source
        self.atomize_max_residues = int(atomize_max_residues)
        # Large-assembly chain "pre-crop" (>max_chains effective chains). Unlike
        # the old anchor-independent greedy selection (precomputed per record),
        # the OF3 ``crop_chainwise`` selection is spatial and depends on the
        # sampled anchor, so it must run per __getitem__ (see _load_sample). Only
        # the reference distance is stored here.
        self.chain_crop_interface_distance = interface_distance
        self.is_distillation = is_distillation
        # Per-dataset exclusion: tokens whose mol_type is in this list are
        # synthesised into a ``MASK_LOSS_WEIGHT`` channel (1.0 elsewhere,
        # 0.0 on excluded mol_types) at ``__getitem__`` time. The token mask
        # rides through cropping and featurisation; the loss orchestrator
        # picks it up automatically.
        self.exclude_loss_mol_types = tuple(int(m) for m in (exclude_loss_mol_types or ()))
        self.tokenizer = Tokenizer()
        self.cropper = Cropper(
            max_neighborhood=max_neighborhood,
            contiguous_weight=contiguous_crop_weight,
            spatial_weight=spatial_crop_weight,
            interface_weight=interface_crop_weight,
            interface_distance=interface_distance,
        )
        # Ligand automorphism cache for the training-time atom permutation
        # loss (AF3 §4.2). When ``ccd_cache_path`` is supplied (e.g.
        # ``/data/ccd_cache.json`` on the shared PVC), pull the
        # ``automorphisms`` subsection of the unified CCD cache. Otherwise
        # fall back to the per-dataset ``ligand_symmetry.json`` sidecar
        # for backwards compatibility. When neither is available the
        # featurizer runs with ``ligand_perm_cache=None`` and ligand
        # symmetry is simply disabled.
        ligand_perm_cache: dict | None = None
        # Order-preserving CCD ``atoms`` list per code (same index space as the
        # automorphism perms). Lets the featurizer place ligand / modified-residue
        # atoms by name → CCD index, correct regardless of structure atom order.
        ccd_atom_order: dict | None = None
        if ccd_cache_path is not None:
            from deepfold3.data.ccd import load_ligand_symmetry

            try:
                # Single parse of the (large, ref_pos-bearing) cache for both the
                # automorphism perms and their atom-order index space.
                ligand_perm_cache, ccd_atom_order = load_ligand_symmetry(ccd_cache_path)
                logger.info(
                    "Loaded %d ligand automorphism entries (+%d atom orders) from %s",
                    len(ligand_perm_cache),
                    len(ccd_atom_order),
                    ccd_cache_path,
                )
            except Exception:
                logger.warning("Failed to load %s; ligand symmetry disabled", ccd_cache_path)
        else:
            sidecar = self.data_dir / "ligand_symmetry.json"
            if sidecar.exists():
                try:
                    with open(sidecar) as f:
                        ligand_perm_cache = json.load(f)
                    logger.info("Loaded ligand symmetry cache: %d CCDs (%s)", len(ligand_perm_cache), sidecar)
                except Exception:
                    logger.warning("Failed to load %s; ligand symmetry disabled", sidecar)
        self.featurizer = Featurizer(
            max_msa_seqs=max_msa_seqs,
            # Forwarded so the padded template axis matches the dataset's own
            # sampling cap (the Featurizer default would silently disagree).
            max_templates=max_templates,
            ligand_perm_cache=ligand_perm_cache,
            msa_crop_size=msa_crop_size,
            n_msa_crops=n_msa_crops,
            ccd_atom_order=ccd_atom_order,
            pad_templates=pad_templates,
        )
        # Mask/template availability is checked lazily per record at __getitem__
        # time. Pre-scanning would walk hundreds of thousands of files per worker
        # under DDP and is unnecessary — kernel dentry cache makes the per-record
        # exists() effectively free after the first hit.
        pack.register(self.data_dir)
        self._has_masks = pack.is_dir(self.data_dir / "masks")
        self.mmcif_dir = Path(mmcif_dir) if mmcif_dir else None
        self.max_templates = max_templates
        self.template_cutoff_days = template_cutoff_days
        self.max_template_date = max_template_date
        self.default_template_cutoff_date = default_template_cutoff_date
        self.prefer_monomer_templates = prefer_monomer_templates
        # Template hit files store structure-space (0-based resolved-residue)
        # template indices instead of SEQRES-space ones. When set, the
        # seqres→structure realignment is skipped at feature-extraction time.
        # Applies to BOTH the monomer and multimer index files: the frame is a
        # property of how the search was run, not of which loader reads it.
        self.template_struct_space = template_struct_space
        # One-shot guard for the legacy-index warning in ``_indices_are_struct``.
        self._warned_index_space = False
        self._has_templates = pack.is_dir(self.data_dir / "templates")
        self._has_monomer_templates = pack.is_dir(self.data_dir / "monomer_templates")
        # Per-chain template cache (memory LRU + hierarchical sharded disk).
        # Replaces the per-hit full-mmCIF gemmi parse with a cached per-chain
        # slice — the dominant, query-size-independent dataloader-stall cost.
        # ``mmcif_dir`` is the cold-parse fallback; with a fully pre-built cache
        # (``deepfold3 build-pdb-cache``) parsing never happens at train time.
        self.pdb_chain_cache = PdbChainCache(
            cache_dir=pdb_chain_cache_dir,
            mmcif_dir=self.mmcif_dir,
            mem_capacity=pdb_chain_cache_mem,
        )
        # Tier-4 diagnostics: warn (with phase breakdown) when a single
        # __getitem__ exceeds this wall-clock budget so periodic step-time
        # spikes can be attributed to specific records. 0 = disabled.
        self.slow_sample_log_threshold_s = float(slow_sample_log_threshold_s)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return _retry_getitem(self, idx, self._load_sample, self._sample_id)

    def _sample_id(self, idx: int) -> str:
        return self.records[idx].id

    def _load_sample(self, idx: int) -> dict[str, torch.Tensor]:
        # Resolve idx → record; draw its crop anchor (AF3 §2.5.1 conditional,
        # weighted datasets only — else None and the cropper self-seeds).
        record = self.records[idx]
        crop_anchor: tuple[str, int, int] | None = _draw_crop_anchor(self._record_anchors, idx)

        # Per-phase wall-clock breakdown for Tier-4 slow-sample diagnostics.
        timing: dict[str, float] = {}
        _t = time.perf_counter()

        # Load structure
        struct_path = structure_path(self.data_dir / "structures", record.id)
        if not pack.exists(struct_path):
            raise FileNotFoundError(f"Structure file not found: {struct_path}")
        structure = Structure.load(struct_path)

        # Training-time large-assembly chain "pre-crop" (AF3 §2.5.4 / Table 6):
        # keep the spatially-closest ``max_chains`` chains around a random
        # reference atom in the sampled anchor (chain/interface), re-include
        # proximal ligands, and always keep the preferred chain/interface (OF3
        # ``crop_chainwise``). Runs per sample because it is anchor- and
        # geometry-dependent. The anchor drawn by the sampler is threaded through
        # so it is never dropped (the crop then seeds on a surviving chain).
        dropped_chain_ids: set[int] = set()
        if self.max_chains > 0:
            from deepfold3.data.prepare.filters import select_interface_chains

            all_ids = {int(a) for a in structure.chains["asym_id"]}
            # Deterministic under a fixed global seed (determinism mode), varies
            # per sample and per epoch (torch worker seed reseeded per epoch). A
            # distinct salt from the anchor draw decorrelates the two streams.
            precrop_rng = _per_sample_rng(idx, 0x9E3779B1)
            kept = select_interface_chains(
                structure,
                anchor=crop_anchor,
                max_chains=self.max_chains,
                interface_distance=self.chain_crop_interface_distance,
                ligand_inclusion_distance=5.0,  # OF3 default ligand_inclusion_distance
                rng=precrop_rng,
            )
            dropped_chain_ids = all_ids - kept
            if dropped_chain_ids:
                mask = np.isin(structure.chains["asym_id"], list(kept))
                structure = Structure(
                    atoms=structure.atoms,
                    residues=structure.residues,
                    chains=structure.chains[mask],
                    bonds=structure.bonds,
                    name=structure.name,
                )
        timing["structure"], _t = time.perf_counter() - _t, time.perf_counter()

        # Tokenize on the structure ALONE (empty MSA). Tokenization never reads
        # MSA content — it only builds ``msa_col_idx`` from the structure — so
        # the expensive per-chain MSA load and template parse are deferred until
        # AFTER cropping, when we know which (few) chains actually survive. For a
        # giant assembly cropped to ``max_tokens`` this turns O(all chains) I/O
        # into O(cropped chains), which is the dominant dataloader-stall fix.
        inp = Input(structure=structure, paired_msa={}, unpaired_msa={})
        tokenized = self.tokenizer.tokenize(inp)

        # Skip disk-stored masks even when present — see _DISK_TOKEN_MASKS_ENABLED.
        if _DISK_TOKEN_MASKS_ENABLED and self._has_masks:
            mp = mask_path(self.data_dir / "masks", record.id)
            if pack.exists(mp):
                tokenized = tokenized.with_mask(TokenMask.load(str(mp)))

        # Synthesize the per-token loss-weight channel from mol_type when the
        # dataset config marks any mol_types for exclusion (e.g. PATCHR
        # ligands). Must run before cropping so the channel is sliced
        # alongside the rest of the token state.
        if self.exclude_loss_mol_types:
            from deepfold3.data.masking import MASK_LOSS_WEIGHT

            arr = TokenMask.loss_weight_excluding_mol_types(tokenized.tokens, self.exclude_loss_mol_types)
            existing = tokenized.token_mask
            if existing is None:
                tokenized = tokenized.with_mask(TokenMask({MASK_LOSS_WEIGHT: arr}))
            else:
                channels = dict(existing.channels)
                channels[MASK_LOSS_WEIGHT] = arr
                tokenized = tokenized.with_mask(TokenMask(channels))

        # Stochastic atomization (training regularizer): expand a small random
        # subset of standard residues into per-atom tokens BEFORE cropping, so
        # the crop's token budget accounts for the expansion. No-op at
        # atomize_prob=0 (default). Inference paths never call this.
        tokenized = maybe_atomize(
            tokenized,
            idx,
            prob=self.atomize_prob,
            source=self.atomize_source,
            max_residues=self.atomize_max_residues,
            interface_distance=self.cropper.interface_distance,
        )

        tokenized = self.cropper.crop(tokenized, self.max_tokens, self.max_atoms, anchor=crop_anchor)
        timing["tokenize_crop"], _t = time.perf_counter() - _t, time.perf_counter()

        # Chains that survived the crop, and the chain-local MSA columns each
        # references. Only these chains need MSA / templates loaded.
        surviving_asym = {int(a) for a in tokenized.tokens["asym_id"]}
        chain_msa_cols = compute_chain_msa_cols(tokenized.tokens["asym_id"], tokenized.msa_col_idx)

        # Load per-chain MSAs (unpaired + optional paired) for surviving chains
        # only, sliced to the cropped columns at load time. The sliced MSAs are
        # re-indexed 0..k-1, so ``msa_col_idx`` must be rebased to match or the
        # featurizer gathers stale (mis-aligned / dropped) columns.
        paired_msa, unpaired_msa = self._load_cropped_msa(record, tokenized, surviving_asym, chain_msa_cols)
        # Original (pre-rebase) chain-local column per token == the per-chain
        # template array's query index; keep it for correct template placement
        # before rebasing msa_col_idx for the sliced MSA gather.
        template_col_idx = tokenized.query_col_idx
        remapped_col_idx = remap_msa_col_idx_to_slice(
            tokenized.tokens["asym_id"], tokenized.msa_col_idx, chain_msa_cols
        )
        tokenized = replace(tokenized, paired_msa=paired_msa, unpaired_msa=unpaired_msa, msa_col_idx=remapped_col_idx)
        timing["msa"], _t = time.perf_counter() - _t, time.perf_counter()

        # Load templates for surviving chains only: prefer per-sequence monomer
        # (AF3 §2.4) when available, fall back to multimer otherwise.
        template_feats_per_chain: dict[int, list[TemplateFeatures]] | None = None
        if self.mmcif_dir or self.pdb_chain_cache.cache_dir is not None:
            if self.prefer_monomer_templates and self._has_monomer_templates:
                template_feats_per_chain = self._load_monomer_templates(record, surviving_asym, dropped_chain_ids)
            elif self._has_templates:
                tpl_index = template_path(self.data_dir / "templates", record.id)
                if pack.exists(tpl_index):
                    # AF3 §2.4 sampling (``k = min(Uniform[0, n_avail],
                    # max_templates)``) is applied inside
                    # ``_load_multimer_templates`` from the actual hit count.
                    template_feats_per_chain = self._load_multimer_templates(record, surviving_asym)
        timing["template"], _t = time.perf_counter() - _t, time.perf_counter()

        feats = self.featurizer.process(
            tokenized,
            training=True,
            random_rot=True,
            template_feats_per_chain=template_feats_per_chain,
            template_col_idx=template_col_idx,
        )
        timing["featurize"] = time.perf_counter() - _t

        # Tier-4: attribute periodic step-time spikes to specific records.
        if self.slow_sample_log_threshold_s > 0:
            total = sum(timing.values())
            if total >= self.slow_sample_log_threshold_s:
                breakdown = " ".join(f"{k}={v:.2f}" for k, v in timing.items())
                logger.warning(
                    "slow sample %s: %.2fs (%d chains→%d cropped) | %s",
                    record.id,
                    total,
                    len(record.chains),
                    len(surviving_asym),
                    breakdown,
                )

        # Resolution from manifest metadata (drives the AF3 §4.3 confidence-loss
        # gate). Emitted UNCONDITIONALLY — see ``const.RESOLUTION_UNKNOWN``: a
        # 0-dim key that only some samples carry is dropped from the whole batch
        # by ``collate_fn``, taking the gate down with it.
        resolution = None if record.structure_info is None else record.structure_info.resolution
        feats["resolution"] = torch.tensor(
            const.RESOLUTION_UNKNOWN if resolution is None else resolution, dtype=torch.float32
        )

        # Distillation flag: disables confidence/distogram losses
        feats["is_distillation"] = torch.tensor(self.is_distillation, dtype=torch.bool)

        # Non-tensor diagnostic: record id threads through the trainer so that
        # NaN/Inf skips can name the offending example. Popped in trainer.py
        # before the forward pass so it never reaches the model. Prefixed
        # with the dataset name when set so record ids stay unambiguous
        # across ``--dataset`` extras that can share raw PDB ids.
        feats["_record_id"] = f"{self.name}:{record.id}" if self.name else record.id

        return feats

    def _load_cropped_msa(
        self,
        record: Record,
        tokenized: Tokenized,
        surviving_asym: set[int],
        chain_msa_cols: dict[int, np.ndarray],
    ) -> tuple[dict[int, MSA], dict[int, MSA]]:
        """Load unpaired + paired MSAs for surviving chains, sliced to cropped cols.

        Only chains present in the crop (``surviving_asym``) with at least one
        retained MSA column (``chain_msa_cols``) are read from disk. Each chain's
        MSA is column-sliced to exactly the tokens that survived — equivalent to
        the cropper's former post-load slice, but without ever loading the MSAs
        of chains the crop discarded.

        A chain with no MSA on disk falls back to a single-sequence MSA built
        from its own tokens (:meth:`_self_msa_from_tokens`), matching what the
        inference path already does via ``_sequence_to_msa``.
        """
        msa_dir = self.data_dir / "msa"
        unpaired_msa: dict[int, MSA] = {}
        paired_msa: dict[int, MSA] = {}
        for chain_info in record.chains:
            cid = chain_info.chain_id
            if cid not in surviving_asym:
                continue
            cols = chain_msa_cols.get(cid)
            if cols is None:
                continue  # ligand / chain contributes no MSA column to the crop
            msa_path = unpaired_msa_path(msa_dir, str(chain_info.msa_id))
            if pack.exists(msa_path):
                m = self._load_msa_sliced(msa_path, cols)
                if m is not None:
                    unpaired_msa[cid] = m
            if cid not in unpaired_msa:
                # No MSA on disk (or it was empty / corrupt). Without a fallback
                # the chain is simply absent from ``unpaired_msa``, and the
                # featurizer leaves its columns at MSA_GAP with msa_mask=True —
                # the model never even sees the query sequence, and the chain's
                # profile / deletion_mean are all-zero. Inference never hits this
                # (``from_sequence`` / ``from_json`` fall back to
                # ``_sequence_to_msa``), so leaving it unhandled here is a pure
                # train/inference skew.
                unpaired_msa[cid] = self._self_msa_from_tokens(tokenized, cid, cols)
                # Warn for protein only. MSA search is protein-only, so a nucleic
                # chain having no MSA is by construction, not a coverage gap —
                # and it is the common case (on the patchr set, ~76% of chains
                # reaching this branch are DNA/RNA), so warning on it buries the
                # protein chains that genuinely should have had an MSA. Short
                # peptides below the search length still warn: they are cheap to
                # ignore and indistinguishable here from a real materialization
                # gap. Revisit if an RNA MSA pipeline lands.
                if chain_info.mol_type == const.MOL_PROTEIN:
                    _count_drop("MSA file missing (self-MSA fallback)", f"{record.id} chain {cid}: {msa_path}")
            if chain_info.paired_msa_id is not None:
                paired_path = paired_msa_path(msa_dir, str(chain_info.paired_msa_id))
                if pack.exists(paired_path):
                    m = self._load_msa_sliced(paired_path, cols)
                    if m is not None:
                        paired_msa[cid] = m
        return paired_msa, unpaired_msa

    @staticmethod
    def _self_msa_from_tokens(tokenized: Tokenized, cid: int, cols: np.ndarray) -> MSA:
        """Single-sequence MSA for one chain, read off its own tokens.

        ``res_type`` and the MSA vocabulary are the *same* 32-class space (AF3
        Table 5: "one-hot encoding of the processed MSA, using the same classes
        as restype") — ALA..VAL 0-19, UNK/X 20, gap 21, RNA 22-25, DNA 26-29,
        unknown-nucleic 30 — so token types are already MSA class indices and
        need no lookup. Building the row from tokens rather than from a 1-letter
        sequence also keeps modified-residue handling identical to the rest of
        the featurizer, which the string round-trip would have to re-derive.

        Returns a row already in *cropped-column* space (``cols``), matching what
        :meth:`_load_msa_sliced` returns, so the caller's rebased ``msa_col_idx``
        gathers it correctly. Columns no token claims stay ``MSA_GAP``.
        """
        pos = np.where(tokenized.tokens["asym_id"] == cid)[0]
        tok_cols = tokenized.msa_col_idx[pos]
        keep = tok_cols >= 0  # ligand / column-less tokens
        pos, tok_cols = pos[keep], tok_cols[keep]

        # col → res_type, then gather in the caller's column order. ``cols`` is
        # ordered by first appearance, not sorted, so index rather than align.
        width = int(tok_cols.max()) + 1 if len(tok_cols) else 0
        lut = np.full(max(width, 1), const.MSA_GAP, dtype=np.int8)
        if len(tok_cols):
            lut[tok_cols] = tokenized.tokens["res_type"][pos].astype(np.int8)
        in_range = cols < width
        row = np.full(len(cols), const.MSA_GAP, dtype=np.int8)
        row[in_range] = lut[cols[in_range]]

        return MSA(
            sequences=row[None],
            mask=np.ones((1, len(cols)), dtype=bool),
            deletion_matrix=np.zeros((1, len(cols)), dtype=np.int8),
        )

    @staticmethod
    def _load_msa_sliced(path: Path, cols: np.ndarray) -> MSA | None:
        """Load an MSA npz and column-slice it to ``cols`` (cropped columns)."""
        try:
            with read_npz(path) as data:
                seqs = data["sequences"]
                if seqs.shape[0] == 0 or seqs.shape[1] == 0:
                    logger.warning("Empty MSA %s (shape %s) — skipping", path, seqs.shape)
                    return None
                return MSA(
                    sequences=seqs[:, cols],
                    mask=data["mask"][:, cols],
                    deletion_matrix=data["deletion_matrix"][:, cols],
                )
        except (EOFError, ValueError, OSError, zipfile.BadZipFile, IndexError) as e:
            logger.warning("Corrupt MSA %s: %s — skipping", path, e)
            return None

    def _indices_are_struct(self, recorded: str | None) -> bool:
        """Resolve the template index frame for one hit.

        The value recorded in the index file wins: it is written by whatever
        produced the alignment and cannot drift out of sync with the data. The
        per-dataset ``template_struct_space`` flag is the fallback for files
        written before the field existed — warned about once per dataset,
        because a wrong fallback is invisible at runtime (templates just become
        noise) and the only cure is regenerating the index.
        """
        from deepfold3.data.prepare.template_search import INDEX_SPACE_STRUCT

        if recorded is not None:
            return recorded == INDEX_SPACE_STRUCT
        if not self._warned_index_space:
            self._warned_index_space = True
            logger.warning(
                "%s: template index files do not record `index_space`; falling back to "
                "template_struct_space=%s from the dataset config. If that is wrong, every "
                "template is silently misaligned by the template's unresolved N-terminal "
                "count. Re-run `deepfold3 prepare-templates` to stamp the frame into the "
                "index, or verify by re-deriving each hit's identity and comparing it to the "
                "identity the search recorded.",
                self.name or self.data_dir,
                self.template_struct_space,
            )
        return self.template_struct_space

    def _effective_template_cutoff(self, record: Record) -> str | None:
        """Effective template release-date cutoff for a record (fail-closed).

        ``min(max_template_date hard cap, per-record bound)``, where the
        per-record bound is ``released - cutoff_days`` when the query's release
        date is known, else the configured global ``default_template_cutoff_date``
        fallback. Returns ``None`` when no cutoff can be established — callers
        must then fail-closed (drop all hits) so a record with an unknown release
        date never leaks future-dated templates. Shared by the monomer and
        multimer template loaders.
        """
        import datetime

        cutoff: str | None = self.max_template_date
        query_release = record.structure_info.released if record.structure_info else None
        per_record: str | None = self.default_template_cutoff_date
        if query_release is not None:
            try:
                d = datetime.date.fromisoformat(query_release) - datetime.timedelta(days=self.template_cutoff_days)
                per_record = d.isoformat()
            except ValueError:
                pass
        if per_record is not None:
            cutoff = per_record if cutoff is None else min(cutoff, per_record)
        return cutoff

    def _load_monomer_templates(
        self,
        record: Record,
        surviving_asym: set[int],
        skip_chain_ids: set[int] | None = None,
    ) -> dict[int, list[TemplateFeatures]] | None:
        """Load per-chain monomer template features (AlphaFold 3 §2.4).

        For each protein chain in the record:

        1. Load the precomputed ``monomer_templates/{ab}/{msa_id}.json`` list
           of single-chain hits (already filtered by AF3 per-hit criteria at
           prepare time, sorted by e-value, capped at 20).
        2. Apply the **training-time** release-date cutoff:
           ``min(max_template_date, record.released - cutoff_days)``. The
           cutoff is per-record because the same sequence can appear in
           records with different release dates.
        3. Drop self-hits whose ``template_pdb_id`` matches the record id.
        4. Sample ``k = min(Uniform[0, n], max_templates)`` random hits, then
           extract template features from the local PDB mmCIF mirror.
        """
        from deepfold3.data.parse.templates import cigar_to_index_pairs
        from deepfold3.data.prepare.template_search import load_monomer_template_hits

        templates_root = self.data_dir / "monomer_templates"
        record_id_upper = record.id.upper()

        # Effective release-date cutoff (fail-closed when unestablished).
        cutoff = self._effective_template_cutoff(record)
        if cutoff is None:
            # Fail-closed is correct (never leak a future template), but it
            # drops 100% of this record's templates. Records whose manifest was
            # never enriched have structure_info=None, so with no
            # max_template_date / default_template_cutoff_date configured this
            # silently disables templates for the entire run — indistinguishable
            # from the leakage fix working as intended.
            _count_drop("no template cutoff establishable, dropping all templates", record.id)

        result: dict[int, list[TemplateFeatures]] = {}
        for chain_info in record.chains:
            if not chain_info.valid or chain_info.mol_type != const.MOL_PROTEIN:
                continue
            if chain_info.chain_id not in surviving_asym:
                continue  # crop discarded this chain — skip its template I/O
            if skip_chain_ids and chain_info.chain_id in skip_chain_ids:
                continue
            tpl_path = monomer_template_path(templates_root, str(chain_info.msa_id))
            if not pack.exists(tpl_path):
                continue
            try:
                hits = load_monomer_template_hits(tpl_path)
            except Exception as e:
                _count_drop("monomer template index failed to load", f"{chain_info.msa_id}: {e}", exc_info=True)
                continue

            # Date cutoff (fail-closed) + drop self-hits. A hit is kept only when
            # a cutoff exists and the hit's own release date is known and within
            # it — an unestablished cutoff or an undated hit is dropped, never
            # leaked.
            filtered = [
                h
                for h in hits
                if h.template_pdb_id.upper() != record_id_upper
                and cutoff is not None
                and h.release_date is not None
                and h.release_date <= cutoff
            ]
            if not filtered:
                continue

            # AF3: k ~ min(Uniform[0, n], max_templates)
            n_avail = len(filtered)
            k = min(random.randint(0, n_avail), self.max_templates)
            if k == 0:
                continue
            sampled = random.sample(filtered, k)

            chain_feats: list[TemplateFeatures] = []
            for hit in sampled:
                ct = self.pdb_chain_cache.get(hit.template_pdb_id, hit.template_chain_id)
                if ct is None:
                    continue
                try:
                    query_indices, seqres_indices = cigar_to_index_pairs(hit.cigar, hit.qstart, hit.tstart)
                    if not query_indices:
                        continue
                    query_to_seqres = dict(zip(query_indices, seqres_indices, strict=True))
                    feats = template_features_from_chain(
                        ct,
                        query_to_seqres,
                        chain_info.num_residues,
                        indices_are_struct=self._indices_are_struct(hit.index_space),
                    )
                except Exception as e:
                    _count_drop(
                        "monomer template extraction failed",
                        f"{hit.template_pdb_id}_{hit.template_chain_id} for {record.id}/{chain_info.chain_name}: {e}",
                        exc_info=True,
                    )
                    continue
                chain_feats.append(feats)

            if chain_feats:
                result[chain_info.chain_id] = chain_feats

        return result if result else None

    def _load_multimer_templates(
        self, record: Record, surviving_asym: set[int]
    ) -> dict[int, list[TemplateFeatures]] | None:
        """Load multimer template features from pre-indexed JSON + PdbChainCache.

        Applies the **same** runtime release-date cutoff as the monomer path
        (fail-closed) plus a self-hit drop. This is essential: the pre-indexed
        ``templates/`` JSON is NOT release-date filtered at prepare time, so
        without this filter it leaks future-dated templates (measured ~50% of
        hits postdate the query on patchr_training, worst case +2000 days).

        Sampling follows AF3 §2.4: ``k = min(Uniform[0, n_avail], max_templates)``.
        Per-chain features come from the :class:`PdbChainCache`; only alignments
        onto chains that survived the crop are extracted.
        """
        from deepfold3.data.parse.templates import cigar_to_index_pairs
        from deepfold3.data.prepare.template_search import load_template_hits

        tpl_path = template_path(self.data_dir / "templates", record.id)
        try:
            hits = load_template_hits(tpl_path)
        except Exception as e:
            _count_drop("multimer template index failed to load", f"{record.id}: {e}", exc_info=True)
            return None

        # Runtime date cutoff (fail-closed) + self-hit drop. A hit is kept only
        # when a cutoff exists and its (max-across-chains) release date is known
        # and within it — the prepare-time JSON never applied this.
        cutoff = self._effective_template_cutoff(record)
        if cutoff is None:
            _count_drop("no template cutoff establishable, dropping all templates", record.id)
        record_id_upper = record.id.upper()
        hits = [
            h
            for h in hits
            if h.template_pdb_id.upper() != record_id_upper
            and cutoff is not None
            and h.release_date is not None
            and h.release_date <= cutoff
        ]

        n_avail = len(hits)
        if n_avail == 0:
            return None
        # AF3 §2.4: k ~ min(Uniform[0, n_avail], max_templates).
        k = min(random.randint(0, n_avail), self.max_templates)
        if k == 0:
            return None
        hits = random.sample(hits, k) if n_avail > k else hits

        result: dict[int, list[TemplateFeatures]] = {}
        for hit in hits:
            # Per-chain dropout: randomly drop chain alignments so the model
            # sees partial multimer context.  Always keep at least one chain.
            alignments = hit.chain_alignments
            if len(alignments) > 1:
                alignments = [a for a in alignments if random.random() < 0.5]
                if not alignments:
                    alignments = [random.choice(hit.chain_alignments)]

            for aln in alignments:
                chain_info = record.chains[aln.query_chain_idx]
                if chain_info.chain_id not in surviving_asym:
                    continue  # crop discarded this chain — skip its template I/O
                query_indices, seqres_indices = cigar_to_index_pairs(aln.cigar, aln.qstart, aln.tstart)
                if not query_indices:
                    continue
                ct = self.pdb_chain_cache.get(hit.template_pdb_id, aln.template_chain_id)
                if ct is None:
                    continue
                try:
                    query_to_seqres = dict(zip(query_indices, seqres_indices, strict=True))
                    # ``template_struct_space`` applies here too: the multimer index
                    # is produced by the same search as the monomer one, so its
                    # template indices live in the same frame. Omitting this ran the
                    # seqres->structure realignment on already-structure-space
                    # indices, shifting every hit by the template's unresolved
                    # N-terminal count.
                    feats = template_features_from_chain(
                        ct,
                        query_to_seqres,
                        chain_info.num_residues,
                        indices_are_struct=self._indices_are_struct(hit.index_space),
                    )
                except Exception as e:
                    _count_drop(
                        "multimer template extraction failed",
                        f"{hit.template_pdb_id}_{aln.template_chain_id} for {record.id}: {e}",
                        exc_info=True,
                    )
                    continue
                result.setdefault(chain_info.chain_id, []).append(feats)

        return result if result else None


def _tokenize_and_featurize(
    inp: Input,
    max_msa_seqs: int = const.MAX_MSA_ROWS,
    templates: list[TemplateFeatures] | None = None,
    template_feats_per_chain: dict[int, list[TemplateFeatures]] | None = None,
    dedup_unpaired_against_paired: bool = False,
) -> dict[str, torch.Tensor]:
    """Shared tokenize -> featurize pipeline for inference inputs.

    ``templates`` is a whole-query flat list (JSON dialect); its rows are
    gathered by each token's index into the concatenated polymer sequences.
    ``template_feats_per_chain`` is per chain, gathered by the chain-local
    residue index. Either way the index comes from ``msa_col_idx``, so a chain
    holding a modified residue — which tokenizes to one token per atom — keeps
    its templates on the right residues instead of shifting everything after it.
    ``dedup_unpaired_against_paired`` is set for species-paired inputs; see
    ``Featurizer``.
    """
    tokenizer = Tokenizer()
    featurizer = Featurizer(max_msa_seqs=max_msa_seqs, dedup_unpaired_against_paired=dedup_unpaired_against_paired)
    tokenized = tokenizer.tokenize(inp)
    if template_feats_per_chain:
        col_idx = tokenized.query_col_idx
    else:
        col_idx = _whole_query_col_idx(tokenized)
    return featurizer.process(
        tokenized,
        training=False,
        random_rot=False,
        templates=templates,
        template_feats_per_chain=template_feats_per_chain,
        template_col_idx=col_idx,
    )


def _whole_query_col_idx(tokenized: Tokenized) -> np.ndarray | None:
    """Per-token index into the concatenated per-chain query sequences.

    The flat (JSON) template list is aligned to all polymer chains concatenated
    in chain order, so a token's row is its chain's residue offset plus its
    chain-local column. Non-polymer tokens stay at -1 and take no template row,
    matching how the offsets are counted (ligands contribute no columns).

    Inference only: chain lengths are read off the tokens present, which assumes
    nothing was cropped away.
    """
    col_idx = tokenized.query_col_idx
    if col_idx is None:
        return None
    asym = tokenized.tokens["asym_id"]
    out = np.full(len(col_idx), -1, dtype=np.int64)
    offset = 0
    for aid in sorted({int(a) for a in asym}):
        sel = np.where((asym == aid) & (col_idx >= 0))[0]
        if len(sel) == 0:
            continue
        cols = col_idx[sel].astype(np.int64)
        out[sel] = cols + offset
        offset += int(cols.max()) + 1
    return out


class InferenceInput:
    """Build model-ready features from a raw sequence string, A3M, or JSON.

    Uses CCD ideal coordinates for structure building.  MSA is either
    parsed from an A3M file or synthesized as a single-sequence placeholder.
    """

    @staticmethod
    def from_a3m(
        a3m_text: str,
        mol_type: str = "protein",
        max_msa_seqs: int = const.MAX_MSA_ROWS,
        templates: list[TemplateFeatures] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build features from an A3M-format MSA string.

        The query sequence (first record) determines the structure;
        all sequences contribute to the MSA features.

        Args:
            a3m_text: Full A3M text (headers + sequences).
            mol_type: 'protein', 'rna', or 'dna'.
            max_msa_seqs: Maximum MSA sequences to keep.
            templates: Optional list of TemplateFeatures for template embedding.
        """
        from deepfold3.data.parse.a3m import parse_a3m

        mt = const.MOL_TYPE_IDS.get(mol_type.upper(), const.MOL_PROTEIN)

        msa_obj = parse_a3m(a3m_text, mt)
        if msa_obj.sequences.shape[0] == 0:
            raise ValueError("A3M file contains no sequences")

        query_seq = _msa_indices_to_sequence(msa_obj.sequences[0], mt)
        structure = _sequence_to_structure(query_seq, mt)
        inp = Input(structure=structure, paired_msa={}, unpaired_msa={0: msa_obj})
        return _tokenize_and_featurize(inp, max_msa_seqs=max_msa_seqs, templates=templates)

    @staticmethod
    def from_sequence(
        sequence: str,
        mol_type: str = "protein",
        max_msa_seqs: int = 1,
        templates: list[TemplateFeatures] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build features for a single sequence.

        Args:
            sequence: 1-letter sequence string (protein) or RNA/DNA sequence.
            mol_type: 'protein', 'rna', or 'dna'.
            max_msa_seqs: Maximum MSA sequences (default 1 = single sequence).
            templates: Optional list of TemplateFeatures for template embedding.

        Returns:
            dict[str, torch.Tensor] ready for model.forward().
        """
        mt = const.MOL_TYPE_IDS.get(mol_type.upper(), const.MOL_PROTEIN)
        structure = _sequence_to_structure(sequence, mt)
        msa = _sequence_to_msa(sequence, mt)
        inp = Input(structure=structure, paired_msa={}, unpaired_msa={0: msa})
        return _tokenize_and_featurize(inp, max_msa_seqs=max_msa_seqs, templates=templates)

    @staticmethod
    def from_json(
        json_path: str | Path,
        job_index: int = 0,
        max_msa_seqs: int = const.MAX_MSA_ROWS,
        max_templates: int = 4,
        template_cache_dir: str | Path | None = None,
        pdb_mmcif_dir: str | Path | None = None,
        msa_pairing: str = "auto",
    ) -> dict[str, torch.Tensor]:
        """Build features from a Protenix-format or AF3-format JSON input file.

        Supports multi-chain complexes with protein, DNA, and RNA chains.
        Each chain can optionally specify paired/unpaired MSAs and templates.

        All file I/O (reading A3M files, downloading template CIFs) is handled
        by ``parse_json_input()``.  This method only does computation.

        Args:
            json_path: Path to the JSON input file.
            job_index: Which job entry to use (default 0).
            max_msa_seqs: Maximum MSA sequences to keep.
            max_templates: Maximum number of templates to extract.
            template_cache_dir: Directory to cache downloaded template mmCIF files.
            pdb_mmcif_dir: Optional local PDB mmCIF mirror directory
                (layout: ``{dir}/{mid2}/{id}.cif.gz``).
            msa_pairing: ``"auto"``, ``"precomputed"`` or ``"species"``; see
                ``parse_json_input``.

        Returns:
            dict[str, torch.Tensor] ready for model.forward().
        """
        from deepfold3.data.parse.json_input import parse_json_input

        json_path = Path(json_path)
        jobs = parse_json_input(
            json_path,
            max_templates=max_templates,
            template_cache_dir=template_cache_dir,
            pdb_mmcif_dir=pdb_mmcif_dir,
            msa_pairing=msa_pairing,
        )

        if job_index >= len(jobs):
            raise IndexError(f"Job index {job_index} out of range (file has {len(jobs)} jobs)")

        return InferenceInput.from_job(jobs[job_index], max_msa_seqs=max_msa_seqs)

    @staticmethod
    def from_job(
        job,
        max_msa_seqs: int = const.MAX_MSA_ROWS,
    ) -> dict[str, torch.Tensor]:
        """Build features from a pre-parsed JobInput object.

        Use this to avoid double-parsing when you already have a JobInput
        (e.g., from ``parse_json_input()``).

        Args:
            job: A ``JobInput`` from ``parse_json_input()``.
            max_msa_seqs: Maximum MSA sequences to keep.

        Returns:
            dict[str, torch.Tensor] ready for model.forward().
        """
        from deepfold3.data.parse.templates import TemplateHitInput, template_hit_input_to_features

        # Collect all polymer chains in order: protein, DNA, RNA
        # Each entry: (sequence, mol_type, paired_msa, unpaired_msa, template_hits)
        chain_specs: list[tuple[str, int, MSA | None, MSA | None, list[TemplateHitInput]]] = []
        for pc in job.protein_chains:
            chain_specs.append((pc.sequence, const.MOL_PROTEIN, pc.paired_msa, pc.unpaired_msa, pc.template_hits))
        for dc in job.dna_chains:
            chain_specs.append((dc.sequence, dc.mol_type, dc.paired_msa, dc.unpaired_msa, []))
        for rc in job.rna_chains:
            chain_specs.append((rc.sequence, rc.mol_type, rc.paired_msa, rc.unpaired_msa, []))

        if not chain_specs and not job.ligand_chains:
            raise ValueError(f"No chains found in job {job.name!r}")

        # Build chain names
        next_chain_idx = 0
        chain_tuples = []
        for seq, mt, _, _, _ in chain_specs:
            chain_tuples.append((seq, mt, chr(ord("A") + next_chain_idx)))
            next_chain_idx += 1

        # TODO: structure builder must consume chain.modifications (per-residue
        # CCD substitutions for PTMs / modified bases), job.bonded_atom_pairs
        # (explicit covalent inter-entity bonds), and job.user_ccd (inline mmCIF
        # block registering custom CCD components).  Currently parsed by
        # parse_json_input() but silently ignored during structure assembly.

        # Build ligand molecules
        ligand_mol_chains: list[tuple[CcdMolecule, str]] = []
        if job.ligand_chains:
            from deepfold3.data.ccd import load_ccd_molecule, smiles_to_molecule

            for lig in job.ligand_chains:
                if lig.ccd_ids:
                    for ccd_code in lig.ccd_ids:
                        mol = load_ccd_molecule(ccd_code)
                        chain_name = chr(ord("A") + next_chain_idx)
                        ligand_mol_chains.append((mol, chain_name))
                        next_chain_idx += 1
                elif lig.smiles:
                    mol = smiles_to_molecule(lig.smiles)
                    chain_name = chr(ord("A") + next_chain_idx)
                    ligand_mol_chains.append((mol, chain_name))
                    next_chain_idx += 1

        # Build structure (with ligands if present)
        if ligand_mol_chains:
            structure = _build_complex_structure_with_ligands(chain_tuples, ligand_mol_chains)
        elif chain_tuples:
            structure = _build_complex_structure(chain_tuples)
        else:
            raise ValueError(f"No valid chains in job {job.name!r}")

        # Assemble MSA dicts from parsed objects (polymer chains only)
        paired_msa_dict: dict[int, MSA] = {}
        unpaired_msa_dict: dict[int, MSA] = {}
        for asym_id, (seq, mt, paired_msa, unpaired_msa, _) in enumerate(chain_specs):
            if paired_msa is not None:
                paired_msa_dict[asym_id] = paired_msa
            if unpaired_msa is not None:
                unpaired_msa_dict[asym_id] = unpaired_msa
            elif paired_msa is None:
                # No MSA at all — fall back to single-sequence MSA
                unpaired_msa_dict[asym_id] = _sequence_to_msa(seq, mt)
            else:
                # Only paired exists — use single-sequence as unpaired
                unpaired_msa_dict[asym_id] = _sequence_to_msa(seq, mt)
        # Ligand chains don't have MSA — skip

        # Convert template hits to features
        # Every hit is aligned to its own chain: ``queryIndices`` in the AF3
        # dialect and the HHR/templatesPath forms all count from that chain's
        # first residue.
        template_feats_per_chain: dict[int, list[TemplateFeatures]] = {}
        for asym_id, (seq, mt, _, _, tmpl_hits) in enumerate(chain_specs):
            if mt == const.MOL_PROTEIN and tmpl_hits:
                template_feats_per_chain[asym_id] = [template_hit_input_to_features(h, len(seq)) for h in tmpl_hits]

        inp = Input(structure=structure, paired_msa=paired_msa_dict, unpaired_msa=unpaired_msa_dict)
        # A single templated chain goes through the per-chain path too: the flat
        # path is aligned to the *whole* query, so handing it one chain's array
        # only works when that chain is first and every other chain follows it.
        return _tokenize_and_featurize(
            inp,
            max_msa_seqs=max_msa_seqs,
            template_feats_per_chain=template_feats_per_chain or None,
            # Species-paired blocks are built from raw UniProt hits, so (as in
            # OF3 / AF3) unpaired rows repeating a paired row are dropped.
            dedup_unpaired_against_paired=getattr(job, "msa_pairing", "precomputed") == "species"
            and bool(paired_msa_dict),
        )


def job_chain_ids(job) -> list[str | None]:
    """Chain names from the input JSON, in ``asym_id`` order.

    ``InferenceInput.from_job`` names its internal chains ``A``, ``B``, ``C``, …
    in the order protein → DNA → RNA → ligand, so ``asym_id`` follows that order
    and the ids the caller wrote in the JSON are otherwise lost by output time.
    This returns the original ids in the same order, with ``None`` wherever the
    dialect supplies no name (AlphaFold-Server / Protenix are count-based) so
    writers can fall back to the synthetic letter for that chain.

    A ligand entry with several ``ccdCodes`` is expanded by ``from_job`` into one
    chain per code, so a single id cannot label them all: such a group yields
    ``None`` throughout rather than guessing which expanded chain owns the name.

    The ordering here MUST match ``from_job``; ``tests/test_json_chain_ids.py``
    pins the two together.
    """
    ids: list[str | None] = []
    for pc in job.protein_chains:
        ids.append(getattr(pc, "chain_id", None))
    for dc in job.dna_chains:
        ids.append(getattr(dc, "chain_id", None))
    for rc in job.rna_chains:
        ids.append(getattr(rc, "chain_id", None))
    for lig in job.ligand_chains:
        n = len(lig.ccd_ids) if lig.ccd_ids else 1
        cid = getattr(lig, "chain_id", None)
        ids.extend([cid if n == 1 else None] * n)
    return ids


def _letter_to_res_name(letter: str, mol_type: int) -> str:
    """Map a single-character residue letter to its canonical residue name."""
    if mol_type == const.MOL_PROTEIN:
        return const.ONE_LETTER_TO_THREE.get(letter, "UNK")
    elif mol_type == const.MOL_RNA:
        return const.RNA_LETTER_TO_RES.get(letter, "N")
    elif mol_type == const.MOL_DNA:
        return const.DNA_LETTER_TO_RES.get(letter, "DN")
    return "UNK"


def _build_chain_atoms_residues(
    sequence: str,
    mol_type: int,
    atom_offset: int,
) -> tuple[list[tuple], list[tuple], int]:
    """Builds atom and residue tuples for a single polymer chain.

    Args:
        sequence: 1-letter sequence string.
        mol_type: Molecule type (MOL_PROTEIN, MOL_DNA, MOL_RNA).
        atom_offset: Global atom index offset for this chain.

    Returns:
        A tuple of (atoms_list, residues_list, total_atoms_in_chain).
    """
    atoms_list: list[tuple] = []
    residues_list: list[tuple] = []

    atom_idx = atom_offset
    for ri, letter in enumerate(sequence):
        res_name_3 = _letter_to_res_name(letter, mol_type)
        res_type = const.res_name_to_type(res_name_3)

        ref = get_ref_atoms(res_name_3, mol_type)
        n_atoms = int(ref.mask.sum())

        for j in range(n_atoms):
            atoms_list.append(
                (
                    float(ref.positions[j, 0]),
                    float(ref.positions[j, 1]),
                    float(ref.positions[j, 2]),
                    int(ref.element[j]),
                    0.0,  # charge
                    _atom_name_from_chars(ref.atom_name_chars[j]).encode("ascii"),
                    True,
                    0.0,  # b_factor
                )
            )

        residues_list.append(
            (
                res_name_3[:5].encode("ascii"),
                res_type,
                mol_type,
                ri + 1,  # res_id (1-based)
                atom_idx,
                n_atoms,
                True,  # is_standard
                True,  # is_present
            )
        )
        atom_idx += n_atoms

    return atoms_list, residues_list, atom_idx - atom_offset


def _build_ligand_chain_atoms_residues(
    mol: CcdMolecule,
    atom_offset: int,
) -> tuple[list[tuple], list[tuple], list[tuple], int]:
    """Builds atom, bond, and residue tuples for a ligand chain.

    Each CCD component becomes one residue with all its heavy atoms.

    Args:
        mol: CcdMolecule with atom data and bonds.
        atom_offset: Global atom index offset for this chain.

    Returns:
        A tuple of (atoms_list, bonds_list, residues_list, total_atoms).
    """
    atoms_list: list[tuple] = []
    bonds_list: list[tuple] = []

    for j, name in enumerate(mol.atom_names):
        elem_int = const.ELEMENT_TO_INT.get(mol.elements[j].upper(), 6)
        atoms_list.append(
            (
                float(mol.positions[j, 0]),
                float(mol.positions[j, 1]),
                float(mol.positions[j, 2]),
                int(np.clip(elem_int, 0, 127)),
                float(mol.charges[j]),
                name[:4].encode("ascii"),
                True,
                0.0,  # b_factor
            )
        )

    for a1, a2, order in mol.bonds:
        bonds_list.append((atom_offset + a1, atom_offset + a2, order))

    n_atoms = len(mol.atom_names)
    res_type = 20  # UNK — ligand tokens use res_type=20 in the model
    residues_list = [
        (
            mol.res_name[:5].encode("ascii"),
            res_type,
            const.MOL_NONPOLYMER,
            1,  # res_id
            atom_offset,
            n_atoms,
            False,  # is_standard
            True,  # is_present
        )
    ]

    return atoms_list, bonds_list, residues_list, n_atoms


def _build_complex_structure(chains: list[tuple[str, int, str]]) -> Structure:
    """Builds a multi-chain Structure with sequence-based entity_id grouping.

    Chains with identical (normalized) sequences share the same entity_id,
    with incrementing sym_id to distinguish copies.

    Args:
        chains: List of (sequence, mol_type, chain_name) tuples.

    Returns:
        Structure with properly assigned entity_id and sym_id.
    """
    # Map normalized sequences to (entity_id, next_sym_id)
    seq_to_entity: dict[str, list[int]] = {}  # normalized_seq -> [entity_id, next_sym_id]
    next_entity_id = 1

    all_atoms: list[tuple] = []
    all_residues: list[tuple] = []
    all_chains: list[tuple] = []

    atom_offset = 0
    res_offset = 0

    for chain_idx, (sequence, mol_type, chain_name) in enumerate(chains):
        sequence = sequence.strip().upper()
        n_residues = len(sequence)

        # Determine entity_id and sym_id based on sequence identity
        if sequence in seq_to_entity:
            entity_id = seq_to_entity[sequence][0]
            sym_id = seq_to_entity[sequence][1]
            seq_to_entity[sequence][1] += 1
        else:
            entity_id = next_entity_id
            sym_id = 0
            seq_to_entity[sequence] = [entity_id, 1]
            next_entity_id += 1

        # Build atoms and residues for this chain
        chain_atoms, chain_residues, chain_atom_count = _build_chain_atoms_residues(sequence, mol_type, atom_offset)
        all_atoms.extend(chain_atoms)
        all_residues.extend(chain_residues)

        all_chains.append(
            (
                chain_name.encode("ascii"),
                mol_type,
                entity_id,  # sequence-based entity_id (1-based)
                sym_id,  # copy index within entity (0-based)
                chain_idx,  # asym_id (unique chain index)
                res_offset,  # res_idx
                n_residues,  # res_num
                atom_offset,  # atom_idx
                chain_atom_count,  # atom_num
            )
        )

        atom_offset += chain_atom_count
        res_offset += n_residues

    atoms_arr = np.array(all_atoms, dtype=ATOM_DTYPE) if all_atoms else np.empty(0, dtype=ATOM_DTYPE)
    residues_arr = np.array(all_residues, dtype=RESIDUE_DTYPE) if all_residues else np.empty(0, dtype=RESIDUE_DTYPE)
    chains_arr = np.array(all_chains, dtype=CHAIN_DTYPE) if all_chains else np.empty(0, dtype=CHAIN_DTYPE)
    bonds_arr = np.empty(0, dtype=BOND_DTYPE)

    return Structure(
        atoms=atoms_arr,
        bonds=bonds_arr,
        residues=residues_arr,
        chains=chains_arr,
    )


def _build_complex_structure_with_ligands(
    chains: list[tuple[str, int, str]],
    ligand_chains: list[tuple[CcdMolecule, str]],
) -> Structure:
    """Builds a multi-chain Structure with both polymer and ligand chains.

    Args:
        chains: List of (sequence, mol_type, chain_name) tuples for polymer chains.
        ligand_chains: List of (CcdMolecule, chain_name) tuples for ligand chains.

    Returns:
        Structure with entity_id, sym_id, and intra-ligand bonds.
    """
    # Start with polymer chains
    structure = _build_complex_structure(chains)

    if not ligand_chains:
        return structure

    # Extend with ligand chains
    all_atoms = list(structure.atoms)
    all_bonds = list(structure.bonds)
    all_residues = list(structure.residues)
    all_chains = list(structure.chains)

    atom_offset = len(structure.atoms)
    res_offset = len(structure.residues)
    chain_idx = len(chains)

    # Entity tracking: ligand sequences grouped by res_name for entity_id
    seq_to_entity: dict[str, list[int]] = {}
    next_entity_id = max((int(c["entity_id"]) for c in structure.chains), default=0) + 1

    for mol, chain_name in ligand_chains:
        lig_atoms, lig_bonds, lig_residues, n_atoms = _build_ligand_chain_atoms_residues(mol, atom_offset)
        all_atoms.extend(lig_atoms)
        all_residues.extend(lig_residues)

        # Add bonds as BOND_DTYPE
        for a1, a2, order in lig_bonds:
            all_bonds.append((a1, a2, order))

        # Entity/sym assignment by ligand res_name
        seq_key = mol.res_name
        if seq_key in seq_to_entity:
            entity_id = seq_to_entity[seq_key][0]
            sym_id = seq_to_entity[seq_key][1]
            seq_to_entity[seq_key][1] += 1
        else:
            entity_id = next_entity_id
            sym_id = 0
            seq_to_entity[seq_key] = [entity_id, 1]
            next_entity_id += 1

        all_chains.append(
            (
                chain_name[:4].encode("ascii"),
                const.MOL_NONPOLYMER,
                entity_id,
                sym_id,
                chain_idx,
                res_offset,
                1,  # res_num (one residue per ligand)
                atom_offset,
                n_atoms,
            )
        )

        atom_offset += n_atoms
        res_offset += 1
        chain_idx += 1

    atoms_arr = np.array(all_atoms, dtype=ATOM_DTYPE) if all_atoms else np.empty(0, dtype=ATOM_DTYPE)
    bonds_arr = np.array(all_bonds, dtype=BOND_DTYPE) if all_bonds else np.empty(0, dtype=BOND_DTYPE)
    residues_arr = np.array(all_residues, dtype=RESIDUE_DTYPE) if all_residues else np.empty(0, dtype=RESIDUE_DTYPE)
    chains_arr = np.array(all_chains, dtype=CHAIN_DTYPE) if all_chains else np.empty(0, dtype=CHAIN_DTYPE)

    return Structure(
        atoms=atoms_arr,
        bonds=bonds_arr,
        residues=residues_arr,
        chains=chains_arr,
    )


def _sequence_to_structure(sequence: str, mol_type: int) -> Structure:
    """Build a single-chain Structure from a 1-letter sequence (no mmCIF needed)."""
    return _build_complex_structure([(sequence, mol_type, "A")])


def _sequence_to_msa(sequence: str, mol_type: int) -> MSA:
    """Build a single-sequence MSA from a 1-letter sequence."""
    N = len(sequence)
    sequences = np.zeros((1, N), dtype=np.int8)
    mask = np.ones((1, N), dtype=bool)
    deletion_matrix = np.zeros((1, N), dtype=np.int8)

    if mol_type == const.MOL_PROTEIN:
        lookup = const.PROTEIN_MSA_ID
        unk = 20
    elif mol_type == const.MOL_RNA:
        lookup = const.RNA_MSA_ID
        unk = const.MSA_UNK_NUCLEIC
    elif mol_type == const.MOL_DNA:
        lookup = const.DNA_MSA_ID
        unk = const.MSA_UNK_NUCLEIC
    else:
        lookup = const.PROTEIN_MSA_ID
        unk = 20

    for i, ch in enumerate(sequence.upper()):
        sequences[0, i] = lookup.get(ch, unk)

    return MSA(sequences=sequences, mask=mask, deletion_matrix=deletion_matrix)


from deepfold3.data.feature.ref_atoms import decode_atom_name_chars as _atom_name_from_chars  # noqa: E402, F401


def _msa_indices_to_sequence(encoded: np.ndarray, mol_type: int) -> str:
    """Convert MSA-index row back to a 1-letter sequence string.

    Gap positions (index 21) are skipped — this reconstructs the query
    match-state sequence from the first row of a parsed A3M.
    """
    if mol_type == const.MOL_PROTEIN:
        rev = {v: k for k, v in const.PROTEIN_MSA_ID.items() if k not in ("-", "B", "J", "O", "U", "Z")}
        rev[20] = "X"
        rev[21] = "-"
    elif mol_type == const.MOL_RNA:
        rev = {v: k for k, v in const.RNA_MSA_ID.items()}
        rev[const.MSA_UNK_NUCLEIC] = "N"
    elif mol_type == const.MOL_DNA:
        rev = {v: k for k, v in const.DNA_MSA_ID.items()}
        rev[const.MSA_UNK_NUCLEIC] = "N"
    else:
        rev = {v: k for k, v in const.PROTEIN_MSA_ID.items() if k not in ("-", "B", "J", "O", "U", "Z")}
        rev[20] = "X"
        rev[21] = "-"

    return "".join(rev.get(int(idx), "X") for idx in encoded if int(idx) != 21)
