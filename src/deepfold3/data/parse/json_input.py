"""JSON input parser for structure prediction jobs.

Supports four JSON dialects:

1. **AlphaFold 3** (``"dialect": "alphafold3"``) — Faithful reproduction of
   the AF3 input format.  Uses ``protein``, ``dna``, ``rna``, ``ligand``
   keys with ``id`` fields.  MSA via ``unpairedMsa``/``unpairedMsaPath``,
   ``pairedMsa``/``pairedMsaPath``.  Templates via inline ``templates``
   list with ``mmcif``/``mmcifPath`` + ``queryIndices``/``templateIndices``.
   Reference: ``references/alphafold3/src/alphafold3/common/folding_input.py``

2. **AlphaFold Server** (``"dialect": "alphafoldserver"``) — Server format
   with ``proteinChain``, ``rnaSequence``, ``dnaSequence``, ``ion`` keys.
   No MSA/template fields — MSA search and template lookup are run
   automatically via ColabFold MMseqs2 server.

3. **Protenix** (auto-detected) — ``proteinChain``, ``dnaSequence``,
   ``rnaSequence`` keys.  MSA via ``msa.precomputed_msa_dir`` (containing
   ``pairing.a3m`` / ``non_pairing.a3m``) or ``pairedMsaPath``/
   ``unpairedMsaPath``.  Templates via ``templatesPath`` (``.json`` or
   ``.hhr``).  Reference: ``references/Protenix/``

4. **DeepFold3** (``"dialect": "deepfold3"``) — Our superset dialect
   combining AF3 and Protenix features.  Accepts both ``protein``/``dna``/
   ``rna`` (AF3) and ``proteinChain``/``dnaSequence``/``rnaSequence``
   (Protenix) keys.  All MSA/template formats supported.

Format is auto-detected per entry.  All dialects produce the same
``JobInput`` output.

All file I/O (reading A3M files, downloading template CIFs) happens at
parse time.  The resulting dataclasses hold parsed objects (MSA, template
hits) rather than file paths, so downstream consumers need no I/O.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import field
from pathlib import Path

import numpy as np

from deepfold3.data import const
from deepfold3.data.msa.pairing import MSA_PAIRING_MODES, pair_by_species, species_fraction, species_ids
from deepfold3.data.parse.a3m import parse_a3m, parse_a3m_with_headers
from deepfold3.data.parse.templates import TemplateHitInput, fetch_pdb_mmcif, release_date_from_mmcif_text
from deepfold3.data.types import MSA

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ProteinChainInput:
    """A single protein chain specification from JSON input.

    Attributes:
        sequence: 1-letter amino acid sequence.
        count: Number of copies of this chain (must be 1 for now).
        paired_msa: Parsed paired MSA object (optional).
        unpaired_msa: Parsed unpaired MSA object (optional).
        template_hits: Parsed template hits ready for feature extraction.
        modifications: Post-translational modifications as ``(ccd_code, position)``
            tuples where ``position`` is 1-based within the chain. The residue
            at ``position`` should be assembled using ``ccd_code`` (e.g. ``"SEP"``)
            instead of the canonical amino acid.
        description: Optional textual description of this chain.
        chain_id: Chain name from the input JSON (AF3-style ``id``), carried so
            the output mmCIF / confidences can be labelled with the ids the
            caller asked for instead of synthetic A, B, C, … ``None`` for
            dialects that do not name chains (AlphaFold-Server, Protenix), which
            fall back to the synthetic naming.
        paired_msa_species: UniProt species id of each ``paired_msa`` row
            (``""`` where the header has none), used by species pairing.
            Cleared once :func:`parse_json_input` has resolved the pairing.
    """

    sequence: str
    count: int
    paired_msa: MSA | None = None
    unpaired_msa: MSA | None = None
    template_hits: list[TemplateHitInput] = field(default_factory=list)
    modifications: list[tuple[str, int]] = field(default_factory=list)
    description: str | None = None
    chain_id: str | None = None
    paired_msa_species: np.ndarray | None = None


@dataclasses.dataclass
class NucleotideChainInput:
    """A single DNA or RNA chain specification from JSON input.

    Attributes:
        sequence: 1-letter nucleotide sequence (ACGT for DNA, ACGU for RNA).
        count: Number of copies of this chain (must be 1 for now).
        mol_type: const.MOL_DNA or const.MOL_RNA.
        paired_msa: Parsed paired MSA object (optional).
        unpaired_msa: Parsed unpaired MSA object (optional).
        modifications: Modified-base substitutions as ``(ccd_code, position)``
            tuples where ``position`` is 1-based within the chain (e.g.
            ``("5MC", 4)`` for 5-methylcytosine at position 4).
        description: Optional textual description of this chain.
        chain_id: Chain name from the input JSON; see ``ProteinChainInput``.
    """

    sequence: str
    count: int
    mol_type: int
    paired_msa: MSA | None = None
    unpaired_msa: MSA | None = None
    modifications: list[tuple[str, int]] = field(default_factory=list)
    description: str | None = None
    chain_id: str | None = None


@dataclasses.dataclass
class LigandChainInput:
    """A single ligand chain specification from JSON input.

    Exactly one of ``ccd_ids`` or ``smiles`` must be provided.

    Attributes:
        ccd_ids: List of CCD component codes (e.g. ["ATP"]).
        smiles: SMILES string for a custom molecule.
        count: Number of copies of this ligand.
        description: Optional textual description of this ligand.
        chain_id: Chain name from the input JSON; see ``ProteinChainInput``.
    """

    ccd_ids: list[str] | None = None
    smiles: str | None = None
    count: int = 1
    description: str | None = None
    chain_id: str | None = None


BondedAtom = tuple[str, int, str]  # (chain_id, residue_index_1based, atom_name)
BondedAtomPair = tuple[BondedAtom, BondedAtom]


@dataclasses.dataclass
class JobInput:
    """A single prediction job from JSON input.

    Attributes:
        name: Job name / identifier.
        protein_chains: List of protein chain inputs.
        dna_chains: List of DNA chain inputs.
        rna_chains: List of RNA chain inputs.
        ligand_chains: List of ligand chain inputs.
        model_seeds: List of random seeds for diffusion sampling.
        bonded_atom_pairs: Explicit covalent bonds between atoms, as pairs of
            ``(chain_id, residue_index_1based, atom_name)`` tuples. Used for
            inter-entity links like Cys–Cys disulfides, glycan attachments,
            and metal coordination.
        user_ccd: Inline mmCIF text defining custom CCD component(s) for
            ligands not in the standard Chemical Component Dictionary. May
            contain multiple ``data_<NAME>`` blocks.
        max_template_date: ISO ``YYYY-MM-DD`` cutoff; template hits with a
            PDB initial release date strictly later than this are dropped
            during post-parse filtering. ``None`` disables the filter.
            Mirrors AlphaFold 3's ``--max_template_date`` CLI flag (set
            from the top-level ``maxTemplateDate`` JSON field).
        description: Optional job description (written as _struct.title in mmCIF).
        msa_pairing: How the paired MSAs were resolved: ``"precomputed"``
            (row-aligned as given) or ``"species"`` (paired by species at parse
            time; the featurizer then drops unpaired rows already in the paired
            block). Set by :func:`parse_json_input`.
    """

    name: str
    protein_chains: list[ProteinChainInput]
    dna_chains: list[NucleotideChainInput]
    rna_chains: list[NucleotideChainInput]
    ligand_chains: list[LigandChainInput] = field(default_factory=list)
    model_seeds: list[int] = field(default_factory=lambda: [42])
    bonded_atom_pairs: list[BondedAtomPair] = field(default_factory=list)
    user_ccd: str | None = None
    max_template_date: str | None = None
    description: str | None = None
    msa_pairing: str = "precomputed"


_DNA_KEYS = ("dnaSequence", "dnaChain")
_RNA_KEYS = ("rnaSequence", "rnaChain")


def parse_json_input(
    json_path: str | Path,
    max_templates: int = 4,
    template_cache_dir: str | Path | None = None,
    pdb_mmcif_dir: str | Path | None = None,
    msa_pairing: str = "auto",
) -> list[JobInput]:
    """Parse a JSON input file (auto-detects dialect).

    All file I/O (reading A3M files, downloading template CIFs) happens
    here.  The returned ``JobInput`` objects hold fully parsed MSA and
    template hit objects.

    Args:
        json_path: Path to the JSON file.
        max_templates: Maximum number of template hits to load per chain.
        template_cache_dir: Directory to cache downloaded template mmCIF files.
        pdb_mmcif_dir: Optional local PDB mmCIF mirror directory
            (layout: ``{dir}/{mid2}/{id}.cif.gz``).
        msa_pairing: ``"precomputed"`` (paired MSAs are row-aligned, e.g.
            ColabFold ``pairaln`` output), ``"species"`` (paired MSAs are raw
            per-chain UniProt hits, paired by species as AF3/OF3 do) or
            ``"auto"`` (see :func:`resolve_msa_pairing_mode`).

    Returns:
        List of JobInput objects, one per job entry.

    Raises:
        ValueError: If the JSON contains unsupported entity types or invalid
            chain counts, or ``msa_pairing`` is not a known mode.
    """
    if msa_pairing not in MSA_PAIRING_MODES:
        raise ValueError(f"msa_pairing must be one of {MSA_PAIRING_MODES}, got {msa_pairing!r}")
    json_path = Path(json_path)
    base_dir = json_path.parent

    cache_dir = Path(template_cache_dir) if template_cache_dir is not None else None
    pdb_dir = Path(pdb_mmcif_dir) if pdb_mmcif_dir is not None else None

    with open(json_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    jobs: list[JobInput] = []
    for entry in data:
        dialect = _detect_dialect(entry)
        if dialect == "alphafold3":
            job = _parse_af3_entry(entry, base_dir, max_templates, cache_dir, pdb_dir)
        elif dialect == "alphafoldserver":
            job = _parse_alphafoldserver_entry(entry, base_dir, max_templates, cache_dir, pdb_dir)
        elif dialect == "protenix":
            job = _parse_protenix_entry(entry, base_dir, max_templates, cache_dir, pdb_dir)
        else:  # deepfold3
            job = _parse_deepfold3_entry(entry, base_dir, max_templates, cache_dir, pdb_dir)
        job.max_template_date = _parse_max_template_date(entry.get("maxTemplateDate"))
        _filter_template_hits_by_date(job)
        _apply_msa_pairing(job, msa_pairing)
        jobs.append(job)

    return jobs


# ---------------------------------------------------------------------------
# Dialect detection
# ---------------------------------------------------------------------------


def _detect_dialect(entry: dict) -> str:
    """Detect the JSON dialect.

    Returns one of: ``"alphafold3"``, ``"alphafoldserver"``, ``"protenix"``,
    ``"deepfold3"``.

    Detection rules:
    - Explicit ``"dialect"`` field → use as-is (if recognized)
    - No ``dialect`` + bare ``protein``/``dna``/``rna`` keys → ``"alphafold3"``
    - No ``dialect`` + ``proteinChain``/etc with MSA fields → ``"protenix"``
    - No ``dialect`` + ``proteinChain``/etc without MSA → ``"alphafoldserver"``
    - Fallback → ``"deepfold3"``
    """
    dialect = entry.get("dialect")
    if dialect in ("alphafold3", "alphafoldserver", "protenix", "deepfold3"):
        return dialect

    # Check for AF3-style bare keys (protein/dna/rna with id)
    for seq_item in entry.get("sequences", []):
        if any(k in seq_item for k in ("protein", "dna", "rna")):
            if "ligand" in seq_item:
                continue
            return "alphafold3"
        if "ligand" in seq_item:
            lig = seq_item["ligand"]
            if isinstance(lig, dict) and ("ccdCodes" in lig or "id" in lig):
                return "alphafold3"

    # Check for Protenix/server-style keys
    has_msa_fields = False
    for seq_item in entry.get("sequences", []):
        if any(k in seq_item for k in ("proteinChain", "rnaSequence", "dnaSequence", "ion")):
            for key in ("proteinChain", "rnaSequence", "dnaSequence"):
                if key in seq_item:
                    chain = seq_item[key]
                    if any(f in chain for f in ("msa", "pairedMsaPath", "unpairedMsaPath", "templatesPath")):
                        has_msa_fields = True
            if has_msa_fields:
                return "protenix"
            return "alphafoldserver"

    return "deepfold3"


# ---------------------------------------------------------------------------
# AlphaFold 3 dialect parser (faithful to references/alphafold3)
# ---------------------------------------------------------------------------


def _parse_af3_entry(
    entry: dict, base_dir: Path, max_templates: int, cache_dir: Path | None, pdb_mmcif_dir: Path | None = None
) -> JobInput:
    """Parse a single AlphaFold 3-format job entry.

    Faithful to ``references/alphafold3/src/alphafold3/common/folding_input.py``.
    Uses ``protein``, ``dna``, ``rna``, ``ligand`` keys with ``id`` fields.
    MSA via ``unpairedMsa``/``unpairedMsaPath``, ``pairedMsa``/``pairedMsaPath``.
    Templates via inline ``templates`` list or file-based ``templatesPath``.
    """
    name = entry.get("name", "unnamed")
    model_seeds = entry.get("modelSeeds", [42])

    protein_chains: list[ProteinChainInput] = []
    dna_chains: list[NucleotideChainInput] = []
    rna_chains: list[NucleotideChainInput] = []
    ligand_chains: list[LigandChainInput] = []

    for seq_item in entry.get("sequences", []):
        if "protein" in seq_item:
            p = seq_item["protein"]
            ids = _ensure_id_list(p.get("id", []))
            seq = p["sequence"]
            unpaired = _resolve_msa(base_dir, const.MOL_PROTEIN, p.get("unpairedMsaPath"), p.get("unpairedMsa"))
            paired, paired_species = _resolve_paired_msa(base_dir, p.get("pairedMsaPath"), p.get("pairedMsa"))
            # Templates: inline list or file path
            tmpl_hits = _resolve_af3_templates(base_dir, p.get("templates"), max_templates, cache_dir, pdb_mmcif_dir)
            if not tmpl_hits:
                tmpl_hits = _resolve_template_hits(
                    base_dir, p.get("templatesPath"), max_templates, cache_dir, pdb_mmcif_dir
                )
            mods = _parse_modifications(p.get("modifications"))
            desc = p.get("description")
            for cid in ids or [None]:
                protein_chains.append(
                    ProteinChainInput(
                        chain_id=cid,
                        sequence=seq,
                        count=1,
                        paired_msa=paired,
                        paired_msa_species=paired_species,
                        unpaired_msa=unpaired,
                        template_hits=tmpl_hits,
                        modifications=list(mods),
                        description=desc,
                    )
                )
        elif "dna" in seq_item:
            d = seq_item["dna"]
            ids = _ensure_id_list(d.get("id", []))
            seq = d["sequence"]
            unpaired = _resolve_msa(base_dir, const.MOL_DNA, d.get("unpairedMsaPath"), d.get("unpairedMsa"))
            paired = _resolve_msa(base_dir, const.MOL_DNA, d.get("pairedMsaPath"), d.get("pairedMsa"))
            mods = _parse_modifications(d.get("modifications"))
            desc = d.get("description")
            for cid in ids or [None]:
                dna_chains.append(
                    NucleotideChainInput(
                        chain_id=cid,
                        sequence=seq,
                        count=1,
                        mol_type=const.MOL_DNA,
                        paired_msa=paired,
                        unpaired_msa=unpaired,
                        modifications=list(mods),
                        description=desc,
                    )
                )
        elif "rna" in seq_item:
            r = seq_item["rna"]
            ids = _ensure_id_list(r.get("id", []))
            seq = r["sequence"]
            unpaired = _resolve_msa(base_dir, const.MOL_RNA, r.get("unpairedMsaPath"), r.get("unpairedMsa"))
            paired = _resolve_msa(base_dir, const.MOL_RNA, r.get("pairedMsaPath"), r.get("pairedMsa"))
            mods = _parse_modifications(r.get("modifications"))
            desc = r.get("description")
            for cid in ids or [None]:
                rna_chains.append(
                    NucleotideChainInput(
                        chain_id=cid,
                        sequence=seq,
                        count=1,
                        mol_type=const.MOL_RNA,
                        paired_msa=paired,
                        unpaired_msa=unpaired,
                        modifications=list(mods),
                        description=desc,
                    )
                )
        elif "ligand" in seq_item:
            lig = seq_item["ligand"]
            ids = _ensure_id_list(lig.get("id", []))
            ccd_codes = lig.get("ccdCodes")
            smiles_str = lig.get("smiles")
            desc = lig.get("description")
            for cid in ids or [None]:
                if ccd_codes:
                    ligand_chains.append(
                        LigandChainInput(chain_id=cid, ccd_ids=list(ccd_codes), count=1, description=desc)
                    )
                elif smiles_str:
                    ligand_chains.append(LigandChainInput(chain_id=cid, smiles=smiles_str, count=1, description=desc))

    if not protein_chains and not dna_chains and not rna_chains and not ligand_chains:
        raise ValueError(f"No chains found in job {name!r}")

    bonded_atom_pairs = _parse_bonded_atom_pairs(entry.get("bondedAtomPairs"))
    user_ccd = _resolve_user_ccd(base_dir, entry.get("userCCD"), entry.get("userCCDPath"))

    return JobInput(
        name=name,
        protein_chains=protein_chains,
        dna_chains=dna_chains,
        rna_chains=rna_chains,
        ligand_chains=ligand_chains,
        model_seeds=model_seeds,
        bonded_atom_pairs=bonded_atom_pairs,
        user_ccd=user_ccd,
        description=entry.get("description"),
    )


# ---------------------------------------------------------------------------
# AlphaFold Server dialect parser
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _MsaSearchResult:
    """Cached MSA + template search result for a single sequence."""

    paired_msa: MSA | None = None
    unpaired_msa: MSA | None = None
    template_hits: list[TemplateHitInput] = field(default_factory=list)


def _run_msa_search(
    protein_seqs: list[str],
    max_templates: int = 4,
    cache_dir: Path | None = None,
    pdb_mmcif_dir: Path | None = None,
) -> dict[str, _MsaSearchResult]:
    """Run ColabFold MMseqs2 server MSA search + template lookup for protein sequences.

    Args:
        protein_seqs: List of unique protein sequences.
        max_templates: Maximum template hits per chain.
        cache_dir: Cache directory for MSA server results and template CIFs.
        pdb_mmcif_dir: Optional local PDB mmCIF mirror directory.

    Returns:
        Dict mapping sequence -> _MsaSearchResult with parsed MSAs and template hits.
    """
    if not protein_seqs:
        return {}

    from deepfold3.data.msa.mmseqs2 import parse_m8, search_server

    use_pairing = len(protein_seqs) > 1
    logger.info(
        "Running MMseqs2 MSA search for %d protein sequence(s) (pairing=%s)",
        len(protein_seqs),
        use_pairing,
    )
    msa_result = search_server(
        protein_seqs,
        use_pairing=use_pairing,
        cache_dir=cache_dir,
    )

    results: dict[str, _MsaSearchResult] = {}
    for i, seq in enumerate(protein_seqs):
        unpaired_text = msa_result.unpaired.get(i, "")
        paired_text = msa_result.paired.get(i, "")
        unpaired_msa = parse_a3m(unpaired_text, const.MOL_PROTEIN) if unpaired_text.strip() else None
        paired_msa = parse_a3m(paired_text, const.MOL_PROTEIN) if paired_text.strip() else None

        # Template search from m8 hits
        tmpl_hits: list[TemplateHitInput] = []
        if msa_result.m8_text:
            m8_hits = parse_m8(msa_result.m8_text)
            chain_m8 = m8_hits.get(i, [])
            for hit in chain_m8[:max_templates]:
                try:
                    mmcif_text = fetch_pdb_mmcif(hit.pdb_id, cache_dir=cache_dir, pdb_mmcif_dir=pdb_mmcif_dir)
                    query_indices = list(range(hit.qstart - 1, hit.qend))
                    template_indices = list(range(hit.tstart - 1, hit.tend))
                    min_len = min(len(query_indices), len(template_indices))
                    tmpl_hits.append(
                        TemplateHitInput(
                            mmcif_text=mmcif_text,
                            query_indices=query_indices[:min_len],
                            template_indices=template_indices[:min_len],
                            chain_id=hit.chain_id,
                        )
                    )
                except Exception:
                    logger.warning("Failed to fetch template %s_%s", hit.pdb_id, hit.chain_id, exc_info=True)

        if tmpl_hits:
            logger.info("Chain %d (%s...): %d template(s)", i, seq[:20], len(tmpl_hits))

        results[seq] = _MsaSearchResult(
            paired_msa=paired_msa,
            unpaired_msa=unpaired_msa,
            template_hits=tmpl_hits,
        )

    return results


def _parse_alphafoldserver_entry(
    entry: dict,
    base_dir: Path,
    max_templates: int = 4,
    cache_dir: Path | None = None,
    pdb_mmcif_dir: Path | None = None,
) -> JobInput:
    """Parse a single AlphaFold Server format job entry.

    Faithful to ``references/alphafold3/src/alphafold3/common/folding_input.py``
    ``Input.from_alphafoldserver_fold_job``.

    MSA/templates are searched automatically via ColabFold MMseqs2 server.
    """
    import random as _random

    name = entry.get("name", "unnamed")
    raw_seeds = entry.get("modelSeeds", [])
    model_seeds = [int(s) for s in raw_seeds] if raw_seeds else [_random.randint(0, 2**32 - 1)]

    # First pass: collect chain specs (sequence, count, type, modifications) in order
    chain_specs: list[tuple[str, str, int, list[tuple[str, int]]]] = []
    ligand_chains: list[LigandChainInput] = []

    for seq_item in entry.get("sequences", []):
        if "proteinChain" in seq_item:
            pc = seq_item["proteinChain"]
            chain_specs.append(
                (pc["sequence"], "protein", pc.get("count", 1), _parse_modifications(pc.get("modifications")))
            )
        elif "rnaSequence" in seq_item:
            rc = seq_item["rnaSequence"]
            chain_specs.append(
                (rc["sequence"], "rna", rc.get("count", 1), _parse_modifications(rc.get("modifications")))
            )
        elif "dnaSequence" in seq_item:
            dc = seq_item["dnaSequence"]
            chain_specs.append(
                (dc["sequence"], "dna", dc.get("count", 1), _parse_modifications(dc.get("modifications")))
            )
        elif "ion" in seq_item:
            ion = seq_item["ion"]
            ion_code = ion.get("ion", "")
            count = ion.get("count", 1)
            for _ in range(count):
                ligand_chains.append(LigandChainInput(ccd_ids=[ion_code], count=1))
        elif "ligand" in seq_item:
            lig = seq_item["ligand"]
            ligand_str = lig.get("ligand", "")
            count = lig.get("count", 1)
            ccd_code = ligand_str.removeprefix("CCD_") if isinstance(ligand_str, str) else ""
            if ccd_code:
                for _ in range(count):
                    ligand_chains.append(LigandChainInput(ccd_ids=[ccd_code], count=1))

    # Expand chain_specs by count, tracking unique protein sequences for MSA
    expanded: list[tuple[str, str, int, list[tuple[str, int]]]] = []  # (sequence, entity_type, spec_idx, mods)
    for spec_idx, (seq, etype, count, mods) in enumerate(chain_specs):
        for _ in range(count):
            expanded.append((seq, etype, spec_idx, mods))

    if not expanded and not ligand_chains:
        raise ValueError(f"No chains found in job {name!r}")

    # Run MSA search for all unique protein sequences
    protein_seqs = list(dict.fromkeys(seq for seq, etype, _, _ in expanded if etype == "protein"))
    msa_cache = _run_msa_search(protein_seqs, max_templates, cache_dir, pdb_mmcif_dir)

    # Build chain inputs
    protein_chains: list[ProteinChainInput] = []
    dna_chains: list[NucleotideChainInput] = []
    rna_chains: list[NucleotideChainInput] = []

    for seq, etype, _, mods in expanded:
        if etype == "protein":
            cached = msa_cache.get(seq)
            protein_chains.append(
                ProteinChainInput(
                    sequence=seq,
                    count=1,
                    paired_msa=cached.paired_msa if cached else None,
                    unpaired_msa=cached.unpaired_msa if cached else None,
                    template_hits=cached.template_hits if cached else [],
                    modifications=list(mods),
                )
            )
        elif etype == "rna":
            rna_chains.append(
                NucleotideChainInput(sequence=seq, count=1, mol_type=const.MOL_RNA, modifications=list(mods))
            )
        elif etype == "dna":
            dna_chains.append(
                NucleotideChainInput(sequence=seq, count=1, mol_type=const.MOL_DNA, modifications=list(mods))
            )

    bonded_atom_pairs = _parse_bonded_atom_pairs(entry.get("bondedAtomPairs"))
    user_ccd = _resolve_user_ccd(base_dir, entry.get("userCCD"), entry.get("userCCDPath"))

    return JobInput(
        name=name,
        protein_chains=protein_chains,
        dna_chains=dna_chains,
        rna_chains=rna_chains,
        ligand_chains=ligand_chains,
        model_seeds=model_seeds,
        bonded_atom_pairs=bonded_atom_pairs,
        user_ccd=user_ccd,
    )


# ---------------------------------------------------------------------------
# Protenix dialect parser (faithful to references/Protenix)
# ---------------------------------------------------------------------------


def _parse_protenix_entry(
    entry: dict, base_dir: Path, max_templates: int, cache_dir: Path | None, pdb_mmcif_dir: Path | None = None
) -> JobInput:
    """Parse a single Protenix-format job entry.

    Faithful to ``references/Protenix/`` format.  Supports:
    - ``proteinChain``, ``dnaSequence``/``dnaChain``, ``rnaSequence``/``rnaChain``
    - ``msa.precomputed_msa_dir`` (directory with ``pairing.a3m`` / ``non_pairing.a3m``)
    - ``pairedMsaPath``/``unpairedMsaPath`` (direct A3M file paths)
    - ``templatesPath`` (JSON or HHR file)
    - ``count > 1`` for chain duplication
    - ``ligand`` with ``CCD_`` prefix
    """
    name = entry.get("name", "unnamed")
    raw_seeds = entry.get("modelSeeds", [])
    model_seeds = [int(s) for s in raw_seeds] if raw_seeds else [42]

    protein_chains: list[ProteinChainInput] = []
    dna_chains: list[NucleotideChainInput] = []
    rna_chains: list[NucleotideChainInput] = []
    ligand_chains: list[LigandChainInput] = []

    for seq_item in entry.get("sequences", []):
        if "proteinChain" in seq_item:
            pc = seq_item["proteinChain"]
            count = pc.get("count", 1)
            # Resolve MSA: try pairedMsaPath/unpairedMsaPath first, then msa.precomputed_msa_dir
            paired_path = pc.get("pairedMsaPath")
            unpaired_path = pc.get("unpairedMsaPath")
            if not paired_path and not unpaired_path and pc.get("msa"):
                msa_dir = pc["msa"].get("precomputed_msa_dir")
                if msa_dir:
                    resolved_dir = _resolve_path(base_dir, msa_dir)
                    if resolved_dir and resolved_dir.is_dir():
                        pairing_file = resolved_dir / "pairing.a3m"
                        non_pairing_file = resolved_dir / "non_pairing.a3m"
                        if pairing_file.exists():
                            paired_path = str(pairing_file)
                        if non_pairing_file.exists():
                            unpaired_path = str(non_pairing_file)
            mods = _parse_modifications(pc.get("modifications"))
            paired, paired_species = _resolve_paired_msa(base_dir, paired_path, None)
            for _ in range(count):
                protein_chains.append(
                    ProteinChainInput(
                        sequence=pc["sequence"],
                        count=1,
                        paired_msa=paired,
                        paired_msa_species=paired_species,
                        unpaired_msa=_resolve_msa(base_dir, const.MOL_PROTEIN, unpaired_path, None),
                        template_hits=_resolve_template_hits(
                            base_dir, pc.get("templatesPath"), max_templates, cache_dir, pdb_mmcif_dir
                        ),
                        modifications=list(mods),
                        description=pc.get("description"),
                    )
                )
        elif any(k in seq_item for k in _DNA_KEYS):
            key = next(k for k in _DNA_KEYS if k in seq_item)
            dc = seq_item[key]
            count = dc.get("count", 1)
            mods = _parse_modifications(dc.get("modifications"))
            for _ in range(count):
                dna_chains.append(
                    NucleotideChainInput(
                        sequence=dc["sequence"],
                        count=1,
                        mol_type=const.MOL_DNA,
                        paired_msa=_resolve_msa(base_dir, const.MOL_DNA, dc.get("pairedMsaPath"), None),
                        unpaired_msa=_resolve_msa(base_dir, const.MOL_DNA, dc.get("unpairedMsaPath"), None),
                        modifications=list(mods),
                        description=dc.get("description"),
                    )
                )
        elif any(k in seq_item for k in _RNA_KEYS):
            key = next(k for k in _RNA_KEYS if k in seq_item)
            rc = seq_item[key]
            count = rc.get("count", 1)
            # RNA also supports msa.precomputed_msa_dir or direct path
            unpaired_path = rc.get("unpairedMsaPath")
            if not unpaired_path and rc.get("msa"):
                msa_dir = rc["msa"].get("precomputed_msa_dir")
                if msa_dir:
                    resolved_dir = _resolve_path(base_dir, msa_dir)
                    if resolved_dir and resolved_dir.is_dir():
                        non_pairing_file = resolved_dir / "non_pairing.a3m"
                        if non_pairing_file.exists():
                            unpaired_path = str(non_pairing_file)
            mods = _parse_modifications(rc.get("modifications"))
            for _ in range(count):
                rna_chains.append(
                    NucleotideChainInput(
                        sequence=rc["sequence"],
                        count=1,
                        mol_type=const.MOL_RNA,
                        unpaired_msa=_resolve_msa(base_dir, const.MOL_RNA, unpaired_path, None),
                        modifications=list(mods),
                        description=rc.get("description"),
                    )
                )
        elif "ion" in seq_item:
            ion = seq_item["ion"]
            ion_code = ion.get("ion", "")
            count = ion.get("count", 1)
            for _ in range(count):
                ligand_chains.append(LigandChainInput(ccd_ids=[ion_code], count=1))
        elif "ligand" in seq_item:
            lig = seq_item["ligand"]
            ligand_str = lig.get("ligand", "")
            count = lig.get("count", 1)
            if isinstance(ligand_str, str) and ligand_str.startswith("CCD_"):
                ccd_code = ligand_str[4:]
                for _ in range(count):
                    ligand_chains.append(LigandChainInput(ccd_ids=[ccd_code], count=1))
            elif isinstance(ligand_str, str) and ligand_str:
                for _ in range(count):
                    ligand_chains.append(LigandChainInput(smiles=ligand_str, count=1))

    if not protein_chains and not dna_chains and not rna_chains and not ligand_chains:
        raise ValueError(f"No chains found in job {name!r}")

    return JobInput(
        name=name,
        protein_chains=protein_chains,
        dna_chains=dna_chains,
        rna_chains=rna_chains,
        ligand_chains=ligand_chains,
        model_seeds=model_seeds,
        bonded_atom_pairs=_parse_bonded_atom_pairs(entry.get("bondedAtomPairs")),
        user_ccd=_resolve_user_ccd(base_dir, entry.get("userCCD"), entry.get("userCCDPath")),
        description=entry.get("description"),
    )


# ---------------------------------------------------------------------------
# DeepFold3 dialect parser (superset of AF3 + Protenix)
# ---------------------------------------------------------------------------


def _parse_deepfold3_entry(
    entry: dict, base_dir: Path, max_templates: int, cache_dir: Path | None, pdb_mmcif_dir: Path | None = None
) -> JobInput:
    """Parse a single DeepFold3-format job entry.

    DeepFold3 format is our superset dialect that accepts both AF3-style keys
    (``protein``, ``dna``, ``rna``, ``ligand`` with ``id``) and Protenix-style
    keys (``proteinChain``, ``dnaSequence``, ``rnaSequence``).  Both
    underlying parsers honour ``modifications`` / ``bondedAtomPairs`` /
    ``userCCD`` so the dialect needs no additional overlay step.
    """
    has_af3_keys = False
    has_protenix_keys = False
    for seq_item in entry.get("sequences", []):
        if any(k in seq_item for k in ("protein", "dna", "rna")):
            has_af3_keys = True
        if any(k in seq_item for k in ("proteinChain", "dnaSequence", "dnaChain", "rnaSequence", "rnaChain")):
            has_protenix_keys = True

    if has_af3_keys and not has_protenix_keys:
        return _parse_af3_entry(entry, base_dir, max_templates, cache_dir, pdb_mmcif_dir)
    # Pure Protenix or mixed: protenix parser is the more permissive one.
    return _parse_protenix_entry(entry, base_dir, max_templates, cache_dir, pdb_mmcif_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_id_list(ids: str | list[str]) -> list[str]:
    """Normalize ``id`` field to a list of chain IDs."""
    if isinstance(ids, str):
        return [ids]
    return list(ids)


def _parse_modifications(raw: list | None) -> list[tuple[str, int]]:
    """Parse a ``modifications`` JSON list into ``(ccd_code, 1-based position)`` tuples.

    Accepts the AF3 schema ``[{"ptmType": "SEP", "ptmPosition": 3}, ...]``.
    Invalid entries are skipped with a warning.
    """
    if not raw:
        return []
    result: list[tuple[str, int]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            logger.warning("Skipping non-dict modification entry: %r", entry)
            continue
        ccd_code = entry.get("ptmType") or entry.get("modificationType")
        position = entry.get("ptmPosition")
        if position is None:
            position = entry.get("modificationPosition")
        if not ccd_code or position is None:
            logger.warning("Skipping modification missing ptmType/ptmPosition: %r", entry)
            continue
        try:
            result.append((str(ccd_code).strip().upper(), int(position)))
        except (TypeError, ValueError):
            logger.warning("Skipping modification with non-integer position: %r", entry)
    return result


def _parse_bonded_atom_pairs(raw: list | None) -> list[BondedAtomPair]:
    """Parse ``bondedAtomPairs`` JSON into a list of atom-pair tuples.

    Expects the AF3 schema::

        [[[chain_id, res_idx_1based, atom_name], [chain_id, res_idx_1based, atom_name]], ...]

    Invalid entries are skipped with a warning.
    """
    if not raw:
        return []
    pairs: list[BondedAtomPair] = []
    for entry in raw:
        if not isinstance(entry, list | tuple) or len(entry) != 2:
            logger.warning("Skipping bondedAtomPairs entry (expected 2-element list): %r", entry)
            continue
        a, b = entry
        if not (isinstance(a, list | tuple) and isinstance(b, list | tuple) and len(a) == 3 and len(b) == 3):
            logger.warning("Skipping bondedAtomPairs entry (atoms must be 3-element lists): %r", entry)
            continue
        try:
            pair: BondedAtomPair = (
                (str(a[0]), int(a[1]), str(a[2])),
                (str(b[0]), int(b[1]), str(b[2])),
            )
        except (TypeError, ValueError):
            logger.warning("Skipping malformed bondedAtomPairs entry: %r", entry)
            continue
        pairs.append(pair)
    return pairs


def _parse_max_template_date(raw) -> str | None:
    """Validate a ``maxTemplateDate`` JSON value as ``YYYY-MM-DD``.

    Returns ``None`` if the value is missing, empty, or not a parseable
    ISO 8601 date. Warns on malformed values.
    """
    if raw in (None, ""):
        return None
    import datetime as _dt

    try:
        return _dt.date.fromisoformat(str(raw).strip()).isoformat()
    except (TypeError, ValueError):
        logger.warning("Ignoring malformed maxTemplateDate %r (expected YYYY-MM-DD)", raw)
        return None


def _filter_template_hits_by_date(job: JobInput) -> None:
    """Drop template hits released strictly after ``job.max_template_date``.

    Hits with ``release_date is None`` are kept (we cannot verify them);
    callers relying on strict filtering should prefer template sources
    that populate ``release_date``. Mirrors AlphaFold 3's inference-time
    template cutoff behaviour.
    """
    cutoff = job.max_template_date
    if cutoff is None:
        return
    for chain in job.protein_chains:
        before = len(chain.template_hits)
        chain.template_hits = [h for h in chain.template_hits if h.release_date is None or h.release_date <= cutoff]
        dropped = before - len(chain.template_hits)
        if dropped:
            logger.info("Filtered %d template hit(s) past cutoff %s on chain %s", dropped, cutoff, chain.sequence[:20])


def _resolve_user_ccd(base_dir: Path, user_ccd: str | None, user_ccd_path: str | None) -> str | None:
    """Resolve ``userCCD`` (inline mmCIF text) / ``userCCDPath`` (file path) to mmCIF text.

    If both are set, ``userCCD`` wins and a warning is logged.
    """
    if user_ccd and user_ccd_path:
        logger.warning("Both userCCD and userCCDPath are set; using userCCD (inline text)")
        user_ccd_path = None
    if user_ccd:
        return user_ccd
    if user_ccd_path:
        resolved = _resolve_path(base_dir, user_ccd_path)
        if resolved is None or not resolved.exists():
            logger.warning("userCCDPath not found: %s", user_ccd_path)
            return None
        try:
            return _read_mmcif_file(resolved)
        except OSError as e:
            logger.warning("Failed to read userCCDPath %s: %s", resolved, e)
            return None
    return None


# ---------------------------------------------------------------------------
# MSA resolution
# ---------------------------------------------------------------------------


def _resolve_msa(base_dir: Path, mol_type: int, path_str: str | None, inline_str: str | None) -> MSA | None:
    """Resolve MSA from a file path or inline string into a parsed MSA object.

    Args:
        base_dir: Base directory for resolving relative paths.
        mol_type: Molecule type (MOL_PROTEIN, MOL_DNA, MOL_RNA).
        path_str: Path to an A3M file (relative or absolute), or None.
        inline_str: Inline A3M text, or None.

    Returns:
        Parsed MSA object, or None if neither path nor inline string is provided.
    """
    text = _read_msa_text(base_dir, path_str, inline_str)
    if text is None:
        return None
    msa = parse_a3m(text, mol_type)
    return msa if msa.sequences.shape[0] > 0 else None


def _resolve_paired_msa(
    base_dir: Path, path_str: str | None, inline_str: str | None
) -> tuple[MSA | None, np.ndarray | None]:
    """Resolve a protein paired MSA and the species id of each row.

    Same resolution as :func:`_resolve_msa`; the headers are kept as species ids
    so the job-level pass (:func:`_apply_msa_pairing`) can pair rows by species.

    Returns:
        ``(msa, species)``, or ``(None, None)`` when there is no MSA.
    """
    text = _read_msa_text(base_dir, path_str, inline_str)
    if text is None:
        return None, None
    msa, headers = parse_a3m_with_headers(text, const.MOL_PROTEIN)
    if msa.sequences.shape[0] == 0:
        return None, None
    return msa, species_ids(headers)


def _read_msa_text(base_dir: Path, path_str: str | None, inline_str: str | None) -> str | None:
    """A3M text from a path (preferred) or inline string; ``None`` if absent or unreadable."""
    if path_str is not None:
        resolved = _resolve_path(base_dir, path_str)
        if resolved is not None:
            try:
                return resolved.read_text()
            except (FileNotFoundError, OSError) as e:
                logger.warning("Failed to read MSA file %s: %s", resolved, e)
        return None
    if inline_str is not None and inline_str.strip():
        return inline_str
    return None


# ---------------------------------------------------------------------------
# MSA pairing (job-level pass)
# ---------------------------------------------------------------------------


def resolve_msa_pairing_mode(job: JobInput, mode: str) -> tuple[str, str]:
    """Pick ``precomputed`` or ``species`` for one job.

    ``auto`` selects ``species`` only when the paired MSAs are evidently raw
    per-chain UniProt hits: at least two distinct protein sequences carry a
    paired MSA with hits, every one of them has a UniProt species id on at least
    half of its non-query rows, and their row counts differ. Anything else —
    ColabFold/MMseqs2 ``pairaln`` output (``UniRef100_…`` / ``DUMMY`` headers,
    equal row counts), monomers, homomers — stays ``precomputed``, i.e. exactly
    the behaviour before this option existed. The JSON dialect is deliberately
    not consulted: FoldBench writes both kinds of input as ``"alphafold3"``.

    Args:
        job: Parsed job (paired MSAs and species ids already attached).
        mode: ``"auto"``, ``"precomputed"`` or ``"species"``.

    Returns:
        ``(resolved_mode, reason)``.

    Raises:
        ValueError: On an unknown mode.
    """
    if mode not in MSA_PAIRING_MODES:
        raise ValueError(f"msa_pairing must be one of {MSA_PAIRING_MODES}, got {mode!r}")
    if mode != "auto":
        return mode, "requested"
    reps = _paired_reps(job)
    if len(reps) < 2:
        return "precomputed", "fewer than two distinct protein chains with paired hits"
    fractions = [species_fraction(c.paired_msa_species) for c in reps.values()]
    depths = {c.paired_msa.sequences.shape[0] for c in reps.values()}
    if min(fractions) < 0.5:
        return "precomputed", "paired MSA headers carry no UniProt species ids"
    if len(depths) == 1:
        return "precomputed", "paired MSAs are already row-aligned (equal row counts)"
    return "species", "raw UniProt hits with per-chain row counts"


def _paired_reps(job: JobInput) -> dict[str, ProteinChainInput]:
    """One protein chain per distinct sequence whose paired MSA has hits beyond the query."""
    reps: dict[str, ProteinChainInput] = {}
    for c in job.protein_chains:
        if c.paired_msa is not None and c.paired_msa.sequences.shape[0] > 1:
            reps.setdefault(c.sequence, c)
    return reps


def _apply_msa_pairing(job: JobInput, mode: str) -> None:
    """Resolve the pairing mode for ``job`` and, for ``species``, pair rows in place.

    ``precomputed`` leaves every paired MSA untouched (row *i* of each chain is
    taken to be paired with row *i* of the others). ``species`` follows OF3
    ``MsaSampleProcessorInference.create_paired_msa``:

    - a monomer or homomer (one distinct protein sequence) is not paired, and its
      paired MSA is dropped, as OF3 does (its UniProt hits are not part of the
      main MSA either; DF3 ``format: of3pdb`` training does the same) — the
      featurizer then synthesises homomer paired rows from the unpaired MSA;
    - otherwise the distinct protein sequences with hits are paired with
      :func:`pair_by_species` and every copy of a sequence gets that sequence's
      row-aligned block; chains with no pairable hits get none (gaps in the
      paired rows);
    - non-protein paired MSAs are dropped (only proteins are paired).

    Logs one line per job with the mode and the resulting paired depth.
    """
    resolved, reason = resolve_msa_pairing_mode(job, mode)
    job.msa_pairing = resolved
    name = job.name
    if resolved == "precomputed":
        depths = [c.paired_msa.sequences.shape[0] for c in job.protein_chains if c.paired_msa is not None]
        if len(set(depths)) > 1:
            logger.warning(
                "[%s] MSA pairing: precomputed (%s) but paired row counts differ per chain %s; rows are zipped by "
                "index and shorter chains gap-padded. Use --msa-pairing species for raw per-chain UniProt hits.",
                name,
                reason,
                depths,
            )
        else:
            logger.info("[%s] MSA pairing: precomputed (%s); paired rows per chain %s", name, reason, depths or "none")
        for c in job.protein_chains:
            c.paired_msa_species = None
        return

    for c in (*job.dna_chains, *job.rna_chains):
        c.paired_msa = None
    reps = _paired_reps(job)
    before = {seq: c.paired_msa.sequences.shape[0] - 1 for seq, c in reps.items()}
    n_distinct = len({c.sequence for c in job.protein_chains})
    result = None
    if n_distinct >= 2 and len(reps) >= 2:
        seqs = list(reps)
        species = [
            c.paired_msa_species
            if c.paired_msa_species is not None
            else np.full(c.paired_msa.sequences.shape[0], "", dtype=object)  # no headers kept (MSA server path)
            for c in (reps[s] for s in seqs)
        ]
        result = pair_by_species([reps[s].paired_msa for s in seqs], species)

    paired_by_seq: dict[str, MSA] = {}
    if result is not None:
        paired_by_seq = dict(zip(seqs, result.msas, strict=True))
        logger.info(
            "[%s] MSA pairing: species (%s); %d paired rows (+query) from hits per sequence %s; "
            "rows with a hit per sequence %s",
            name,
            reason,
            result.n_rows,
            list(before.values()),
            result.rows_per_chain,
        )
    elif n_distinct < 2:
        logger.info("[%s] MSA pairing: species (%s); monomer/homomer, pairing skipped (AF3/OF3)", name, reason)
    else:
        logger.info(
            "[%s] MSA pairing: species (%s); no species shared by two chains (hits per sequence %s), no paired rows",
            name,
            reason,
            list(before.values()),
        )
    for c in job.protein_chains:
        c.paired_msa = paired_by_seq.get(c.sequence)
        c.paired_msa_species = None
        if c.paired_msa is None and c.unpaired_msa is None and before.get(c.sequence, 0) > 0:
            logger.warning(
                "[%s] MSA pairing: chain %s has paired hits but no unpaired MSA and was not paired; "
                "it falls back to a single-sequence MSA",
                name,
                c.chain_id or c.sequence[:20],
            )


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------


def _resolve_af3_templates(
    base_dir: Path,
    raw_templates: list[dict] | None,
    max_templates: int,
    cache_dir: Path | None,
    pdb_mmcif_dir: Path | None = None,
) -> list[TemplateHitInput]:
    """Resolve inline AF3-format templates.

    AF3 format stores templates as::

        [
            {
                "mmcif": "...",          // or "mmcifPath": "path/to/file.cif"
                "queryIndices": [0, 1, ...],
                "templateIndices": [5, 6, ...]
            }
        ]

    Returns:
        List of TemplateHitInput objects.
    """
    if raw_templates is None:
        return []

    result: list[TemplateHitInput] = []
    for tmpl in raw_templates[:max_templates]:
        mmcif_text = tmpl.get("mmcif")
        mmcif_path = tmpl.get("mmcifPath")

        if mmcif_text and mmcif_path:
            logger.warning("Template has both mmcif and mmcifPath; using mmcifPath")
            mmcif_text = None
        if mmcif_path:
            resolved = _resolve_path(base_dir, mmcif_path)
            if resolved is not None and resolved.exists():
                mmcif_text = _read_mmcif_file(resolved)
            else:
                logger.warning("Template mmcifPath not found: %s", mmcif_path)
                continue

        if not mmcif_text:
            logger.warning("Template has no mmcif content, skipping")
            continue

        query_indices = tmpl.get("queryIndices", [])
        template_indices = tmpl.get("templateIndices", [])
        if len(query_indices) != len(template_indices):
            logger.warning("Mismatched index lengths in template, skipping")
            continue

        result.append(
            TemplateHitInput(
                mmcif_text=mmcif_text,
                query_indices=query_indices,
                template_indices=template_indices,
                release_date=release_date_from_mmcif_text(mmcif_text),
            )
        )

    return result


def _resolve_template_hits(
    base_dir: Path,
    templates_path_str: str | None,
    max_templates: int,
    cache_dir: Path | None,
    pdb_mmcif_dir: Path | None = None,
) -> list[TemplateHitInput]:
    """Resolve template hits from a file path into parsed TemplateHitInput objects.

    Supports template path formats:
    - ``.json``: JSON template hits file with pre-computed alignments and local CIF files
    - ``.hhr``: HHsearch output — parses hits and downloads CIFs from RCSB
    - ``.a3m``: HMM search A3M output (treated as HHR for template search)

    Args:
        base_dir: Base directory for resolving relative paths.
        templates_path_str: Path to template file (JSON, HHR, or A3M), or None.
        max_templates: Maximum number of template hits to load.
        cache_dir: Directory to cache downloaded template mmCIF files.
        pdb_mmcif_dir: Optional local PDB mmCIF mirror directory.

    Returns:
        List of TemplateHitInput objects (empty if no templates specified).
    """
    if templates_path_str is None:
        return []

    resolved = _resolve_path(base_dir, templates_path_str)
    if resolved is None or not resolved.exists():
        logger.warning("Templates path not found: %s", templates_path_str)
        return []

    suffix = resolved.suffix.lower()
    if suffix == ".json":
        return _load_json_template_hits(resolved, max_templates)
    elif suffix in (".hhr", ".a3m"):
        return _load_hhr_template_hits(resolved, max_templates, cache_dir, pdb_mmcif_dir=pdb_mmcif_dir)
    else:
        logger.warning("Unsupported template file type: %s", suffix)
        return []


def _load_json_template_hits(json_path: Path, max_templates: int) -> list[TemplateHitInput]:
    """Load template hits from a JSON file with pre-computed alignments.

    Expected format::

        [
            {
                "name": "template_hit_0.cif",
                "queryIndices": [0, 1, 5, ...],
                "templateIndices": [10, 11, 15, ...],
                "chainId": "A"
            },
            ...
        ]

    ``chainId`` is optional — selects which chain to use from the template
    mmCIF.  Defaults to the first polymer chain if omitted.

    Returns:
        List of TemplateHitInput with mmCIF text and index pairs.
    """
    json_dir = json_path.parent

    try:
        with open(json_path) as f:
            hits = json.load(f)
    except Exception:
        logger.warning("Failed to read JSON template file %s", json_path, exc_info=True)
        return []

    if not isinstance(hits, list):
        logger.warning("JSON template hits file %s is not an array", json_path)
        return []

    result: list[TemplateHitInput] = []
    for hit in hits[:max_templates]:
        try:
            cif_name = hit["name"]
            query_indices = hit["queryIndices"]
            template_indices = hit["templateIndices"]

            if len(query_indices) != len(template_indices):
                logger.warning("Mismatched index lengths in template %s, skipping", cif_name)
                continue

            cif_path = json_dir / cif_name
            if not cif_path.exists():
                logger.warning("Template CIF not found: %s", cif_path)
                continue

            mmcif_text = _read_mmcif_file(cif_path)
            result.append(
                TemplateHitInput(
                    mmcif_text=mmcif_text,
                    query_indices=query_indices,
                    template_indices=template_indices,
                    chain_id=hit.get("chainId"),
                    release_date=hit.get("releaseDate") or release_date_from_mmcif_text(mmcif_text),
                )
            )
            logger.info("Loaded JSON template hit %s", cif_name)
        except Exception:
            logger.warning("Failed to process JSON template hit", exc_info=True)

    return result


def _load_hhr_template_hits(
    hhr_path: Path,
    max_templates: int,
    cache_dir: Path | None,
    pdb_mmcif_dir: Path | None = None,
) -> list[TemplateHitInput]:
    """Load template hits from an HHR file, downloading CIFs from RCSB.

    Returns:
        List of TemplateHitInput with mmCIF text and HHR hit objects.
    """
    from deepfold3.data.parse.hhr import parse_hhr

    try:
        hhr_text = hhr_path.read_text()
    except (FileNotFoundError, OSError) as e:
        logger.warning("Failed to read HHR file %s: %s", hhr_path, e)
        return []

    hits = parse_hhr(hhr_text)
    if not hits:
        logger.warning("No hits found in HHR file %s", hhr_path)
        return []

    result: list[TemplateHitInput] = []
    for hit in hits:
        if len(result) >= max_templates:
            break
        try:
            mmcif_text = fetch_pdb_mmcif(hit.pdb_id, cache_dir=cache_dir, pdb_mmcif_dir=pdb_mmcif_dir)
            result.append(
                TemplateHitInput(
                    mmcif_text=mmcif_text,
                    hhr_hit=hit,
                    chain_id=hit.chain_id,
                    release_date=release_date_from_mmcif_text(mmcif_text),
                )
            )
            logger.info(
                "Loaded HHR template %s_%s (e_value=%.1e, aligned_cols=%d)",
                hit.pdb_id,
                hit.chain_id,
                hit.e_value,
                hit.aligned_cols,
            )
        except Exception:
            logger.warning("Failed to download template %s_%s", hit.pdb_id, hit.chain_id, exc_info=True)

    return result


# ---------------------------------------------------------------------------
# Path resolution / file I/O helpers
# ---------------------------------------------------------------------------


def _read_mmcif_file(path: Path) -> str:
    """Read an mmCIF file, transparently decompressing ``.gz`` and ``.zst`` files."""
    s = str(path)
    if s.endswith(".zst"):
        import zstandard as zstd

        dctx = zstd.ZstdDecompressor()
        with open(path, "rb") as f:
            return dctx.decompress(f.read()).decode("utf-8")
    if s.endswith(".gz"):
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as gz:
            return gz.read()
    return path.read_text()


def _resolve_path(base_dir: Path, raw: str | None) -> Path | None:
    """Resolve a possibly-relative path against the JSON file's directory.

    Tries resolution in order:
    1. Relative to ``base_dir`` (the JSON file's parent directory)
    2. Relative to ancestor directories of ``base_dir`` (walks up)
    3. Relative to CWD
    4. Falls back to the JSON-relative path (for a clear error message)
    """
    if raw is None:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p
    # Try JSON directory and its ancestors
    candidate = base_dir
    for _ in range(6):
        resolved = candidate / p
        if resolved.exists():
            return resolved
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    # Try CWD
    if p.exists():
        return p.resolve()
    # Return the JSON-relative path (caller will get a clear error)
    return base_dir / p
