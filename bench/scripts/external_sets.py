"""Benchmark sets prepared outside FoldBench: 2026ARK-AB and PXM-2025.

The inputs for these sets come from a shared benchmark folder (default
``/store/deepfold/users/gunwoong/Benchmark``, override with ``DF3_GW_BENCH``)
that is only read, never written. Each set directory holds ``targets.csv``,
``chain_map.csv``, ``interfaces.csv`` and ``gt/<target>.cif``.

Two things differ from FoldBench and decide how the sets are scored:

1. The model input holds only the chains that appear in a scored interface,
   while the ground truth is the whole RCSB assembly. Comparing against the
   full assembly lets OST's chain mapper place a model chain on another copy
   and drags the global lDDT/TM down with chains the model never saw. So the
   reference is a copy of the ground truth that keeps only the chains listed in
   ``chain_map.csv`` (:func:`trimmed_reference`).
2. The model chains (A, B, C, ... in input order) are not the ground-truth
   chains, and symmetry copies are named ``B.1`` in the CSVs but ``B-2`` in the
   RCSB assembly file. The names are resolved with the rule the set's own
   ``scripts/verify_mapping.py`` uses (:func:`resolve_label`), and OST gets the
   model-to-reference correspondence explicitly (``-c ref:model``) instead of
   searching for one.

Trimmed references are cached under ``<bench>/data/ground_truths_trimmed/<category>/``
together with a JSON manifest (source path and sha256, kept chains), so a
cached file is rebuilt whenever its source changes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

GW_ROOT = Path(os.environ.get("DF3_GW_BENCH", "/store/deepfold/users/gunwoong/Benchmark"))

# eval category -> set directory under GW_ROOT
SETS = {
    "ark_antibody_antigen": "Ark/AbAg",
    "px_antibody_antigen": "PX/AbAg",
    "px_protein_protein": "PX/PPI",
}

_lock = threading.Lock()


@dataclass
class Target:
    name: str
    gt: Path
    chain_map: dict[str, str]  # model chain -> ground-truth label_asym_id
    interfaces: list[tuple[str, str]] = field(default_factory=list)  # ground-truth label pairs


def set_dir(category: str) -> Path:
    return GW_ROOT / SETS[category]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gt_chains(gt: Path) -> dict[str, str]:
    """label_asym_id -> auth_asym_id for every chain with atoms in ``gt``."""
    import gemmi

    block = gemmi.cif.read(str(gt)).sole_block()
    out: dict[str, str] = {}
    for label, auth in block.find("_atom_site.", ["label_asym_id", "auth_asym_id"]):
        out.setdefault(label, auth)
    return out


def resolve_label(name: str, chains: dict[str, str]) -> str:
    """Ground-truth label_asym_id for a chain name from chain_map/interfaces.csv.

    Same rule, in the same order, as ``resolve`` in the set's
    ``scripts/verify_mapping.py``, which passed for every target:

    - the name is a label in the file;
    - ``X.k`` (a PXMeter symmetry copy) is the RCSB copy label ``X-(k+1)``;
    - ``X`` with exactly one ``X-k`` and no ``X`` is that copy (RCSB names a
      chain placed by a non-identity operator ``X-k`` even when it is alone);
    - the name is the author chain of exactly one label;
    - ``X-k`` is the k-th label, in sorted order, whose author chain is ``X``.
    """
    if name in chains:
        return name
    if "." in name:
        base, k = name.rsplit(".", 1)
        alt = f"{base}-{int(k) + 1}"
        if alt in chains:
            return alt
    only = [lab for lab in chains if "-" in lab and lab.rsplit("-", 1)[0] == name]
    if len(only) == 1:
        return only[0]
    byauth = [lab for lab, auth in chains.items() if auth == name]
    if len(byauth) == 1:
        return byauth[0]
    if "-" in name:
        base, k = name.rsplit("-", 1)
        cand = sorted(lab for lab, auth in chains.items() if auth == base)
        if len(cand) >= int(k):
            return cand[int(k) - 1]
    raise KeyError(f"chain {name!r} not found in ground truth (labels: {sorted(chains)})")


@cache
def load(category: str) -> dict[str, Target]:
    """target name -> :class:`Target`, with every chain name resolved to a GT label."""
    root = set_dir(category)
    gt_file = {r["target"]: root / r["gt_file"] for r in csv.DictReader(open(root / "targets.csv"))}
    cmap: dict[str, dict[str, str]] = {}
    for r in csv.DictReader(open(root / "chain_map.csv")):
        cmap.setdefault(r["target"], {})[r["model_chain"]] = r["gt_chain"]
    ifaces: dict[str, list[tuple[str, str]]] = {}
    for r in csv.DictReader(open(root / "interfaces.csv")):
        ifaces.setdefault(r["target"], []).append((r["gt_chain_1"], r["gt_chain_2"]))

    out: dict[str, Target] = {}
    for name, gt in gt_file.items():
        chains = gt_chains(gt)
        m = {mdl: resolve_label(g, chains) for mdl, g in cmap[name].items()}
        if len(set(m.values())) != len(m):
            raise ValueError(f"{category}/{name}: two model chains resolve to one GT chain: {m}")
        kept = set(m.values())
        pairs = []
        for a, b in ifaces.get(name, []):
            la, lb = resolve_label(a, chains), resolve_label(b, chains)
            if la not in kept or lb not in kept:
                raise ValueError(f"{category}/{name}: interface {a}-{b} uses a chain the model lacks")
            pairs.append((la, lb))
        out[name] = Target(name=name, gt=gt, chain_map=m, interfaces=pairs)
    return out


def interfaces(category: str) -> dict[str, list[tuple[str, str]]]:
    return {name: t.interfaces for name, t in load(category).items()}


def chain_mapping_args(category: str, target: str) -> list[str]:
    """OST ``-c`` arguments: one ``reference:model`` pair per model chain."""
    m = load(category)[target].chain_map
    return ["-c", *(f"{ref}:{mdl}" for mdl, ref in sorted(m.items()))]


def trim_cif(src: Path, dst: Path, keep: set[str]) -> int:
    """Write ``src`` to ``dst`` keeping only ``_atom_site`` rows whose
    label_asym_id is in ``keep``. Every other category (``_entity_poly_seq``
    and the rest of the SEQRES OST reads) is copied unchanged. Returns the
    number of atoms kept."""
    import gemmi

    doc = gemmi.cif.read(str(src))
    block = doc.sole_block()
    cat = block.get_mmcif_category("_atom_site.", raw=True)
    rows = [i for i, lab in enumerate(cat["label_asym_id"]) if lab in keep]
    block.set_mmcif_category("_atom_site.", {k: [v[i] for i in rows] for k, v in cat.items()}, raw=True)
    tmp = dst.with_suffix(".cif.tmp")
    doc.write_file(str(tmp))
    tmp.replace(dst)
    return len(rows)


def trimmed_reference(category: str, target: str, cache_root: Path) -> Path:
    """Cached ground truth for ``target`` holding only the model's chains."""
    t = load(category)[target]
    keep = sorted(set(t.chain_map.values()))
    dst = cache_root / category / f"{target}.cif"
    manifest = dst.with_suffix(".json")
    want = {"source": str(t.gt), "source_sha256": sha256(t.gt), "kept_label_asym_ids": keep}
    with _lock:
        if dst.exists() and manifest.exists():
            try:
                if {k: v for k, v in json.load(open(manifest)).items() if k in want} == want:
                    return dst
            except json.JSONDecodeError:
                pass
        dst.parent.mkdir(parents=True, exist_ok=True)
        n = trim_cif(t.gt, dst, set(keep))
        json.dump({**want, "n_atoms": n, "sha256": sha256(dst)}, open(manifest, "w"), indent=1)
    return dst
