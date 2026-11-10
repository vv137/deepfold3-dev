"""Species-based online MSA pairing (AF2-Multimer / AF3 / OpenFold3 style).

An AlphaFold 3 ``pairedMsa`` is each chain's *raw* UniProt search result: the
row counts differ per chain and row *i* of one chain has nothing to do with row
*i* of another. AF3 pairs those rows by species at featurization time. This
module does the same, following OpenFold3's ``create_paired``
(``openfold3/core/data/pipelines/sample_processing/msa.py``):

1. Species id of every non-query row, from the UniProt header
   (``tr|A0A3L7GYM1|A0A3L7GYM1_CRIGR/53-158 ...`` → ``CRIGR``).
2. Keep at most ``max_per_species`` rows per species per chain, in file order.
3. Count rows per (chain, species); drop species seen by only one chain.
4. For ``n`` = number of chains down to ``min_chains``: every species present in
   exactly ``n`` chains contributes ``min(nonzero count)`` rows; those counts are
   subtracted, so a species' leftover rows are paired again at a smaller ``n``.
   Within one ``n`` tier species are in sorted order.
5. The *r*-th row of a species in a chain is that chain's *r*-th hit of the
   species (search rank order). A chain absent from a row gets an all-gap row.
6. Over ``max_rows`` rows: sort each tier by the product of the source row
   indices (best-ranked hits first) and truncate.

The returned blocks are row-aligned — exactly what the featurizer expects of a
precomputed paired MSA — with row 0 the query of every chain.
"""

from __future__ import annotations

import dataclasses
import re
from collections import defaultdict, deque
from collections.abc import Sequence

import numpy as np

from deepfold3.data import const
from deepfold3.data.types import MSA

MSA_PAIRING_MODES: tuple[str, ...] = ("auto", "precomputed", "species")
"""Values accepted by ``--msa-pairing`` / ``parse_json_input(msa_pairing=...)``."""

MAX_SEQ_PER_SPECIES = 600
"""OF3 ``MSASettings.max_seq_per_species`` (AF3 ``max_paired_sequence_per_species``)."""

MIN_CHAINS_PAIRED_PARTIAL = 2
"""OF3 ``MSASettings.min_chains_paired_partial``: shortest partial pairing kept."""

# AF3 ``msa_identifiers._UNIPROT_PATTERN``, applied to the header's first token
# with any ``/start-end`` suffix removed. Anchored, so UniRef / ColabFold
# (``UniRef100_UPI…``), BFD and ``DUMMY`` rows yield no species and never pair.
_UNIPROT_PATTERN = re.compile(
    r"^(?:tr|sp)\|[A-Za-z0-9]{6,10}(?:_\d)?\|[A-Za-z0-9]+_(?P<species>[A-Za-z0-9]{1,5})(?:_\d+)?$"
)


def species_id(header: str) -> str:
    """UniProt species mnemonic of one A3M header, or ``""`` if it has none.

    Args:
        header: Header text without the leading ``>``.

    Returns:
        E.g. ``"HUMAN"`` for ``sp|P69905|HBA_HUMAN/2-142 Hemoglobin ...``.
    """
    token = header.split(maxsplit=1)[0] if header.strip() else ""
    match = _UNIPROT_PATTERN.match(token.partition("/")[0])
    return match.group("species") if match else ""


def species_ids(headers: Sequence[str]) -> np.ndarray:
    """Species id of every row; row 0 (the query) is always ``""``.

    Args:
        headers: One header per MSA row, query first.

    Returns:
        Object array ``[S]`` of species strings.
    """
    out = np.array([species_id(h) for h in headers], dtype=object)
    if len(out):
        out[0] = ""
    return out


def species_fraction(species: np.ndarray | None) -> float:
    """Fraction of non-query rows that carry a species id (0 when there are none)."""
    if species is None or len(species) <= 1:
        return 0.0
    return float(np.mean([bool(s) for s in species[1:]]))


@dataclasses.dataclass
class PairingResult:
    """Row-aligned paired blocks plus the counts worth logging.

    Attributes:
        msas: One paired :class:`MSA` per input chain (same order), each
            ``[1 + n_rows, L_chain]``; row 0 is the chain's query.
        n_rows: Paired rows, excluding the query row.
        rows_per_chain: Paired rows in which each chain has a real hit
            (the rest are all-gap rows for that chain).
    """

    msas: list[MSA]
    n_rows: int
    rows_per_chain: list[int]


def find_pairing_indices(
    count_array: np.ndarray,
    pairing_mask: np.ndarray,
    max_rows: int,
    min_chains: int = MIN_CHAINS_PAIRED_PARTIAL,
) -> np.ndarray:
    """Species index of each paired row per chain (``-1`` = chain absent).

    Port of OF3 ``find_pairing_indices``.

    Args:
        count_array: ``[n_chains, n_species]`` rows per chain per species.
        pairing_mask: ``[n_species]`` species eligible for pairing.
        max_rows: Stop adding tiers once this many rows exist.
        min_chains: Smallest number of chains a row may cover.

    Returns:
        ``[n_rows, n_chains]`` int array.
    """
    counts = count_array[:, pairing_mask].copy()
    species_index = np.arange(count_array.shape[1])[pairing_mask]
    n_chains = count_array.shape[0]
    blocks: list[np.ndarray] = []
    n_rows = 0
    for n in range(n_chains, min(min_chains, n_chains) - 1, -1):
        in_n = (counts > 0).sum(axis=0) == n
        if not in_n.any():
            continue
        sub = counts[:, in_n]
        k = np.where(sub > 0, sub, np.iinfo(np.int64).max).min(axis=0)
        cols = np.repeat(species_index[in_n], k)
        block = np.tile(cols, (n_chains, 1))
        block[np.repeat(sub, k, axis=1) == 0] = -1
        blocks.append(block.T)
        n_rows += block.shape[1]
        counts[:, in_n] = np.where(sub != 0, sub - k, sub)
        if n_rows >= max_rows:
            break
    if not blocks:
        return np.empty((0, n_chains), dtype=np.int64)
    return np.concatenate(blocks, axis=0)


def pair_by_species(
    msas: Sequence[MSA],
    species: Sequence[np.ndarray],
    max_rows: int = const.MAX_PAIRED_MSA_ROWS,
    max_per_species: int = MAX_SEQ_PER_SPECIES,
    min_chains: int = MIN_CHAINS_PAIRED_PARTIAL,
) -> PairingResult | None:
    """Pair the rows of several chains' UniProt MSAs by species.

    Args:
        msas: One raw (unpaired-by-row) MSA per *distinct* chain sequence, query
            as row 0.
        species: Per-row species ids for each MSA (see :func:`species_ids`).
        max_rows: Paired rows to keep, excluding the query
            (``const.MAX_PAIRED_MSA_ROWS`` = OF3 ``max_rows_paired``).
        max_per_species: Rows kept per species per chain before pairing.
        min_chains: Smallest number of chains a paired row may cover.

    Returns:
        A :class:`PairingResult`, or ``None`` when fewer than two chains have
        hits or no species is shared by at least ``min_chains`` chains.
    """
    if len(msas) != len(species):
        raise ValueError(f"{len(msas)} MSAs but {len(species)} species arrays")
    if len(msas) < 2:
        return None

    # species -> deque of source rows, per chain (capped, file order).
    rows_by_species: list[dict[str, deque[int]]] = []
    for sp in species:
        per: dict[str, deque[int]] = defaultdict(deque)
        for row in range(1, len(sp)):
            s = sp[row]
            if s and len(per[s]) < max_per_species:
                per[s].append(row)
        rows_by_species.append(per)

    names = sorted({s for per in rows_by_species for s in per})
    if not names:
        return None
    col = {s: i for i, s in enumerate(names)}
    counts = np.zeros((len(msas), len(names)), dtype=np.int64)
    for c, per in enumerate(rows_by_species):
        for s, rows in per.items():
            counts[c, col[s]] = len(rows)

    shared = (counts > 0).sum(axis=0) >= 2  # OF3 pairing mask "shared_by_two"
    if not shared.any():
        return None
    paired_species = find_pairing_indices(counts, shared, max_rows, min_chains)
    if paired_species.shape[0] == 0:
        return None

    # Species index -> source row, popping each chain's hits in rank order.
    row_ids = np.full(paired_species.shape, -1, dtype=np.int64)
    for c, per in enumerate(rows_by_species):
        queues = {col[s]: deque(rows) for s, rows in per.items()}
        for r, s_idx in enumerate(paired_species[:, c]):
            if s_idx >= 0:
                row_ids[r, c] = queues[s_idx].popleft()

    if row_ids.shape[0] > max_rows:
        row_ids = _sort_by_row_product(row_ids)[:max_rows]

    out: list[MSA] = []
    for c, msa in enumerate(msas):
        out.append(_gather_rows(msa, row_ids[:, c]))
    return PairingResult(
        msas=out,
        n_rows=int(row_ids.shape[0]),
        rows_per_chain=[int((row_ids[:, c] >= 0).sum()) for c in range(len(msas))],
    )


def _sort_by_row_product(row_ids: np.ndarray) -> np.ndarray:
    """OF3 ``sort_by_row_id_product``: within each coverage tier, best-ranked hits first.

    ``|prod(row ids)|`` with absent chains (``-1``) contributing a factor 1; the
    log-sum gives the same order without overflowing for many chains.
    """
    out = row_ids.copy()
    n_absent = (row_ids < 0).sum(axis=1)
    score = np.log(np.abs(row_ids).astype(np.float64)).sum(axis=1)
    for tier in np.unique(n_absent):
        idx = np.flatnonzero(n_absent == tier)
        out[idx] = row_ids[idx[np.argsort(score[idx], kind="stable")]]
    return out


def _gather_rows(msa: MSA, rows: np.ndarray) -> MSA:
    """``[query; msa[rows]]`` with all-gap rows where ``rows == -1``."""
    width = msa.sequences.shape[1]
    n = 1 + rows.shape[0]
    seqs = np.full((n, width), const.MSA_GAP, dtype=msa.sequences.dtype)
    mask = np.ones((n, width), dtype=bool)
    dels = np.zeros((n, width), dtype=msa.deletion_matrix.dtype)
    seqs[0], mask[0], dels[0] = msa.sequences[0], msa.mask[0], msa.deletion_matrix[0]
    hit = np.flatnonzero(rows >= 0)
    src = rows[hit]
    seqs[1 + hit] = msa.sequences[src]
    mask[1 + hit] = msa.mask[src]
    dels[1 + hit] = msa.deletion_matrix[src]
    return MSA(sequences=seqs, mask=mask, deletion_matrix=dels)


def unpaired_rows_not_in_paired(unpaired: MSA, paired: MSA) -> np.ndarray:
    """Mask of unpaired rows to keep: not identical to any non-query paired row.

    OF3 ``create_main`` / AF3 ``deduplicate_unpaired_sequences``: an unpaired row
    whose residues (deletions ignored) equal a paired row of the same chain is
    dropped. Row 0 (the query) is always kept and the paired query row is not a
    match target, so a hit identical to the query is only dropped when the
    paired block holds it as a hit too.

    Args:
        unpaired: The chain's unpaired MSA.
        paired: The chain's paired block (row 0 = query).

    Returns:
        Bool ``[S_unpaired]``; all ``True`` if the widths differ.
    """
    keep = np.ones(unpaired.sequences.shape[0], dtype=bool)
    if paired.sequences.shape[0] <= 1 or unpaired.sequences.shape[0] <= 1:
        return keep
    if unpaired.sequences.shape[1] != paired.sequences.shape[1]:
        return keep
    void = np.dtype((np.void, unpaired.sequences.dtype.itemsize * unpaired.sequences.shape[1]))
    a = np.ascontiguousarray(unpaired.sequences[1:]).view(void).ravel()
    b = np.ascontiguousarray(paired.sequences[1:].astype(unpaired.sequences.dtype)).view(void).ravel()
    keep[1:] = ~np.isin(a, b)
    return keep
