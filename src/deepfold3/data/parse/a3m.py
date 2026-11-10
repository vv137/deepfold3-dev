"""A3M multiple sequence alignment parser.

Converts raw A3M text into an MSA dataclass.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from deepfold3.data import const
from deepfold3.data.types import MSA

logger = logging.getLogger(__name__)


def _encode_sequence(seq: str, mol_type: int) -> tuple[list[int], list[int]]:
    """Encode one A3M sequence to MSA indices and deletion counts.

    Returns
    -------
    residues : list[int]  — encoded match-state residues
    deletions : list[int] — number of insertions before each match state
    """
    residues: list[int] = []
    deletions: list[int] = []
    del_count = 0

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

    for ch in seq:
        if ch == ".":
            # Insert-state gap (some A3M dialects): no residue and not a match
            # column, so skip it entirely — it neither consumes a query column
            # nor counts as a deletion. Was previously uppercase-tested false and
            # encoded as a spurious UNK match column, shifting the whole row.
            continue
        if ch.islower():
            # Insertion: increment deletion counter, don't add to match states
            del_count += 1
        else:
            # Match state
            encoded = lookup.get(ch.upper(), unk)
            residues.append(encoded)
            deletions.append(del_count)
            del_count = 0

    return residues, deletions


def parse_a3m(text: str, mol_type: int = const.MOL_PROTEIN) -> MSA:
    """Parse an A3M-format MSA string into an MSA dataclass.

    Parameters
    ----------
    text :     A3M text (lines may or may not start with '>')
    mol_type : MOL_PROTEIN / MOL_RNA / MOL_DNA

    Returns
    -------
    MSA with:
        sequences [S, N] int8
        mask      [S, N] bool   (always True for non-gap positions)
        deletion_matrix [S, N] int8
    """
    return _encode_records(_split_records(text)[1], mol_type)


def parse_a3m_with_headers(text: str, mol_type: int = const.MOL_PROTEIN) -> tuple[MSA, list[str]]:
    """Parse an A3M string and keep each row's header.

    Same rows as :func:`parse_a3m` (a header with no sequence lines yields no
    row), so ``headers[i]`` describes ``msa.sequences[i]``.

    Args:
        text: A3M text.
        mol_type: ``MOL_PROTEIN`` / ``MOL_RNA`` / ``MOL_DNA``.

    Returns:
        ``(msa, headers)``; each header is the text after ``>`` (``""`` for a
        row with no header line).
    """
    headers, seqs = _split_records(text)
    return _encode_records(seqs, mol_type), headers


def _split_records(text: str) -> tuple[list[str], list[str]]:
    """Split A3M text into parallel ``(headers, sequences)`` lists."""
    headers: list[str] = []
    seqs: list[str] = []
    current_seq: list[str] = []
    current_header = ""

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_seq:
                headers.append(current_header)
                seqs.append("".join(current_seq))
                current_seq = []
            current_header = line[1:]
        else:
            current_seq.append(line)
    if current_seq:
        headers.append(current_header)
        seqs.append("".join(current_seq))
    return headers, seqs


def _encode_records(seqs: list[str], mol_type: int) -> MSA:
    """Encode A3M row strings into an :class:`MSA` (query = row 0 sets the width)."""
    if not seqs:
        # Return single empty sequence
        return MSA(
            sequences=np.zeros((0, 0), dtype=np.int8),
            mask=np.zeros((0, 0), dtype=bool),
            deletion_matrix=np.zeros((0, 0), dtype=np.int8),
        )

    # Encode all sequences; determine N from query (first) sequence
    encoded_list: list[list[int]] = []
    deletion_list: list[list[int]] = []

    for seq in seqs:
        res, dels = _encode_sequence(seq, mol_type)
        encoded_list.append(res)
        deletion_list.append(dels)

    # N = number of match states in query
    N = len(encoded_list[0])

    # Build arrays; sequences shorter/longer than N are padded/truncated
    S = len(encoded_list)
    sequences = np.zeros((S, N), dtype=np.int8)
    deletion_matrix = np.zeros((S, N), dtype=np.int8)

    for i, (res, dels) in enumerate(zip(encoded_list, deletion_list, strict=False)):
        L = min(len(res), N)
        sequences[i, :L] = res[:L]
        deletion_matrix[i, :L] = [min(d, 127) for d in dels[:L]]

    # mask: True at valid aligned positions (including gaps).
    # AF3 treats gaps as valid evolutionary signal — only padding is False.
    mask = np.ones((S, N), dtype=bool)
    for i, res in enumerate(encoded_list):
        L = min(len(res), N)
        if L < N:
            mask[i, L:] = False  # padding beyond sequence length

    return MSA(sequences=sequences, mask=mask, deletion_matrix=deletion_matrix)


def convert_a3m_file_to_npz(
    a3m_path: Path,
    npz_path: Path,
    mol_type: int = const.MOL_PROTEIN,
) -> bool:
    """Parse an A3M file and write the resulting MSA as a compressed NPZ.

    Single source of truth for the A3M → MSA NPZ conversion used by
    ``msa-batch``, ``import-msa``, and ``import-paired-msa``. Failures
    are logged at WARNING and reported as a False return so batch
    callers can tally a count.
    """
    try:
        a3m_text = Path(a3m_path).read_text()
        msa = parse_a3m(a3m_text, mol_type=mol_type)
        if msa.is_empty:
            logger.warning("convert_a3m_file_to_npz: empty MSA (shape %s) for %s", msa.sequences.shape, a3m_path)
            return False
        msa.save(Path(npz_path))
        return True
    except Exception:
        logger.warning("convert_a3m_file_to_npz failed for %s", a3m_path, exc_info=True)
        return False
