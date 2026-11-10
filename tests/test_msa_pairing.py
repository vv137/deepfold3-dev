"""Species-based online MSA pairing (``--msa-pairing``).

An AF3 ``pairedMsa`` is each chain's raw UniProt hits — per-chain row counts,
rows unrelated across chains — and has to be paired by species, as AF3 / OF3 do.
A ColabFold ``pairaln`` paired a3m is already row-aligned and must keep being
zipped by row index. These tests pin the OF3 pairing algorithm on small
synthetic MSAs, the ``auto`` rule that tells the two kinds apart, and that the
precomputed path is untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from deepfold3.data import const
from deepfold3.data.msa.pairing import (
    find_pairing_indices,
    pair_by_species,
    species_id,
    species_ids,
    unpaired_rows_not_in_paired,
)
from deepfold3.data.parse.a3m import parse_a3m, parse_a3m_with_headers
from deepfold3.data.parse.json_input import parse_json_input

Q1 = "ACDEFGHIKL"
Q2 = "MNPQRSTVWY"
Q3 = "GHIKLMNPQR"
_SUBST = "ACDEFGHIKLMNPQRSTVWY"


def _hit(query: str, i: int) -> str:
    """The ``i``-th single-substitution variant of ``query`` (distinct for distinct ``i``)."""
    variants = [(p, a) for p in range(len(query)) for a in _SUBST if a != query[p]]
    pos, sub = variants[i]
    return query[:pos] + sub + query[pos + 1 :]


def _uniprot(acc: str, species: str) -> str:
    return f"tr|{acc}|{acc}_{species}/1-10 [subseq from] protein OS=Some organism OX=1"


def _a3m(query: str, species: list[str], offset: int = 0) -> str:
    """Query row + one hit per ``species`` entry (``""`` = a header with no species)."""
    lines = [">query", query]
    for i, sp in enumerate(species, start=1):
        acc = f"A{offset + i:05d}"
        lines += [f">{_uniprot(acc, sp) if sp else 'UniRef100_' + acc}", _hit(query, offset + i)]
    return "\n".join(lines) + "\n"


def _msa_and_species(text: str):
    msa, headers = parse_a3m_with_headers(text)
    return msa, species_ids(headers)


def _rows(query: str, msa) -> list[int]:
    """Source hit index of each paired row (0 = query, -1 = all-gap row)."""
    lookup = {parse_a3m(f">q\n{_hit(query, i)}\n").sequences[0].tobytes(): i for i in range(1, 50)}
    lookup[parse_a3m(f">q\n{query}\n").sequences[0].tobytes()] = 0
    out = []
    for row in msa.sequences:
        if (row == const.MSA_GAP).all():
            out.append(-1)
        else:
            out.append(lookup[row.tobytes()])
    return out


# ---------------------------------------------------------------------------
# Species ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("tr|A0A3L7GYM1|A0A3L7GYM1_CRIGR/53-158 [subseq from] IGH-VJ558 OS=Cricetulus griseus OX=10029", "CRIGR"),
        ("tr|A0A218V2J3|A0A218V2J3_9PASE/68-135 Ig lambda chain V-1 region", "9PASE"),
        ("sp|P69905|HBA_HUMAN/2-142 Hemoglobin subunit alpha", "HUMAN"),
        ("sp|P69905|HBA_HUMAN", "HUMAN"),
        ("tr|D0VWU8|D0VWU8_BORBG/5-120", "BORBG"),
        ("UniRef100_UPI001E281CD0", ""),
        ("UniRef90_A0A8T2NDK6/4-78 Ig-like n=1 Tax=Albula RepID=A0A8T2NDK6_9TELE", ""),
        ("DUMMY", ""),
        ("query", ""),
        ("101", ""),
        ("", ""),
    ],
)
def test_species_id(header, expected):
    assert species_id(header) == expected


def test_species_ids_blank_the_query_row():
    assert list(species_ids(["sp|P69905|HBA_HUMAN", "sp|P69905|HBA_MOUSE"])) == ["", "MOUSE"]


def test_parse_a3m_with_headers_matches_parse_a3m():
    text = ">q desc\nACDEF\n>empty header\n>h1\nAC-EF\n>h2\nACdDEF\n"
    msa, headers = parse_a3m_with_headers(text)
    ref = parse_a3m(text)
    assert headers == ["q desc", "h1", "h2"]
    for a, b in ((msa.sequences, ref.sequences), (msa.mask, ref.mask), (msa.deletion_matrix, ref.deletion_matrix)):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Pairing algorithm
# ---------------------------------------------------------------------------


def test_two_chains_with_different_row_counts():
    """Species shared by both chains pair; species seen by one chain do not."""
    m1, s1 = _msa_and_species(_a3m(Q1, ["HUMAN", "MOUSE", "HUMAN", "YEAST"]))
    m2, s2 = _msa_and_species(_a3m(Q2, ["MOUSE", "HUMAN", "RAT"]))
    res = pair_by_species([m1, m2], [s1, s2])
    assert res is not None
    assert res.n_rows == 2
    assert [m.sequences.shape[0] for m in res.msas] == [3, 3]
    # Within the 2-chain tier species are sorted: HUMAN, then MOUSE.
    assert _rows(Q1, res.msas[0]) == [0, 1, 2]
    assert _rows(Q2, res.msas[1]) == [0, 2, 1]
    assert res.rows_per_chain == [2, 2]


def test_three_chains_with_a_chain_missing_a_species():
    """Rows covering more chains come first; the absent chain gets an all-gap row.

    HUMAN is in all three chains (counts 2/1/2): one full row, then — OF3 subtracts
    the pairing count and goes again — a second HUMAN row pairing chains 1 and 3
    only. MOUSE is in chains 1 and 2.
    """
    m1, s1 = _msa_and_species(_a3m(Q1, ["HUMAN", "MOUSE", "HUMAN"]))
    m2, s2 = _msa_and_species(_a3m(Q2, ["MOUSE", "HUMAN"]))
    m3, s3 = _msa_and_species(_a3m(Q3, ["HUMAN", "HUMAN", "YEAST"]))
    res = pair_by_species([m1, m2, m3], [s1, s2, s3])
    assert res is not None
    assert res.n_rows == 3
    assert {m.sequences.shape[0] for m in res.msas} == {4}
    assert _rows(Q1, res.msas[0]) == [0, 1, 3, 2]
    assert _rows(Q2, res.msas[1]) == [0, 2, -1, 1]
    assert _rows(Q3, res.msas[2]) == [0, 1, 2, -1]
    # An all-gap row keeps a full mask and zero deletions, as the featurizer's padding does.
    assert res.msas[1].mask[2].all() and not res.msas[1].deletion_matrix[2].any()


def test_no_shared_species_returns_none():
    m1, s1 = _msa_and_species(_a3m(Q1, ["HUMAN"]))
    m2, s2 = _msa_and_species(_a3m(Q2, ["MOUSE"]))
    assert pair_by_species([m1, m2], [s1, s2]) is None


def test_rows_without_species_never_pair():
    m1, s1 = _msa_and_species(_a3m(Q1, ["", "HUMAN"]))
    m2, s2 = _msa_and_species(_a3m(Q2, ["", "HUMAN"]))
    res = pair_by_species([m1, m2], [s1, s2])
    assert res.n_rows == 1
    assert _rows(Q1, res.msas[0]) == [0, 2]


def test_max_per_species_caps_rows():
    m1, s1 = _msa_and_species(_a3m(Q1, ["HUMAN"] * 4))
    m2, s2 = _msa_and_species(_a3m(Q2, ["HUMAN"] * 3))
    assert pair_by_species([m1, m2], [s1, s2]).n_rows == 3
    assert pair_by_species([m1, m2], [s1, s2], max_per_species=2).n_rows == 2


def test_truncation_keeps_best_ranked_rows():
    """Over ``max_rows``, rows are sorted by the product of their source ranks (OF3)."""
    m1, s1 = _msa_and_species(_a3m(Q1, ["BBB", "BBB", "AAA", "AAA"]))
    m2, s2 = _msa_and_species(_a3m(Q2, ["AAA", "AAA", "BBB", "BBB"]))
    full = pair_by_species([m1, m2], [s1, s2])
    assert _rows(Q1, full.msas[0]) == [0, 3, 4, 1, 2]  # species order, untouched below the cap
    cut = pair_by_species([m1, m2], [s1, s2], max_rows=2)
    assert cut.n_rows == 2
    assert _rows(Q1, cut.msas[0]) == [0, 3, 1]
    assert _rows(Q2, cut.msas[1]) == [0, 1, 3]


def test_find_pairing_indices_matches_of3_subtract_min():
    counts = np.array([[3, 1], [1, 1], [2, 0]])
    out = find_pairing_indices(counts, np.array([True, True]), max_rows=100)
    # n=3: species 0 once; n=2: species 0 (chains 0, 2) and species 1 (chains 0, 1).
    np.testing.assert_array_equal(out, [[0, 0, 0], [0, -1, 0], [1, 1, -1]])


def test_unpaired_rows_not_in_paired():
    m1, s1 = _msa_and_species(_a3m(Q1, ["HUMAN", "MOUSE"]))
    m2, s2 = _msa_and_species(_a3m(Q2, ["HUMAN"]))
    res = pair_by_species([m1, m2], [s1, s2])
    unpaired = parse_a3m(f">q\n{Q1}\n>a\n{_hit(Q1, 1)}\n>b\n{_hit(Q1, 7)}\n>c\n{Q1}\n")
    # Row 0 is kept; hit 1 is in the paired block; a hit equal to the query is not
    # matched against the paired query row.
    np.testing.assert_array_equal(unpaired_rows_not_in_paired(unpaired, res.msas[0]), [True, False, True, True])


# ---------------------------------------------------------------------------
# JSON inputs: mode resolution, homomers, precomputed unchanged
# ---------------------------------------------------------------------------


def _job_json(tmp_path: Path, chains: list[tuple[list[str], str, str | None, str | None]]) -> Path:
    """AF3-dialect job; each chain is ``(ids, sequence, pairedMsa, unpairedMsa)``."""
    seqs = []
    for ids, seq, paired, unpaired in chains:
        p = {"id": ids, "sequence": seq}
        if paired is not None:
            p["pairedMsa"] = paired
        if unpaired is not None:
            p["unpairedMsa"] = unpaired
        seqs.append({"protein": p})
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"name": "t", "dialect": "alphafold3", "version": 2, "sequences": seqs}))
    return path


def _raw_hits_job(tmp_path: Path) -> Path:
    return _job_json(
        tmp_path,
        [
            (
                ["A"],
                Q1,
                _a3m(Q1, ["HUMAN", "MOUSE", "HUMAN", "YEAST"]),
                f">q\n{Q1}\n>u\n{_hit(Q1, 1)}\n>v\n{_hit(Q1, 9)}\n",
            ),
            (["B"], Q2, _a3m(Q2, ["MOUSE", "HUMAN", "RAT"]), None),
        ],
    )


def test_auto_selects_species_for_raw_uniprot_hits(tmp_path):
    job = parse_json_input(_raw_hits_job(tmp_path), msa_pairing="auto")[0]
    assert job.msa_pairing == "species"
    depths = [c.paired_msa.sequences.shape[0] for c in job.protein_chains]
    assert depths == [3, 3]
    assert all(c.paired_msa_species is None for c in job.protein_chains)


def _colabfold_paired(query: str, n: int, offset: int) -> str:
    lines = [">101", query]
    for i in range(1, n + 1):
        lines += [f">UniRef100_UPI{offset + i:010d}" if i % 3 else ">DUMMY", _hit(query, i)]
    return "\n".join(lines) + "\n"


def test_precomputed_colabfold_input_is_untouched(tmp_path):
    """Row-aligned ColabFold pairs stay row-aligned under the default (auto)."""
    pa, pb = _colabfold_paired(Q1, 5, 0), _colabfold_paired(Q2, 5, 100)
    path = _job_json(tmp_path, [(["A"], Q1, pa, None), (["B"], Q2, pb, None)])
    for mode in ("auto", "precomputed"):
        job = parse_json_input(path, msa_pairing=mode)[0]
        assert job.msa_pairing == "precomputed"
        for chain, text in zip(job.protein_chains, (pa, pb), strict=True):
            ref = parse_a3m(text)
            np.testing.assert_array_equal(chain.paired_msa.sequences, ref.sequences)
            np.testing.assert_array_equal(chain.paired_msa.deletion_matrix, ref.deletion_matrix)
    # Forcing species on rows without UniProt headers finds nothing to pair.
    job = parse_json_input(path, msa_pairing="species")[0]
    assert job.msa_pairing == "species"
    assert all(c.paired_msa is None for c in job.protein_chains)


def test_auto_keeps_equal_depth_uniprot_pairs_row_aligned(tmp_path):
    pa, pb = _a3m(Q1, ["HUMAN", "MOUSE"]), _a3m(Q2, ["MOUSE", "HUMAN"])
    path = _job_json(tmp_path, [(["A"], Q1, pa, None), (["B"], Q2, pb, None)])
    job = parse_json_input(path)[0]
    assert job.msa_pairing == "precomputed"
    np.testing.assert_array_equal(job.protein_chains[1].paired_msa.sequences, parse_a3m(pb).sequences)


def test_precomputed_mode_features_unchanged(tmp_path):
    """Explicit precomputed zips raw rows by index and does not deduplicate — the old behaviour."""
    from deepfold3.data.module.dataset import InferenceInput, _tokenize_and_featurize
    from deepfold3.data.types import Input

    path = _raw_hits_job(tmp_path)
    job = parse_json_input(path, msa_pairing="precomputed")[0]
    assert job.msa_pairing == "precomputed"
    feats = InferenceInput.from_job(job)

    entry = json.loads(path.read_text())["sequences"]
    paired = {i: parse_a3m(e["protein"]["pairedMsa"]) for i, e in enumerate(entry)}
    unpaired = {0: parse_a3m(entry[0]["protein"]["unpairedMsa"]), 1: parse_a3m(f">q\n{Q2}\n")}
    from deepfold3.data.module.dataset import _build_complex_structure

    structure = _build_complex_structure([(Q1, const.MOL_PROTEIN, "A"), (Q2, const.MOL_PROTEIN, "B")])
    ref = _tokenize_and_featurize(Input(structure=structure, paired_msa=paired, unpaired_msa=unpaired))
    keys = [k for k in ref if any(t in k for t in ("msa", "deletion", "profile"))]
    assert "msa" in keys and "profile" in keys
    for key in keys:
        np.testing.assert_array_equal(feats[key].numpy(), ref[key].numpy(), err_msg=key)


def test_species_mode_deduplicates_unpaired_but_not_profile(tmp_path):
    from deepfold3.data.module.dataset import InferenceInput

    path = _raw_hits_job(tmp_path)
    species = InferenceInput.from_job(parse_json_input(path, msa_pairing="species")[0])
    pre = InferenceInput.from_job(parse_json_input(path, msa_pairing="precomputed")[0])
    # Chain A's unpaired hit 1 is also its first paired hit, so it leaves the
    # dense unpaired block: query + 2 paired rows + 1 unpaired row.
    assert species["msa"].shape[0] == 4
    np.testing.assert_array_equal(species["profile"].numpy(), pre["profile"].numpy())


def test_homomer_is_not_paired(tmp_path):
    path = _job_json(tmp_path, [(["A", "B"], Q1, _a3m(Q1, ["HUMAN", "MOUSE"]), None)])
    job = parse_json_input(path, msa_pairing="species")[0]
    assert job.msa_pairing == "species"
    assert all(c.paired_msa is None for c in job.protein_chains)
    # The default leaves a homomer as it was.
    job = parse_json_input(path)[0]
    assert job.msa_pairing == "precomputed"
    assert all(c.paired_msa is not None for c in job.protein_chains)


def test_copies_of_a_sequence_share_its_paired_block(tmp_path):
    path = _job_json(
        tmp_path,
        [
            (["A", "B"], Q1, _a3m(Q1, ["HUMAN", "MOUSE", "HUMAN"]), None),
            (["C"], Q2, _a3m(Q2, ["MOUSE", "HUMAN"]), None),
        ],
    )
    job = parse_json_input(path, msa_pairing="species")[0]
    a, b, c = job.protein_chains
    assert a.paired_msa is b.paired_msa
    assert a.paired_msa.sequences.shape[0] == c.paired_msa.sequences.shape[0] == 3


def test_unknown_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="msa_pairing"):
        parse_json_input(_raw_hits_job(tmp_path), msa_pairing="rows")


def test_cli_exposes_msa_pairing():
    from click.testing import CliRunner

    from deepfold3.scripts.inference import inference
    from deepfold3.scripts.inference_batch import inference_batch

    for cmd in (inference, inference_batch):
        out = CliRunner().invoke(cmd, ["--help"])
        assert out.exit_code == 0, out.output
        assert "--msa-pairing" in out.output and "species" in out.output


# ---------------------------------------------------------------------------
# Real input: FoldBench 8tuz (AF3 5-DB MSAs, online pairing)
# ---------------------------------------------------------------------------

_8TUZ = Path("/store/deepfold/users/gunwoong/Benchmark/Foldbench/AbAg/targets/8tuz-assembly1/inputs/A1B1C1")


@pytest.mark.skipif(not (_8TUZ / "af3_msa" / "input.json").exists(), reason="8tuz FoldBench input not present")
def test_real_8tuz_af3_msa_pairs_by_species():
    job = parse_json_input(_8TUZ / "af3_msa" / "input.json", max_templates=0)[0]
    assert job.msa_pairing == "species"
    depths = {c.paired_msa.sequences.shape[0] for c in job.protein_chains}
    assert len(depths) == 1 and depths.pop() > 1


@pytest.mark.skipif(not (_8TUZ / "mmseqs2_gpu" / "input.json").exists(), reason="8tuz FoldBench input not present")
def test_real_8tuz_precomputed_stays_row_aligned():
    job = parse_json_input(_8TUZ / "mmseqs2_gpu" / "input.json", max_templates=0)[0]
    assert job.msa_pairing == "precomputed"
    assert [c.paired_msa.sequences.shape[0] for c in job.protein_chains] == [1765, 1765, 1765]
