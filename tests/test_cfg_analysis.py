"""bench/scripts/cfg_analysis.py: exact best-of-k, W-pooling and rescue counts."""

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "scripts"))
import cfg_analysis as A  # noqa: E402


def _samples(dqs, seeds=None, scores=None):
    seeds = seeds or [str(i // 5) for i in range(len(dqs))]
    scores = scores or [0.5] * len(dqs)
    return [A.Sample(s, r, d) for s, r, d in zip(seeds, scores, dqs, strict=True)]


def _brute(vals, k, fn):
    combos = list(itertools.combinations(vals, k))
    return sum(fn(c) for c in combos) / len(combos)


@pytest.mark.parametrize("k", [1, 2, 3, 6])
def test_pass_and_emax_match_enumeration(k):
    vals = [0.05, 0.3, 0.1, 0.6, 0.22, 0.24]
    c = sum(v >= A.SUCCESS for v in vals)
    assert A.pass_at(len(vals), c, k) == pytest.approx(_brute(vals, k, lambda s: any(v >= A.SUCCESS for v in s)))
    assert A.emax_at(vals, k) == pytest.approx(_brute(vals, k, max))


def test_curve_endpoints_are_mean_and_oracle():
    dqs = [0.1 * (i % 7) for i in range(25)]
    d = {("x", "A", "B"): _samples(dqs)}
    c = A.curves(d, list(d))
    assert c["emax"][0] == pytest.approx(A.mean(d[("x", "A", "B")]))
    assert c["emax"][-1] == pytest.approx(A.oracle(d[("x", "A", "B")]))
    assert c["pass_seed"][-1] == pytest.approx(100.0)
    assert len(c["pass_seed"]) == 5


def test_pool_of_one_series_is_pass_at_k():
    dqs = [0.0] * 20 + [0.5] * 5
    d = {("x", "A", "B"): _samples(dqs)}
    assert A.pooled_pass([d], list(d), 10) == pytest.approx(100 * A.pass_at(25, 5, 10))
    assert A.pooled_pass([d, d], list(d), 10) == pytest.approx(100 * (1 - (A.comb(20, 5) / A.comb(25, 5)) ** 2))
    assert A.pooled_pass([d, d], list(d), 7) is None  # does not split evenly
    assert A.pooled_pass([d], list(d), 30) is None  # more than the series holds


def test_conf_averages_tied_top_scores():
    v = _samples([0.1, 0.5, 0.9], seeds=["0", "0", "0"], scores=[0.8, 0.8, 0.2])
    assert A.conf(v) == pytest.approx(0.3)


def test_rescue_and_loss():
    keys = [("a", "A", "B"), ("b", "A", "B"), ("c", "A", "B")]
    base = {keys[0]: _samples([0.1] * 25), keys[1]: _samples([0.5] * 25), keys[2]: _samples([0.3] * 25)}
    other = {keys[0]: _samples([0.1] * 24 + [0.4]), keys[1]: _samples([0.1] * 25), keys[2]: _samples([0.3] * 25)}
    res, lost = A.rescue(base, other, keys)
    assert [r.key for r in res] == [keys[0]] and res[0].other_hits == 1
    assert [r.key for r in lost] == [keys[1]] and lost[0].base_hits == 25


@pytest.mark.parametrize(
    ("name", "parsed"),
    [
        ("stage3-001", ("stage3-001", "—", "1")),
        ("stage3-001-cfg-full_w0.8", ("stage3-001", "full", "0.8")),
        ("stage3-s1-frozen-zero-cfg-abag_w-0.3", ("stage3-s1-frozen-zero", "abag", "-0.3")),
    ],
)
def test_parse_series(name, parsed):
    assert A.parse_series(name) == parsed


def test_split_chunks_gives_ordinary_runs_and_null_counts():
    """125 samples = 5 passes of 5 per seed; pass 0 vs pass c is the W = 1 null."""
    key = ("x", "A", "B")
    v = [A.Sample(str(seed), 0.5, 0.5 if (seed == 0 and s == 7) else 0.0, s) for seed in range(5) for s in range(25)]
    passes = A.split_chunks({key: v})
    assert len(passes) == 5 and all(len(p[key]) == 25 for p in passes)
    assert {s.sample for s in passes[1][key]} == set(range(5, 10))
    assert A.null_rescue(passes, [key]) == [(1, 0), (0, 0), (0, 0), (0, 0)]


def test_pooled_emax_matches_enumeration_and_single_series():
    key = ("x", "A", "B")
    a = {key: _samples([0.1, 0.5, 0.2, 0.9, 0.3])}
    b = {key: _samples([0.4, 0.05, 0.6, 0.2, 0.15])}
    assert A.pooled_emax([a], [key], 3) == pytest.approx(A.emax_at([s.dockq for s in a[key]], 3))
    va, vb = [s.dockq for s in a[key]], [s.dockq for s in b[key]]
    want = sum(max(x + y) for x in itertools.combinations(va, 2) for y in itertools.combinations(vb, 2))
    want /= len(list(itertools.combinations(va, 2))) * len(list(itertools.combinations(vb, 2)))
    assert A.pooled_emax([a, b], [key], 4) == pytest.approx(want)
    assert A.pooled_emax([a, b], [key], 3) is None
