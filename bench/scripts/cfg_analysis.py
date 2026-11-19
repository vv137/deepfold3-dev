"""Classifier-free-guidance analysis over scored FoldBench interface series.

Reads the per-sample raw CSVs that ``eval_ost.py`` writes
(``<bench>/runs/<series>/eval/<cat>/step_<N>/raw/<cat>_ost.csv``) and compares series
on one interface set: the interfaces for which every compared series has all its
samples (25 = 5 seeds x 5 samples in the FoldBench protocol). Nothing here runs OST.

Per interface, over its samples:
  oracle = best DockQ; conf = DockQ of the top ``ranking_score`` sample (ties averaged,
  as ``eval_ost.conf_pick``); mean = mean DockQ. Success = DockQ >= 0.23.

Subcommands (series are ``<series>[:<step>]``, step defaults to 5000; the ``hangang-mmseqs2-``
prefix is added when the bare name is not a run directory):

  table   Oracle / Conf / Mean DockQ and success on the common set.
  curves  best-of-k: success probability and expected best DockQ when k of the
          samples are drawn (exact, without replacement), and the same by whole seeds.
  pool    success when a budget of k samples is split evenly over several series
          (W-pooling), against one series alone at the same k.
  rescue  interfaces the baseline fails (no sample succeeds) that another series
          solves, and the reverse, with each interface's best DockQ on both sides.

    python bench/scripts/cfg_analysis.py table interface_antibody_antigen stage3-001 stage3-001-cfg-full_w0.8
    python bench/scripts/cfg_analysis.py rescue interface_protein_protein stage3-s1-frozen-zero \\
        stage3-s1-frozen-zero-cfg-full_w0.6 --list
    python bench/scripts/cfg_analysis.py pool interface_antibody_antigen stage3-001 \\
        --with stage3-001-cfg-full_w0.8 --k 10 30 50

The bench root is ``$DF3_BENCH`` when set (as for ``eval_ost.py``), else ``<repo>/bench``.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import re
from dataclasses import dataclass
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = Path(os.environ.get("DF3_BENCH", REPO / "bench"))
SUCCESS = 0.23
DEFAULT_STEP = 5000
TOTAL_INTERFACES = {"interface_antibody_antigen": 172, "interface_protein_protein": 279}
SERIES_PREFIX = "hangang-mmseqs2-"


@dataclass(frozen=True)
class Sample:
    seed: str
    ranking_score: float
    dockq: float


Interface = tuple[str, str, str]  # (pdb_id, ic1, ic2)


def series_dir(series: str) -> Path:
    runs = BENCH / "runs"
    return runs / series if (runs / series).exists() else runs / f"{SERIES_PREFIX}{series}"


def split_arg(arg: str) -> tuple[str, int]:
    series, _, step = arg.partition(":")
    return series, int(step or DEFAULT_STEP)


def load(arg: str, category: str, samples: int | None = 25) -> dict[Interface, list[Sample]]:
    """Per-interface samples of one series; interfaces with fewer than ``samples`` rows are dropped.

    ``samples=None`` keeps every interface whatever its count.
    """
    series, step = split_arg(arg)
    f = series_dir(series) / "eval" / category / f"step_{step}" / "raw" / f"{category}_ost.csv"
    if not f.exists():
        return {}
    by: dict[Interface, list[Sample]] = collections.defaultdict(list)
    for r in csv.DictReader(open(f)):
        try:
            s = Sample(r["seed"], float(r["ranking_score"]), float(r["dockq"]))
        except (KeyError, ValueError):
            continue
        by[(r["pdb_id"], r["ic1"], r["ic2"])].append(s)
    return {k: v for k, v in by.items() if samples is None or len(v) == samples}


def common_set(
    data: dict[str, dict[Interface, list[Sample]]], min_fraction: float = 0.0, total: int | None = None
) -> list[Interface]:
    """Interfaces present in every series that has at least ``min_fraction`` of ``total``."""
    keep = [d for d in data.values() if total is None or len(d) >= min_fraction * total]
    return sorted(set.intersection(*(set(d) for d in keep))) if keep else []


def oracle(v: list[Sample]) -> float:
    return max(s.dockq for s in v)


def conf(v: list[Sample]) -> float:
    top = max(s.ranking_score for s in v)
    tied = [s.dockq for s in v if s.ranking_score == top]
    return sum(tied) / len(tied)


def mean(v: list[Sample]) -> float:
    return sum(s.dockq for s in v) / len(v)


AGGREGATES = {"oracle": oracle, "conf": conf, "mean": mean}


def summarize(d: dict[Interface, list[Sample]], keys: list[Interface]) -> dict[str, float]:
    """Mean DockQ and success % of each aggregate over ``keys``."""
    out = {}
    n = len(keys)
    for name, fn in AGGREGATES.items():
        vals = [fn(d[k]) for k in keys]
        out[f"{name}_dq"] = sum(vals) / n if n else float("nan")
        out[f"{name}_succ"] = 100 * sum(v >= SUCCESS for v in vals) / n if n else float("nan")
    return out


def pass_at(n: int, c: int, k: int) -> float:
    """P(at least one success among k of n drawn without replacement), c of n succeed."""
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    return 1.0 - comb(n - c, k) / comb(n, k)


def emax_at(vals: list[float], k: int) -> float:
    """E[max of k of vals drawn without replacement] (exact order statistic)."""
    v = sorted(vals)
    n = len(v)
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    return sum(v[i] * comb(i, k - 1) for i in range(k - 1, n)) / comb(n, k)


def curves(d: dict[Interface, list[Sample]], keys: list[Interface]) -> dict[str, list[float]]:
    """best-of-k over samples (k = 1..n) and over whole seeds (j = 1..seeds), averaged over keys."""
    n = len(d[keys[0]])
    seeds = len({s.seed for s in d[keys[0]]})
    p, e, ps = [0.0] * n, [0.0] * n, [0.0] * seeds
    for key in keys:
        v = d[key]
        dq = [s.dockq for s in v]
        c = sum(x >= SUCCESS for x in dq)
        by_seed = collections.defaultdict(list)
        for s in v:
            by_seed[s.seed].append(s.dockq)
        cs = sum(max(z) >= SUCCESS for z in by_seed.values())
        for k in range(1, n + 1):
            p[k - 1] += pass_at(n, c, k)
            e[k - 1] += emax_at(dq, k)
        for j in range(1, seeds + 1):
            ps[j - 1] += pass_at(len(by_seed), cs, j)
    m = len(keys)
    return {"pass": [100 * x / m for x in p], "emax": [x / m for x in e], "pass_seed": [100 * x / m for x in ps]}


def pooled_pass(ds: list[dict[Interface, list[Sample]]], keys: list[Interface], k: int) -> float | None:
    """Success % when k samples are split evenly over the series in ``ds``; None if k does not split."""
    if k % len(ds):
        return None
    kw = k // len(ds)
    tot = 0.0
    for key in keys:
        fail = 1.0
        for d in ds:
            v = d[key]
            if kw > len(v):
                return None
            c = sum(s.dockq >= SUCCESS for s in v)
            fail *= comb(len(v) - c, kw) / comb(len(v), kw)
        tot += 1 - fail
    return 100 * tot / len(keys)


@dataclass
class Rescue:
    key: Interface
    base_best: float
    other_best: float
    base_hits: int
    other_hits: int


def rescue(
    base: dict[Interface, list[Sample]], other: dict[Interface, list[Sample]], keys: list[Interface]
) -> tuple[list[Rescue], list[Rescue]]:
    """(rescued, lost): interfaces only ``other`` solves at the oracle, and only ``base`` solves."""

    def row(k):
        return Rescue(
            k,
            oracle(base[k]),
            oracle(other[k]),
            sum(s.dockq >= SUCCESS for s in base[k]),
            sum(s.dockq >= SUCCESS for s in other[k]),
        )

    rows = [row(k) for k in keys]
    rescued = sorted((r for r in rows if r.base_best < SUCCESS <= r.other_best), key=lambda r: -r.other_best)
    lost = sorted((r for r in rows if r.other_best < SUCCESS <= r.base_best), key=lambda r: -r.base_best)
    return rescued, lost


# ---------------------------------------------------------------- labels

W_RE = re.compile(r"(.+?)(?:-cfg-(full|abag|all)_w(-?[\d.]+))?")


def parse_series(series: str) -> tuple[str, str, str]:
    """(base series, drop scope or '—', W) of a series name; W = '1' for an unguided run."""
    m = W_RE.fullmatch(series)
    base, scope, w = m.group(1), m.group(2), m.group(3)
    return base, scope or "—", f"{float(w):g}" if w else "1"


# ---------------------------------------------------------------- CLI


def _fmt_table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def cmd_table(a):
    data = {s: load(s, a.category) for s in a.series}
    total = TOTAL_INTERFACES.get(a.category)
    keys = common_set(data, 0.5, total)
    rows = []
    for s, d in data.items():
        base, scope, w = parse_series(split_arg(s)[0])
        dag = " †" if total and len(d) < total else ""
        if len(d) < 0.5 * (total or 0):
            rows.append([base + dag, w, scope, f"{len(d)}/{total}"] + ["pending"] * 6)
            continue
        m = summarize(d, keys)
        rows.append(
            [base + dag, w, scope, f"{len(d)}/{total}"]
            + [
                f"{m[f'{x}_{y}']:.3f}" if y == "dq" else f"{m[f'{x}_{y}']:.1f}"
                for x in AGGREGATES
                for y in ("dq", "succ")
            ]
        )
    print(f"{a.category}: {len(keys)} common interfaces (of {total})\n")
    print(
        _fmt_table(
            rows,
            [
                "series",
                "W",
                "drop",
                "collected",
                "Oracle DQ",
                "Oracle Succ",
                "Conf DQ",
                "Conf Succ",
                "Mean DQ",
                "Mean Succ",
            ],
        )
    )


def cmd_curves(a):
    data = {s: load(s, a.category) for s in a.series}
    keys = common_set(data)
    ks = a.k or [1, 2, 5, 10, 25]
    rows = []
    for s, d in data.items():
        c = curves(d, keys)
        rows.append(
            [s]
            + [f"{c['pass'][k - 1]:.1f}" for k in ks]
            + [f"{c['emax'][k - 1]:.3f}" for k in ks]
            + [f"{x:.1f}" for x in c["pass_seed"]]
        )
    print(f"{a.category}: {len(keys)} common interfaces\n")
    print(
        _fmt_table(
            rows,
            ["series"]
            + [f"succ@{k}" for k in ks]
            + [f"bestDQ@{k}" for k in ks]
            + [f"succ@{j}seed" for j in range(1, 6)],
        )
    )


def cmd_pool(a):
    base = load(a.base, a.category)
    others = [load(s, a.category) for s in a.with_]
    keys = common_set({"b": base, **{str(i): d for i, d in enumerate(others)}})
    rows = []
    for name, ds in ((a.base, [base]), (" + ".join([a.base] + a.with_), [base] + others)):
        rows.append([name] + [("—" if (v := pooled_pass(ds, keys, k)) is None else f"{v:.1f}") for k in a.k])
    print(f"{a.category}: {len(keys)} common interfaces; budget split evenly over the pooled series\n")
    print(_fmt_table(rows, ["pool"] + [f"succ@{k}" for k in a.k]))


def _fmt_rescue(r: Rescue) -> str:
    return f"{r.key[0]} {r.key[1]}:{r.key[2]} {r.base_best:.2f}({r.base_hits})→{r.other_best:.2f}({r.other_hits})"


def cmd_rescue(a):
    base = load(a.base, a.category)
    rows = []
    for s in a.others:
        other = load(s, a.category)
        keys = common_set({"b": base, "o": other})
        res, lost = rescue(base, other, keys)
        fail = sum(oracle(base[k]) < SUCCESS for k in keys)
        rows.append([s, str(len(keys)), str(fail), f"+{len(res)}", f"-{len(lost)}"])
        if a.list:
            print(
                f"\n{s}\n  rescued: "
                + "; ".join(map(_fmt_rescue, res))
                + "\n  lost:    "
                + "; ".join(map(_fmt_rescue, lost))
            )
    print(f"\n{a.category}: baseline {a.base}; best DockQ (successful samples) shown as base→other\n")
    print(_fmt_table(rows, ["series", "interfaces", "baseline fails", "rescued", "lost"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("table")
    p.add_argument("category")
    p.add_argument("series", nargs="+")
    p.set_defaults(fn=cmd_table)
    p = sub.add_parser("curves")
    p.add_argument("category")
    p.add_argument("series", nargs="+")
    p.add_argument("--k", type=int, nargs="+")
    p.set_defaults(fn=cmd_curves)
    p = sub.add_parser("pool")
    p.add_argument("category")
    p.add_argument("base")
    p.add_argument("--with", dest="with_", nargs="+", required=True)
    p.add_argument("--k", type=int, nargs="+", default=[10, 20, 50])
    p.set_defaults(fn=cmd_pool)
    p = sub.add_parser("rescue")
    p.add_argument("category")
    p.add_argument("base")
    p.add_argument("others", nargs="+")
    p.add_argument("--list", action="store_true", help="print each rescued / lost interface")
    p.set_defaults(fn=cmd_rescue)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
