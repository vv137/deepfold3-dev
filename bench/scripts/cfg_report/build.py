"""Build the classifier-free-guidance results page from scored FoldBench series.

All numbers come from ``cfg_analysis`` (the sibling module) over the raw OST CSVs; this
file only chooses which series go in which table and writes ``template.html`` with the
data inlined. Rows with under half of a category's interfaces collected stay out of a
table's common set and render as pending.

    DF3_BENCH=/path/to/bench python bench/scripts/cfg_report/build.py --out report.html
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cfg_analysis as A  # noqa: E402

HERE = Path(__file__).parent
W_GRID = ["-0.5", "-0.4", "-0.3", "-0.2", "-0.1", "0.0", "0.2", "0.4", "0.6", "0.8", "2.0", "3.0", "4.0", "5.0"]
CURVE_W = ["0.4", "0.6", "0.8", "1", "2.0", "3.0"]
RESCUE_W = ["-0.3", "0.0", "0.4", "0.6", "0.8", "2.0", "3.0"]
CATEGORIES = (
    ("interface_antibody_antigen", "AbAg", ("full", "abag")),
    ("interface_protein_protein", "PPI", ("full", "all")),
)

# node id in the page's lineage graph -> unguided series, sweep prefix
MODELS = {
    "s3": ("stage3-001", "stage3-001"),
    "fz": ("stage3-s1-frozen-zero", "stage3-s1-frozen-zero"),
    "fl": ("stage3-s1-frozen-learned", "stage3-s1-frozen-learned"),
}
BASE = [
    ("stage1-rescore:120000", "s1"),
    ("stage2-001-rescore:50000", "s2"),
    ("stage3-001", "s3"),
    ("stage3-s1-001", "s3s1"),
    ("stage3-s1-frozen-zero", "fz"),
    ("stage3-s1-frozen-learned", "fl"),
]


def series_for(node: str, scope: str, w: str) -> str:
    base, prefix = MODELS[node]
    return base if w == "1" else f"{prefix}-cfg-{scope}_w{w}"


def row(series: str, node: str, w: str, scope: str) -> dict:
    return {"series": series, "node": node, "w": f"{float(w):g}", "scope": scope}


def table(rows: list[dict], cat: str) -> dict:
    total = A.TOTAL_INTERFACES[cat]
    data = {r["series"]: A.load(r["series"], cat) for r in rows}
    keys = A.common_set(data, 0.5, total)
    out = []
    for r in rows:
        d = data[r["series"]]
        rr = {**r, "collected": len(d), "total": total}
        if keys and len(d) >= total / 2:
            rr.update({k: round(v, 3 if k.endswith("dq") else 1) for k, v in A.summarize(d, keys).items()})
        out.append(rr)
    return {"common": len(keys), "total": total, "rows": out}


def curve_set(node: str, scope: str, cat: str) -> dict | None:
    total = A.TOTAL_INTERFACES[cat]
    data = {w: A.load(series_for(node, scope, w), cat) for w in CURVE_W}
    data = {w: d for w, d in data.items() if len(d) >= total / 2}
    if "1" not in data or len(data) < 2:
        return None
    keys = A.common_set(data)
    lines = [
        {"w": f"{float(w):g}", **{k: [round(x, 4) for x in v] for k, v in A.curves(d, keys).items()}}
        for w, d in sorted(data.items(), key=lambda x: float(x[0]))
    ]
    return {"node": node, "scope": scope, "common": len(keys), "total": total, "lines": lines}


def _enc(r: A.Rescue) -> list:
    return [
        f"{r.key[0]} {r.key[1]}:{r.key[2]}",
        round(r.base_best, 2),
        r.base_hits,
        round(r.other_best, 2),
        r.other_hits,
    ]


def rescue_set(node: str, scope: str, cat: str) -> dict | None:
    total = A.TOTAL_INTERFACES[cat]
    base = A.load(series_for(node, scope, "1"), cat)
    if len(base) < total / 2:
        return None
    rows = []
    for w in RESCUE_W:
        other = A.load(series_for(node, scope, w), cat)
        if len(other) < total / 2:
            continue
        keys = A.common_set({"b": base, "o": other})
        res, lost = A.rescue(base, other, keys)
        rows.append(
            {
                "w": f"{float(w):g}",
                "n": len(keys),
                "fail": sum(A.oracle(base[k]) < A.SUCCESS for k in keys),
                "rescued": [_enc(r) for r in res],
                "lost": [_enc(r) for r in lost],
            }
        )
    return {"node": node, "scope": scope, "rows": rows} if rows else None


def points(cat: str) -> dict:
    """Per-interface (oracle, mean, conf) DockQ of every series on the page, for the xy scatter.

    Interfaces are the union over series; a series lacking one stores null there. Values are
    rounded to 3 decimals to keep the page small.
    """
    names = [s for s, _ in BASE]
    for scope in ("full", "abag", "all"):
        for node in MODELS:
            names += [series_for(node, scope, w) for w in W_GRID]
    names = list(dict.fromkeys(names))
    data = {n: A.load(n, cat) for n in names}
    data = {n: d for n, d in data.items() if len(d) >= A.TOTAL_INTERFACES[cat] / 2}
    keys = sorted(set().union(*(set(d) for d in data.values()))) if data else []
    out = {}
    for n, d in data.items():
        out[n.partition(":")[0]] = [
            [round(A.oracle(d[k]), 3), round(A.mean(d[k]), 3), round(A.conf(d[k]), 3)] if k in d else None for k in keys
        ]
    return {"keys": [f"{k[0]} {k[1]}:{k[2]}" for k in keys], "series": out}


POOL_W = ["0.6", "0.8", "1", "2.0", "3.0"]


def many_samples(cat: str) -> dict | None:
    """best-of-k to 125 samples for each model's W = 1 x125 run, and W-pooling at the same budget.

    Pooling splits k evenly over the full-drop series at POOL_W (25 samples each, same seeds), so
    its curve exists at k = 5, 10, ..., 125. One interface set: those every series has in full.
    """
    big = {n: A.load(f"{MODELS[n][0]}-cfg-w1n125", cat, samples=125) for n in MODELS}
    big = {n: d for n, d in big.items() if len(d) >= A.TOTAL_INTERFACES[cat] / 2}
    if not big:
        return None
    pools = {n: [A.load(series_for(n, "full", w), cat) for w in POOL_W] for n in big}
    sets = [set(d) for d in big.values()] + [set(d) for ds in pools.values() for d in ds]
    keys = sorted(set.intersection(*sets))
    lines, pooled = [], []
    for n, d in big.items():
        c = A.curves(d, keys)
        lines.append({"node": n, "pass": [round(x, 2) for x in c["pass"]], "emax": [round(x, 4) for x in c["emax"]]})
        ks = list(range(5, 126, 5))
        pooled.append(
            {
                "node": n,
                "k": ks,
                "pass": [round(A.pooled_pass(pools[n], keys, k), 2) for k in ks],
                "emax": [round(A.pooled_emax(pools[n], keys, k), 4) for k in ks],
            }
        )
    return {"common": len(keys), "total": A.TOTAL_INTERFACES[cat], "pool_w": POOL_W, "lines": lines, "pooled": pooled}


def build() -> dict:
    tables, curves, rescues = [], [], []
    for cat, label, scopes in CATEGORIES:
        tables.append(
            {
                "id": f"base-{label}",
                "cat": label,
                "kind": "base",
                "scope": None,
                **table([row(s, n, "1", "none") for s, n in BASE], cat),
            }
        )
        for scope in scopes:
            rows = []
            for node in MODELS:
                rows += [row(series_for(node, scope, w), node, w, scope) for w in W_GRID[:10]]
                rows.append(row(series_for(node, scope, "1"), node, "1", "none"))
                rows += [row(series_for(node, scope, w), node, w, scope) for w in W_GRID[10:]]
            tables.append({"id": f"{scope}-{label}", "cat": label, "kind": "sweep", "scope": scope, **table(rows, cat)})
            for node in MODELS:
                if c := curve_set(node, scope, cat):
                    curves.append({"cat": label, **c})
                if r := rescue_set(node, scope, cat):
                    rescues.append({"cat": label, **r})
    pts = {label: points(cat) for cat, label, _ in CATEGORIES}
    many = {label: m for cat, label, _ in CATEGORIES if (m := many_samples(cat))}
    return {
        "built": time.strftime("%Y-%m-%d %H:%M KST"),
        "tables": tables,
        "curves": curves,
        "rescues": rescues,
        "points": pts,
        "many": many,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "report.html")
    a = ap.parse_args()
    payload = build()
    html = (HERE / "template.html").read_text().replace("/*__DATA__*/null", json.dumps(payload))
    a.out.write_text(html)
    for t in payload["tables"]:
        print(f"{t['id']}: {t['common']}/{t['total']} common")
    for r in payload["rescues"]:
        print(
            f"rescue {r['cat']} {r['scope']} {r['node']}: "
            + " ".join(f"W{x['w']} +{len(x['rescued'])}/-{len(x['lost'])}" for x in r["rows"])
        )


if __name__ == "__main__":
    main()
