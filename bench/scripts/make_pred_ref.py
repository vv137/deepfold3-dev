"""Build FoldBench ``prediction_reference.csv`` from our inference tree.

FoldBench's ``evaluate.py`` consumes, per algorithm, a CSV:

    pdb_id, seed, sample, ranking_score, prediction_path

We map one df3 checkpoint to one FoldBench "algorithm" named ``step_{N}``,
so the eval dir layout is::

    bench/runs/{series}/eval/step_{N}/prediction_reference.csv

walking ``bench/runs/{series}/pred/step_{N}/{pdb}/`` for
``seed-*_sample-*/model.cif`` and pulling each sample's confidence metrics from
the sibling ``{pdb}_summary_confidences.json`` (written by inference-batch;
falls back to ``ranking_scores.csv`` when the JSON is missing).

The canonical FoldBench 5-tuple (pdb_id, seed, sample, ranking_score,
prediction_path) is written first; three extra confidence columns
(``ptm, iptm, mean_plddt``) are appended so ``eval_ost`` can carry them into
the per-sample CSV and select the confidence top-1 sample. Only the summary
JSON exposes ``mean_plddt`` (pLDDT), so it is the preferred source.

Usage::

    python bench/scripts/make_pred_ref.py --series df3v1 --step 90000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# DF3_BENCH: bench tree (runs/, data/) to read and write; defaults to this checkout's.
BENCH = Path(os.environ["DF3_BENCH"]) if os.environ.get("DF3_BENCH") else REPO / "bench"
_SEED_SAMPLE = re.compile(r"seed-(\d+)_sample-(\d+)")

# Confidence fields carried per (seed, sample) beyond the FoldBench 5-tuple.
CONF_FIELDS = ("ptm", "iptm", "mean_plddt")


def _fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def confidence_map(target_dir: Path) -> dict[tuple[int, int], dict]:
    """(seed, sample) -> {ranking_score, ptm, iptm, mean_plddt} for a target.

    Prefers ``{pdb}_summary_confidences.json`` (the only source with
    ``mean_plddt``); falls back to ``ranking_scores.csv`` (no pLDDT) so older
    prediction trees still yield ranking_score/ptm/iptm. Missing values are
    ``None``.
    """
    out: dict[tuple[int, int], dict] = {}
    keys = ("ranking_score", *CONF_FIELDS)

    js = target_dir / f"{target_dir.name}_summary_confidences.json"
    if js.exists():
        try:
            preds = json.load(open(js)).get("predictions", [])
        except (json.JSONDecodeError, OSError):
            preds = []
        for p in preds:
            try:
                sk = (int(p["seed"]), int(p["sample"]))
            except (KeyError, ValueError, TypeError):
                continue
            out[sk] = {k: _fnum(p.get(k)) for k in keys}
        if out:
            return out

    csv_path = target_dir / "ranking_scores.csv"
    if csv_path.exists():
        for r in csv.DictReader(open(csv_path)):
            try:
                sk = (int(r["seed"]), int(r["sample"]))
            except (KeyError, ValueError):
                continue
            out[sk] = {k: _fnum(r.get(k)) for k in keys}
    return out


def build(series: str, step: int, category: str) -> Path:
    pred_root = BENCH / "runs" / series / "pred" / category / f"step_{step}"
    eval_dir = BENCH / "runs" / series / "eval" / category / f"step_{step}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_csv = eval_dir / "prediction_reference.csv"

    rows: list[dict] = []
    n_targets = 0
    for target_dir in sorted(pred_root.iterdir()):
        if not target_dir.is_dir() or target_dir.name.startswith("_"):
            continue
        pdb_id = target_dir.name
        cmap = confidence_map(target_dir)
        found = False
        for sample_dir in sorted(target_dir.glob("seed-*_sample-*")):
            m = _SEED_SAMPLE.search(sample_dir.name)
            if not m:
                continue
            seed, sample = int(m.group(1)), int(m.group(2))
            cif = sample_dir / "model.cif"
            if not cif.exists():
                continue
            conf = cmap.get((seed, sample), {})
            rs = conf.get("ranking_score")
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "seed": seed,
                    "sample": sample,
                    "ranking_score": rs if rs is not None else 0.0,
                    "prediction_path": str(cif.resolve()),
                    **{k: ("" if conf.get(k) is None else conf[k]) for k in CONF_FIELDS},
                }
            )
            found = True
        n_targets += int(found)

    fieldnames = ["pdb_id", "seed", "sample", "ranking_score", "prediction_path", *CONF_FIELDS]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[pred_ref] {series} step {step}: {n_targets} targets, {len(rows)} predictions -> {out_csv}")
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="df3v1")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--category", default="monomer_protein")
    args = ap.parse_args()
    build(args.series, args.step, args.category)


if __name__ == "__main__":
    main()
