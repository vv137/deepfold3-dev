"""Trunk-pair statistics for initialising ``DiffusionHead.null_pair`` (``cond_drop_fill: learned``).

Runs ``deepfold3 train``'s own setup (config, dataset mix, crops, ``--params``) but replaces the
training loop: each batch only goes through the trunk (``run_evoformer``, recycles drawn as in
training) and the trunk pair ``z_ij`` over valid token pairs is accumulated, separately for
intra-chain (same ``asym_id``) and inter-chain pairs:

- ``mean``: E[z_ij] (the direction a learned fill would imitate an average pair with),
- ``norm_q``: quantiles of ||z_ij|| (the scale; ``layer_norm`` over ``concat(z, rel_feat)`` makes
  the fill's size relative to ||rel_feat|| ~ 2 decide what the conditioning sees),
- ``rms``: per-channel RMS, ``mean_norm / median_norm`` (how far averaging shrinks the vector).

Writes ``<out>.json`` (summary) and ``<out>.pt`` (``mean_all``, ``mean_intra``, ``mean_inter``,
float32 [C]). Single process, one GPU, no gradients. Every ``deepfold3 train`` flag passes through:

    python scripts/null_pair_stats.py --batches 400 --out runs/null_pair_stats -- \\
        --config configs/stage3-s1-frozen-cfg-learned.yaml --output /tmp/x --params <shadow>.safetensors ...
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch


def _parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", type=int, default=400)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quantile-sample", type=int, default=200_000, help="pair norms kept for the quantiles")
    a, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    return a, rest


class _Acc:
    def __init__(self, c: int, device):
        self.sum = torch.zeros(c, dtype=torch.float64, device=device)
        self.sq = torch.zeros(c, dtype=torch.float64, device=device)
        self.n = 0
        self.norms: list[torch.Tensor] = []

    def add(self, z: torch.Tensor, keep: int):
        if not len(z):
            return
        z = z.double()
        self.sum += z.sum(0)
        self.sq += (z * z).sum(0)
        self.n += len(z)
        nrm = z.norm(dim=-1).float()
        if len(nrm) > keep:
            nrm = nrm[torch.randint(len(nrm), (keep,), device=nrm.device)]
        self.norms.append(nrm.cpu())

    def summary(self, cap: int) -> dict:
        mean = self.sum / max(self.n, 1)
        norms = torch.cat(self.norms) if self.norms else torch.zeros(1)
        if len(norms) > cap:
            norms = norms[torch.randperm(len(norms))[:cap]]
        q = torch.quantile(norms, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
        return {
            "pairs": self.n,
            "norm_q05_q25_q50_q75_q95": [round(float(x), 2) for x in q],
            "mean_norm": round(float(mean.norm()), 2),
            "mean_norm_over_median_norm": round(float(mean.norm() / q[2]), 4),
            "channel_rms": round(float((self.sq / max(self.n, 1)).mean().sqrt()), 2),
        }, mean.float().cpu()


def main():
    a, rest = _parse()
    import deepfold3.scripts.train as T
    import deepfold3.training.trainer as TR
    from deepfold3.training.step import prepare_features

    class StatsTrainer(TR.Trainer):
        def run(self, resume_from=None):  # noqa: ARG002 - same signature as Trainer.run
            self._init_distributed()
            self._seed_everything()
            self._init_memory_policy()
            self._init_model()
            train_dl, _ = self._init_dataloaders()
            model = self.model.eval()
            dev = self.dist.device
            acc = None
            rng = random.Random(self.cfg.seed)
            per_batch_keep = max(1, a.quantile_sample // max(a.batches, 1))
            seen = 0
            for batch in train_dl:
                feats = prepare_features({k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}, dev)
                with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
                    emb, _, _, seq_mask, _ = model.run_evoformer(
                        feats, rng.randint(0, self.cfg.max_recycles), seed=seen
                    )
                z = emb["pair"].float()
                c = z.shape[-1]
                if acc is None:
                    acc = {k: _Acc(c, dev) for k in ("all", "intra", "inter")}
                valid = seq_mask.bool()
                pv = valid.unsqueeze(-1) & valid.unsqueeze(-2)
                asym = feats["asym_id"]
                same = asym.unsqueeze(-1) == asym.unsqueeze(-2)
                acc["all"].add(z[pv], per_batch_keep)
                acc["intra"].add(z[pv & same], per_batch_keep)
                acc["inter"].add(z[pv & ~same], per_batch_keep)
                seen += 1
                if seen % 25 == 0:
                    print(f"[null_pair_stats] {seen}/{a.batches} batches", flush=True)
                if seen >= a.batches:
                    break
            out, means = {"batches": seen, "channels": c}, {}
            for k, v in acc.items():
                out[k], means[f"mean_{k}"] = v.summary(a.quantile_sample)
            out["cos_intra_inter"] = round(
                float(torch.nn.functional.cosine_similarity(means["mean_intra"], means["mean_inter"], dim=0)), 4
            )
            a.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(means, a.out.with_suffix(".pt"))
            a.out.with_suffix(".json").write_text(json.dumps(out, indent=1))
            print(json.dumps(out, indent=1))

    TR.Trainer = StatsTrainer  # train() imports Trainer from this module at call time
    sys.argv = ["deepfold3-train", *rest]
    T.train.main(args=rest, standalone_mode=False)


if __name__ == "__main__":
    main()
