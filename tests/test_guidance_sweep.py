"""``inference-batch --guidance-sweep``: one trunk per seed, many guidance settings."""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest
import torch

from deepfold3.scripts.inference_batch import (
    _all_summaries_on_disk,
    _load_sweep,
    _parse_inputs_file,
    _sweep_entries,
)
from tests.test_diffusion_guidance import _tiny_batch, _tiny_model


@pytest.fixture
def model():
    torch.manual_seed(0)
    return _tiny_model(use_conditioning_prob=0.8).eval()


def test_sample_from_trunk_is_forward(model):
    """Trunk + sampler split must give exactly what ``forward`` gives, guided or not."""
    from deepfold3.model.guidance import build_guidance_terms, parse_guidance_spec

    batch = _tiny_batch()
    with torch.no_grad():
        trunk = model.run_evoformer(batch, 0, seed=7)
        for specs in [(), (parse_guidance_spec("2.0"),)]:
            terms = build_guidance_terms(specs, batch["asym_id"], {"A": 0, "B": 1, "C": 2}, 6) if specs else None
            ref = model(batch, seed=7, num_recycles=0, num_samples=2, guidance_terms=terms)
            got = model.sample_from_trunk(batch, trunk, seed=7, num_samples=2, guidance_terms=terms)
            assert torch.equal(ref.atom_positions, got.atom_positions)
            assert torch.equal(ref.plddt, got.plddt)


def test_entries_sharing_a_trunk_start_from_the_same_noise(model):
    """The generator is re-seeded per call: the second entry from one trunk is
    not a continuation of the first entry's random stream."""
    batch = _tiny_batch()
    with torch.no_grad():
        trunk = model.run_evoformer(batch, 0, seed=3)
        a = model.sample_from_trunk(batch, trunk, seed=3, num_samples=2)
        b = model.sample_from_trunk(batch, trunk, seed=3, num_samples=2, distogram=a.distogram)
    assert torch.equal(a.atom_positions, b.atom_positions)
    assert b.distogram is a.distogram


def test_inputs_file_extras(tmp_path):
    f = tmp_path / "in.tsv"
    f.write_text(
        "/x/a.json\t/out/{cfg}/a\tgroup=abag\tseeds=42\tpairs=H:A,L:A\n/x/b.json /out/b\n# comment\n/x/c.json\n"
    )
    rows = _parse_inputs_file(f, "{json_dir}/inf", Path("p.pt"))
    assert rows[0] == (Path("/x/a.json"), Path("/out/{cfg}/a"), {"group": "abag", "seeds": "42", "pairs": "H:A,L:A"})
    assert rows[1] == (Path("/x/b.json"), Path("/out/b"), {})
    assert rows[2] == (Path("/x/c.json"), Path("/x/inf"), {})


def test_inputs_file_rejects_a_bare_extra(tmp_path):
    f = tmp_path / "in.tsv"
    f.write_text("/x/a.json\t/out\tabag\n")
    with pytest.raises(click.UsageError, match="key=value"):
        _parse_inputs_file(f, "", Path("p.pt"))


def _sweep(tmp_path):
    p = tmp_path / "sweep.json"
    p.write_text(
        json.dumps(
            {
                "groups": {
                    "full": [{"name": "full_w2", "guidance": ["2.0"]}],
                    "abag": [{"name": "abag_w0.5", "guidance": ["0.5@{pairs}"]}],
                }
            }
        )
    )
    return _load_sweep(p)


def test_sweep_entries_fill_cfg_and_pairs(tmp_path):
    groups = _sweep(tmp_path)
    ((name, od, specs),) = _sweep_entries(groups, Path("/o/{cfg}/t"), {"group": "abag", "pairs": "H:A,L:A"})
    assert name == "abag_w0.5" and od == Path("/o/abag_w0.5/t")
    assert specs[0].scale == 0.5 and specs[0].target == "pairs"
    assert [(str(a), str(b)) for a, b in specs[0].pairs] == [("H", "A"), ("L", "A")]
    ((_, _, full),) = _sweep_entries(groups, Path("/o/{cfg}/t"), {"group": "full"})
    assert full[0].target == "full" and full[0].scale == 2.0


@pytest.mark.parametrize(
    ("od", "extras", "match"),
    [
        ("/o/t", {"group": "full"}, "placeholder"),
        ("/o/{cfg}/t", {"group": "nope"}, "group="),
        ("/o/{cfg}/t", {"group": "abag"}, "pairs="),
    ],
)
def test_sweep_entries_errors(tmp_path, od, extras, match):
    with pytest.raises(click.UsageError, match=match):
        _sweep_entries(_sweep(tmp_path), Path(od), extras)


def test_summaries_on_disk_cover_every_complete_seed(tmp_path):
    """Per-seed jobs share a target dir; each rewrites the aggregates from every finished seed."""
    for seed, n in ((42, 2), (66, 1)):  # 66 is still being written
        for s in range(n):
            d = tmp_path / f"seed-{seed}_sample-{s}"
            d.mkdir()
            (d / "confidences.json").write_text(json.dumps({"summary": {"ranking_score": 0.1}}))
    rows = _all_summaries_on_disk(tmp_path, 2)
    assert {(r["seed"], r["sample"]) for r in rows} == {(42, 0), (42, 1)}
