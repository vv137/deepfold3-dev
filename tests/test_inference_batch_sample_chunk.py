"""``inference-batch --sample-chunk``: passes, per-pass seeds, numbering and resume."""

from __future__ import annotations

import json

import pytest
import torch

from deepfold3.scripts.inference_batch import (
    _existing_summaries,
    _seed_outputs,
    sample_chunk_seed,
    sample_chunks,
)
from tests.test_diffusion_guidance import _tiny_batch, _tiny_model


@pytest.mark.parametrize(
    ("n", "chunk", "expected"),
    [
        (5, None, [(0, 0, 5)]),
        (5, 5, [(0, 0, 5)]),
        (5, 9, [(0, 0, 5)]),
        (25, 5, [(0, 0, 5), (1, 5, 5), (2, 10, 5), (3, 15, 5), (4, 20, 5)]),
        (7, 3, [(0, 0, 3), (1, 3, 3), (2, 6, 1)]),
    ],
)
def test_sample_chunks_cover_every_sample_once(n, chunk, expected):
    got = sample_chunks(n, chunk)
    assert got == expected
    assert [i for _, start, size in got for i in range(start, start + size)] == list(range(n))


def test_chunk_zero_keeps_the_model_seed_and_later_chunks_differ():
    assert sample_chunk_seed(42, 0) == 42
    seeds = {sample_chunk_seed(s, c) for s in (42, 66, 101, 2024, 8888) for c in range(10)}
    assert len(seeds) == 50  # no two (seed, chunk) pairs share noise


def test_first_chunk_reproduces_an_unchunked_run_and_later_chunks_are_new_samples():
    """Chunk 0 draws with the model seed, so it is the run without --sample-chunk."""
    model = _tiny_model(use_conditioning_prob=0.8).eval()
    batch = _tiny_batch()
    with torch.inference_mode():
        trunk = model.run_evoformer(batch, 1, seed=7)
        full = model.sample_from_trunk(batch, trunk, seed=7, num_samples=2).atom_positions
        c0 = model.sample_from_trunk(batch, trunk, seed=sample_chunk_seed(7, 0), num_samples=2).atom_positions
        c1 = model.sample_from_trunk(batch, trunk, seed=sample_chunk_seed(7, 1), num_samples=2).atom_positions
    assert torch.equal(full, c0)
    assert not torch.allclose(c0, c1)


def _write(tmp_path, seed, samples):
    for s in samples:
        d = tmp_path / f"seed-{seed}_sample-{s}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.cif").write_text("data_x\n")
        (d / "confidences.json").write_text(json.dumps({"summary": {"ranking_score": float(s)}}))


def test_resume_is_per_chunk(tmp_path):
    """A run killed after chunk 0 owes only chunk 1; the skipped chunk is read back for the aggregates."""
    _write(tmp_path, 42, range(5))
    (c0, s0, n0), (c1, s1, n1) = sample_chunks(10, 5)
    assert all(p.exists() for p in _seed_outputs(tmp_path, 42, n0, False, s0))
    assert not any(p.exists() for p in _seed_outputs(tmp_path, 42, n1, False, s1))
    assert [p.parent.name for p in _seed_outputs(tmp_path, 42, n1, False, s1)][::2] == [
        f"seed-42_sample-{i}" for i in range(5, 10)
    ]
    rows = _existing_summaries(tmp_path, 42, n0, s0)
    assert [r["sample"] for r in rows] == [0, 1, 2, 3, 4]
