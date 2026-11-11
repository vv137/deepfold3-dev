"""``OptimConfig.msa_row_chunk``: same output as the whole tensor, inference only."""

from __future__ import annotations

import pytest
import torch

from deepfold3.model.attention import MSAAttention
from deepfold3.model.config import OptimConfig
from deepfold3.model.ops import OuterProductMean
from deepfold3.model.primitives import map_row_chunks

S, N, C_MSA, C_PAIR = 11, 6, 64, 128


def _randomize(module: torch.nn.Module) -> torch.nn.Module:
    torch.manual_seed(0)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return module.eval()


def _inputs():
    torch.manual_seed(1)
    msa = torch.randn(1, S, N, C_MSA)
    mask = torch.rand(1, S, N) > 0.2
    return msa, mask, torch.randn(1, N, N, C_PAIR)


def test_map_row_chunks_matches_the_whole_tensor():
    x = torch.randn(2, 7, 3, 4)
    with torch.no_grad():
        assert torch.equal(map_row_chunks(lambda r: r * 2, x, 3), x * 2)


def test_map_row_chunks_is_whole_tensor_with_grad_enabled():
    calls = []

    def fn(r):
        calls.append(r.shape[-3])
        return r

    map_row_chunks(fn, torch.zeros(1, 7, 3, 4), 3)
    assert calls == [7]


@pytest.mark.parametrize("chunk", [1, 4, 10])
def test_row_attention_chunked_equals_whole(chunk):
    whole = _randomize(MSAAttention(C_MSA, C_PAIR, optim=OptimConfig()))
    rows = MSAAttention(C_MSA, C_PAIR, optim=OptimConfig(msa_row_chunk=chunk)).eval()
    rows.load_state_dict(whole.state_dict())
    msa, mask, pair = _inputs()
    with torch.no_grad():
        assert torch.allclose(whole(msa, mask, pair), rows(msa, mask, pair), atol=1e-5)


@pytest.mark.parametrize("chunk", [1, 4, 10])
def test_outer_product_chunked_equals_whole(chunk):
    whole = _randomize(OuterProductMean(C_MSA, C_PAIR))
    rows = OuterProductMean(C_MSA, C_PAIR).eval()
    rows.load_state_dict(whole.state_dict())
    rows._row_chunk = chunk
    msa, mask, _ = _inputs()
    with torch.no_grad():
        assert torch.allclose(whole(msa, mask), rows(msa, mask), atol=1e-4)


def test_chunking_is_ignored_with_grad_enabled():
    module = _randomize(MSAAttention(C_MSA, C_PAIR, optim=OptimConfig(msa_row_chunk=4))).train()
    msa, mask, pair = _inputs()
    module(msa.requires_grad_(), mask, pair).sum().backward()
    assert msa.grad is not None


def test_every_chunk_has_one_shape():
    """A short last chunk would be a new Triton autotune key at every MSA depth."""
    calls = []

    def fn(rows):
        calls.append(rows.shape[-3])
        return rows * 2

    x = torch.randn(1, 7, 3, 4)
    with torch.no_grad():
        out = map_row_chunks(fn, x, 3)
    assert calls == [3, 3, 3]
    assert torch.equal(out, x * 2)
