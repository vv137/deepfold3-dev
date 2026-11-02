"""The kernel census records what ran, not what was requested."""

from __future__ import annotations

import pytest
import torch

from deepfold3.model import kernel_census
from deepfold3.model.attention import GridSelfAttention
from deepfold3.model.config import OptimConfig
from deepfold3.model.ops import OuterProductMean


@pytest.fixture(autouse=True)
def _clean():
    kernel_census.reset()
    yield
    kernel_census.reset()


def test_snapshot_counts_paths_per_site():
    kernel_census.record("site", "a")
    kernel_census.record("site", "a")
    kernel_census.record("site", "b")
    assert kernel_census.snapshot() == {"site": {"a": 2, "b": 1}}


def test_a_lever_is_shadowed_only_once_its_site_has_dispatched_elsewhere():
    kernel_census.requested("site", "fused")
    assert kernel_census.shadowed() == []  # nothing has run yet
    kernel_census.record("site", "stock")
    assert kernel_census.shadowed() == ["site:fused"]
    kernel_census.record("site", "fused")
    assert kernel_census.shadowed() == []


def test_reset_counts_keeps_the_levers_noted_at_construction():
    kernel_census.requested("site", "fused")
    kernel_census.record("site", "stock")
    kernel_census.reset_counts()
    assert kernel_census.snapshot() == {}
    kernel_census.record("site", "stock")
    assert kernel_census.shadowed() == ["site:fused"]


def test_cpu_attention_records_the_stock_paths():
    attn = GridSelfAttention(128, num_heads=4, optim=OptimConfig())
    attn(torch.randn(1, 8, 8, 128), torch.ones(1, 8, 8))
    assert kernel_census.snapshot() == {
        "GridSelfAttention.attention": {"baseline": 1},
        "GridSelfAttention.norm": {"stock": 1},
    }


def test_opm_records_row_chunking_and_tiling():
    opm = OuterProductMean(16, 32).eval()
    opm._chunk_size, opm._row_chunk = 3, 2
    with torch.no_grad():
        opm(torch.randn(1, 5, 8, 16), torch.ones(1, 5, 8))
    assert kernel_census.snapshot() == {
        "OuterProductMean.contract": {"tiled": 1},
        "OuterProductMean.project": {"row_chunked": 1},
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernels need a GPU")
@torch.no_grad()
def test_ln_transpose_with_evo_attn_matches_stock_column_attention():
    """The transpose kernel must stand aside for the deferred path, not double-transpose it."""
    stock_opt = OptimConfig(use_fused_evo_attention=True)
    lever_opt = OptimConfig(use_fused_evo_attention=True, use_fused_ln_transpose=True)
    stock = GridSelfAttention(128, num_heads=4, transpose=True, optim=stock_opt).cuda().eval()
    lever = GridSelfAttention(128, num_heads=4, transpose=True, optim=lever_opt).cuda().eval()
    torch.manual_seed(0)
    for p in stock.parameters():
        p.normal_(0.0, 0.05)
    lever.load_state_dict(stock.state_dict(), strict=False)
    z = torch.randn(1, 64, 64, 128, device="cuda")
    mask = torch.ones(1, 64, 64, device="cuda")
    mask[:, 50:, :] = 0.0  # asymmetric, so a transposed mask cannot pass by luck
    kernel_census.reset_counts()
    assert torch.equal(stock(z, mask), lever(z, mask))
    assert "fused_ln_transpose" not in kernel_census.snapshot()["GridSelfAttention.norm"]
    assert kernel_census.shadowed() == ["GridSelfAttention.norm:fused_ln_transpose"]
