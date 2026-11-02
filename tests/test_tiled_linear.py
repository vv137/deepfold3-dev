"""Tests for TiledLinear memory-efficient wrapper.

Validates numerical correctness, gradient accuracy, state dict transparency,
and mode switching for the tiled linear computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deepfold3.model.ops import OuterProductMean
from deepfold3.model.primitives import TransitionBlock
from deepfold3.model.tiled_linear import TiledAdaLinear, TiledLinear, _auto_shards
from tests._util import wake_zero_init


def _make_pair(channels=128, shards=3):
    """Create matched TransitionBlock and TiledLinear with shared weights."""
    # Wake the zero-init output projection first: TransitionBlock's linear2 is
    # zeros, so an untouched pair compares zero with zero and says nothing about
    # tiling. Both sides load the same woken weights.
    base = wake_zero_init(TransitionBlock(channels))
    tiled = TiledLinear(TransitionBlock(channels), num_shards=shards)
    tiled.load_state_dict(base.state_dict())
    return base, tiled


class TestForwardExact:
    """Forward pass produces bitwise identical results."""

    def test_forward_exact_fp32(self):
        base, tiled = _make_pair()
        base.eval()
        tiled.eval()
        x = torch.randn(16, 16, 128)
        with torch.no_grad():
            assert torch.equal(base(x), tiled(x))

    def test_forward_exact_bf16(self):
        base, tiled = _make_pair()
        base.eval().bfloat16()
        tiled.eval().bfloat16()
        x = torch.randn(16, 16, 128, dtype=torch.bfloat16)
        with torch.no_grad():
            assert torch.equal(base(x), tiled(x))

    def test_forward_exact_train(self):
        """Training mode forward is also bitwise identical."""
        base, tiled = _make_pair()
        base.train()
        tiled.train()
        x = torch.randn(16, 16, 128)
        # Use no_grad to test forward path without autograd function
        y_base = base(x)
        y_tiled = tiled(x.clone().detach().requires_grad_(True))
        assert torch.equal(y_base, y_tiled)


class TestBackwardGradients:
    """Backward pass produces correct gradients.

    Unlike the forward, the backward cannot be bitwise identical. Tiling splits
    the forward along dim -2 into shards whose rows are independent, so each
    output element is produced by the same reduction in the same order and
    ``TestForwardExact`` can and does demand ``torch.equal``. The input
    gradient is different in kind: it accumulates a contribution from every
    shard, so the tiled path sums shard by shard where the untiled path does
    one reduction. That is a reassociation, not a loss of precision, and it
    lands at the fp32 noise floor -- measured 2.4e-06 against a gradient whose
    peak is 5.2, i.e. ~4 ULP. Both gradient tests therefore use tolerances;
    ``test_param_grad_fp32`` already did, and said so in its comment.
    """

    def test_input_grad_fp32(self):
        base, tiled = _make_pair()
        base.train()
        tiled.train()
        # Seeded for the same reason test_param_grad_fp32 is: the comparison
        # sits at the summation-order floor, so an unseeded draw makes the
        # margin vary run to run.
        torch.manual_seed(0)
        x = torch.randn(16, 16, 128)
        x1 = x.clone().detach().requires_grad_(True)
        x2 = x.clone().detach().requires_grad_(True)

        base(x1).sum().backward()
        tiled(x2).sum().backward()
        torch.testing.assert_close(x1.grad, x2.grad, atol=1e-5, rtol=1e-4)

    def test_param_grad_fp32(self):
        base, tiled = _make_pair()
        base.train()
        tiled.train()
        # Seeded: these gradients now have real magnitude (the pair's output
        # projection is woken), so the fp32 summation-order difference between
        # the tiled and untiled paths lands near the tolerance and an unseeded
        # draw makes the test flaky.
        torch.manual_seed(0)
        x = torch.randn(16, 16, 128)

        base(x.clone().detach().requires_grad_(True)).sum().backward()
        tiled(x.clone().detach().requires_grad_(True)).sum().backward()

        for (n1, p1), (n2, p2) in zip(base.named_parameters(), tiled.named_parameters(), strict=True):
            assert n1 == n2, f"Parameter name mismatch: {n1} vs {n2}"
            torch.testing.assert_close(p1.grad, p2.grad, atol=1e-5, rtol=1e-4, msg=f"Grad mismatch for {n1}")

    def test_param_grad_bf16(self):
        base, tiled = _make_pair()
        base.train().bfloat16()
        tiled.train().bfloat16()
        x = torch.randn(16, 16, 128, dtype=torch.bfloat16)

        base(x.clone().detach().requires_grad_(True)).sum().backward()
        tiled(x.clone().detach().requires_grad_(True)).sum().backward()

        for (n1, p1), (_n2, p2) in zip(base.named_parameters(), tiled.named_parameters(), strict=True):
            # bf16 accumulation order differences are larger; use relative tolerance
            torch.testing.assert_close(p1.grad, p2.grad, atol=0.5, rtol=0.05, msg=f"Grad mismatch for {n1}")

    def test_gradcheck_against_untiled(self):
        """Autograd correctness, checked against the untiled module.

        This replaces a ``torch.autograd.gradcheck`` on the tiled module. That
        check only passed because ``TransitionBlock``'s output projection is
        zero-initialized, which made the Jacobian trivially zero; with the
        weights woken it fails — and so does a bare ``LayerNorm``, because
        ``primitives.LayerNorm`` always runs its forward in fp32. Under
        ``.double()`` the finite differences at eps=1e-6 are then swamped by
        fp32 rounding, so fp64 gradcheck cannot work on any module in this
        repo that uses it. Comparing analytic gradients against the untiled
        implementation tests what the original check was for — that sharding
        does not change the backward — and is valid in fp32.
        """
        base, tiled = _make_pair(channels=16, shards=2)
        base.train()
        tiled.train()
        xb = torch.randn(4, 4, 16, requires_grad=True)
        xt = xb.detach().clone().requires_grad_(True)
        base(xb).sum().backward()
        tiled(xt).sum().backward()
        torch.testing.assert_close(xb.grad, xt.grad, atol=1e-5, rtol=1e-4)
        for (name, pb), (_, pt) in zip(base.named_parameters(), tiled.named_parameters(), strict=True):
            torch.testing.assert_close(pb.grad, pt.grad, atol=1e-5, rtol=1e-4, msg=f"{name}")


class TestBatchDim:
    """Works with both unbatched and batched inputs."""

    def test_unbatched(self):
        base, tiled = _make_pair()
        base.eval()
        tiled.eval()
        x = torch.randn(16, 128)  # [N, C] — single track
        with torch.no_grad():
            assert torch.equal(base(x), tiled(x))

    def test_batched(self):
        base, tiled = _make_pair()
        base.eval()
        tiled.eval()
        x = torch.randn(2, 16, 16, 128)  # [B, N, N, C]
        with torch.no_grad():
            assert torch.equal(base(x), tiled(x))


class TestUnevenChunks:
    """N not divisible by num_shards."""

    def test_uneven_forward(self):
        base, tiled = _make_pair(shards=4)  # 16 / 4 = 4 exact
        base.eval()
        tiled.eval()
        x = torch.randn(17, 17, 128)  # 17 not divisible by 4
        with torch.no_grad():
            assert torch.equal(base(x), tiled(x))

    def test_uneven_backward(self):
        base, tiled = _make_pair(shards=4)
        base.train()
        tiled.train()
        x = torch.randn(17, 17, 128)
        x1 = x.clone().detach().requires_grad_(True)
        x2 = x.clone().detach().requires_grad_(True)
        base(x1).sum().backward()
        tiled(x2).sum().backward()
        # Uneven chunks differ from the untiled path by fp32 summation order.
        # The tolerance was set when TransitionBlock's zero-init output made
        # every gradient exactly zero, so no reordering error could show; with
        # the weights woken, 1 element in ~37k lands at 1.8e-6 absolute /
        # 2.6e-5 relative. Matched to the tolerance the other gradient
        # comparisons in this file already use.
        torch.testing.assert_close(x1.grad, x2.grad, atol=1e-5, rtol=1e-4)


class TestStateDictRoundtrip:
    """State dict keys match and load/save round-trips correctly."""

    def test_keys_match(self):
        base = wake_zero_init(TransitionBlock(128))
        tiled = TiledLinear(wake_zero_init(TransitionBlock(128)), num_shards=3)
        assert list(base.state_dict().keys()) == list(tiled.state_dict().keys())

    def test_cross_load(self):
        base = wake_zero_init(TransitionBlock(128))
        tiled = TiledLinear(wake_zero_init(TransitionBlock(128)), num_shards=3)
        tiled.load_state_dict(base.state_dict())
        for (n1, p1), (_n2, p2) in zip(base.named_parameters(), tiled.named_parameters(), strict=True):
            assert torch.equal(p1, p2), f"Parameter {n1} not equal after load"

    def test_parent_module_keys(self):
        """Keys have correct prefix when embedded in parent module."""

        class Parent(nn.Module):
            def __init__(self, tiled=False):
                super().__init__()
                tb = wake_zero_init(TransitionBlock(128))
                self.pair_transition = TiledLinear(tb, num_shards=3) if tiled else tb

        p_normal = Parent(tiled=False)
        p_tiled = Parent(tiled=True)
        assert list(p_normal.state_dict().keys()) == list(p_tiled.state_dict().keys())

        # Cross-load from normal to tiled
        p_tiled.load_state_dict(p_normal.state_dict())


class TestModeSwitch:
    """Tiling active in train mode, bypassed in eval mode."""

    def test_eval_bypasses_tiling(self):
        tiled = TiledLinear(wake_zero_init(TransitionBlock(128)), num_shards=3)
        tiled.eval()
        x = torch.randn(16, 16, 128, requires_grad=True)
        y = tiled(x)
        # In eval mode, output should be computed directly (no tiled autograd fn)
        assert y.shape == (16, 16, 128)

    def test_train_uses_tiling(self):
        tiled = TiledLinear(wake_zero_init(TransitionBlock(128)), num_shards=3)
        tiled.train()
        x = torch.randn(16, 16, 128, requires_grad=True)
        y = tiled(x)
        y.sum().backward()
        assert x.grad is not None

    def test_inference_tiling(self):
        """tile_inference=True chunks forward even in eval."""
        base = wake_zero_init(TransitionBlock(128))
        tiled = TiledLinear(wake_zero_init(TransitionBlock(128)), num_shards=3, tile_inference=True)
        tiled.load_state_dict(base.state_dict())
        base.eval()
        tiled.eval()
        x = torch.randn(16, 16, 128)
        with torch.no_grad():
            assert torch.equal(base(x), tiled(x))


class TestAutoShards:
    """Auto-shard heuristic returns sensible values."""

    def test_pair_128(self):
        x = torch.randn(384, 384, 128)
        assert _auto_shards(x) == 3  # ceil(384/128)

    def test_pair_64(self):
        x = torch.randn(384, 384, 64)
        assert _auto_shards(x) == 6  # ceil(384/64)

    def test_single(self):
        x = torch.randn(384, 384)
        assert _auto_shards(x) == 1  # ceil(384/384)

    def test_small(self):
        x = torch.randn(4, 4, 128)
        assert _auto_shards(x) == 1  # ceil(4/128) = 1

    def test_shard_1_bypasses(self):
        """When auto_shards returns 1, TiledLinear calls inner directly."""
        tiled = TiledLinear(wake_zero_init(TransitionBlock(128)), num_shards=0)  # auto
        tiled.train()
        # Small input → shards=1 → direct forward
        x = torch.randn(4, 4, 128, requires_grad=True)
        y = tiled(x)
        y.sum().backward()
        assert x.grad is not None


class TestTiledAdaLinear:
    """Tests for TiledAdaLinear (conditioning variant)."""

    def test_forward_match(self):
        from deepfold3.model.primitives import AdaTransitionBlock

        base = wake_zero_init(AdaTransitionBlock(128, 64))
        tiled = TiledAdaLinear(AdaTransitionBlock(128, 64), num_shards=3)
        tiled.load_state_dict(base.state_dict())
        base.eval()
        tiled.eval()

        x = torch.randn(16, 16, 128)
        cond = torch.randn(16, 16, 64)
        with torch.no_grad():
            assert torch.equal(base(x, cond), tiled(x, cond))

    def test_backward_match(self):
        from deepfold3.model.primitives import AdaTransitionBlock

        base = wake_zero_init(AdaTransitionBlock(128, 64))
        tiled = TiledAdaLinear(AdaTransitionBlock(128, 64), num_shards=3)
        tiled.load_state_dict(base.state_dict())
        base.train()
        tiled.train()

        x = torch.randn(16, 16, 128)
        cond = torch.randn(16, 16, 64)
        x1, c1 = x.clone().detach().requires_grad_(True), cond.clone().detach().requires_grad_(True)
        x2, c2 = x.clone().detach().requires_grad_(True), cond.clone().detach().requires_grad_(True)

        base(x1, c1).sum().backward()
        tiled(x2, c2).sum().backward()
        assert torch.equal(x1.grad, x2.grad)
        assert torch.equal(c1.grad, c2.grad)


def _make_opm(chunk_size=0):
    """Create OuterProductMean with optional chunking."""
    opm = wake_zero_init(OuterProductMean(64, 128, c_hidden=32))
    opm._chunk_size = chunk_size
    return opm


class TestTiledOPM:
    """Tests for tiled OuterProductMean (chunked outer product contraction)."""

    def test_forward_match_eval(self):
        """Chunked forward matches unchunked in eval mode."""
        base = _make_opm(0)
        tiled = _make_opm(4)
        tiled.load_state_dict(base.state_dict())
        base.eval()
        tiled.eval()

        msa = torch.randn(16, 12, 64)
        mask = torch.ones(16, 12)
        with torch.no_grad():
            y_base = base(msa, mask)
            y_tiled = tiled(msa, mask)
        # fp32 accumulation order differs slightly
        torch.testing.assert_close(y_base, y_tiled, atol=1e-4, rtol=1e-4)

    def test_forward_match_train(self):
        """Chunked forward matches unchunked in train mode."""
        base = _make_opm(0)
        tiled = _make_opm(4)
        tiled.load_state_dict(base.state_dict())
        base.train()
        tiled.train()

        msa = torch.randn(16, 12, 64)
        mask = torch.ones(16, 12)
        y_base = base(msa, mask)
        y_tiled = tiled(msa, mask)
        torch.testing.assert_close(y_base, y_tiled, atol=1e-4, rtol=1e-4)

    def test_backward_input_grad(self):
        """Input gradients match between chunked and unchunked."""
        base = _make_opm(0)
        tiled = _make_opm(4)
        tiled.load_state_dict(base.state_dict())
        base.train()
        tiled.train()

        msa = torch.randn(16, 12, 64)
        mask = torch.ones(16, 12)
        m1 = msa.clone().detach().requires_grad_(True)
        m2 = msa.clone().detach().requires_grad_(True)

        base(m1, mask).sum().backward()
        tiled(m2, mask).sum().backward()
        torch.testing.assert_close(m1.grad, m2.grad, atol=1e-4, rtol=1e-4)

    def test_backward_param_grad(self):
        """Parameter gradients match between chunked and unchunked."""
        base = _make_opm(0)
        tiled = _make_opm(4)
        tiled.load_state_dict(base.state_dict())
        base.train()
        tiled.train()

        msa = torch.randn(16, 12, 64)
        mask = torch.ones(16, 12)

        base(msa.clone().detach().requires_grad_(True), mask).sum().backward()
        tiled(msa.clone().detach().requires_grad_(True), mask).sum().backward()

        for (n1, p1), (_n2, p2) in zip(base.named_parameters(), tiled.named_parameters(), strict=True):
            if p1.grad is not None:
                torch.testing.assert_close(p1.grad, p2.grad, atol=1e-3, rtol=1e-3, msg=f"Grad mismatch for {n1}")

    def test_batched(self):
        """Works with batch dimension."""
        opm = _make_opm(4)
        opm.eval()
        msa = torch.randn(2, 16, 12, 64)
        mask = torch.ones(2, 16, 12)
        with torch.no_grad():
            y = opm(msa, mask)
        assert y.shape == (2, 12, 12, 128)

    def test_chunk_disabled(self):
        """chunk_size=0 uses original path."""
        opm = _make_opm(0)
        opm.eval()
        msa = torch.randn(16, 12, 64)
        mask = torch.ones(16, 12)
        with torch.no_grad():
            y = opm(msa, mask)
        assert y.shape == (12, 12, 128)
