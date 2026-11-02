"""Softmax Collapse tests for :mod:`deepfold3.training.stable_softmax`.

Each test asserts the property the paper (arXiv:2610.31642 §2) says standard
implementations lose past the precision cliff, and checks that the stock
``torch.nn.functional`` form really does lose it — so the test would fail on
the old form and passes on the new one.

Run:
    uv run pytest tests/test_stable_softmax.py -v
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from deepfold3.training.stable_softmax import cross_entropy, log_softmax, logit_margin, softmax

# Precision cliffs c_d from the paper, Figure 1a.
CLIFF = {torch.bfloat16: 5.5, torch.float32: 16.6, torch.float64: 36.7}


def _margin_logits(m: float, k: int, dtype: torch.dtype) -> torch.Tensor:
    """Logits whose exact logit margin is ``m`` over ``k`` classes."""
    # z_t = m + log(k - 1), others 0  =>  m = z_t - log sum_{j != t} e^0.
    z = torch.zeros(k, dtype=torch.float64)
    z[0] = m + math.log(k - 1)
    return z.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradients_sum_to_zero_past_the_cliff(dtype):
    """``sum_k g_k == 0`` is the invariant Softmax Collapse breaks."""
    m = CLIFF[dtype] + 4.0
    z = _margin_logits(m, 8, dtype).requires_grad_(True)
    cross_entropy(z, torch.tensor(0)).backward()
    g = z.grad

    assert g.sum().abs() <= 8 * torch.finfo(dtype).eps * g.abs().max(), (
        f"fixed CE gradients do not sum to zero: {g.sum().item():.3e}"
    )

    # The stock form does break it, so this test is not vacuous.
    zs = _margin_logits(m, 8, dtype).requires_grad_(True)
    F.cross_entropy(zs.unsqueeze(0), torch.tensor([0])).backward()
    assert zs.grad[0] == 0.0, "stock CE unexpectedly kept the target gradient"
    assert zs.grad.sum() > 0.0, "stock CE unexpectedly summed to zero"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_target_gradient_tracks_minus_exp_minus_m(dtype):
    """``g_t = -e^-m / (1 + e^-m)`` must survive well past the cliff."""
    for m in (CLIFF[dtype] + 2.0, CLIFF[dtype] + 10.0, 60.0):
        z = _margin_logits(m, 4, dtype).requires_grad_(True)
        cross_entropy(z, torch.tensor(0)).backward()
        expected = -math.exp(-m) / (1.0 + math.exp(-m))
        assert z.grad[0] != 0.0, f"target gradient zeroed at m={m}"
        assert z.grad[0].item() == pytest.approx(expected, rel=1e-5), (
            f"m={m}: got {z.grad[0].item():.6e}, want {expected:.6e}"
        )


def test_loss_value_does_not_round_to_zero():
    """``L = softplus(-m)`` stays positive where ``log(1 + e^-m)`` rounds to 0."""
    m = 25.0  # past c_32 = 16.6, far from exponent underflow at ~88
    z = _margin_logits(m, 4, torch.float32)
    assert cross_entropy(z, torch.tensor(0)).item() == pytest.approx(math.exp(-m), rel=1e-4)
    assert F.cross_entropy(z.unsqueeze(0), torch.tensor([0])).item() == 0.0


def test_matches_stock_below_the_cliff():
    """Below the cliff the fixed form must agree with the stock one."""
    torch.manual_seed(0)
    z = torch.randn(64, 32, dtype=torch.float32) * 2.0
    t = torch.randint(0, 32, (64,))
    torch.testing.assert_close(log_softmax(z), F.log_softmax(z, dim=-1), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(softmax(z), F.softmax(z, dim=-1), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(cross_entropy(z, t), F.cross_entropy(z, t, reduction="none"), atol=1e-6, rtol=1e-5)


def test_log_softmax_argmax_gradient_survives():
    """The standalone softmax has the same bug at its argmax slot."""
    m = 25.0
    z = _margin_logits(m, 4, torch.float32).requires_grad_(True)
    softmax(z)[0].backward()
    assert z.grad[0] != 0.0, "fixed softmax zeroed the argmax gradient"
    assert z.grad.sum().abs() < 1e-12

    zs = _margin_logits(m, 4, torch.float32).requires_grad_(True)
    F.softmax(zs, dim=-1)[0].backward()
    assert zs.grad[0] == 0.0, "stock softmax unexpectedly kept the argmax gradient"


def test_logit_margin_is_exact():
    """The margin diagnostic must reproduce the margin it was built from."""
    for m in (-3.0, 0.0, 5.5, 16.6, 40.0):
        z = _margin_logits(m, 16, torch.float64)
        assert logit_margin(z, torch.tensor(0)).item() == pytest.approx(m, abs=1e-9)
