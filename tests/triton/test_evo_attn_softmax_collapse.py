"""Softmax Collapse in the fused evo-attention backward (arXiv:2610.31642 §2).

**Fixed**: ``_attn_bwd_d`` now supplies the paper's eq. 5 centring, so these two
tests pass and stay in the suite as the regression gate. Recomputing ``D`` in
fp32 — the obvious fix, and the one the tier list below used to call sufficient —
was measured on GPU and is *not*: see tier 1 at the bottom.

``test_evo_attn_robustness.py::test_extreme_attention_concentration`` compares
the fused kernel's winner-take-all backward against the fp32 eager reference
and dismisses the 0.81 dQ fail_frac as "both kernel and ref at noise floor".
That reference is itself degraded, so it cannot settle the question. These
tests use an **fp64 oracle** instead.

The kernel's backward forms, per query row,

    dS_j = P_j * (dP_j - D),   D = rowsum(dO * O),   dP_j = dO . V_j

which is the same ``1 + e^-m`` structure the paper describes: at the argmax
slot ``dP_imax`` and ``D`` agree to ``O(e^-m)``, so their difference is pure
cancellation. The kernel makes it strictly worse than fp32 by sourcing ``D``
from the forward's ``O`` rather than from the fp32 ``P`` and ``dP`` it has
in hand. Two independent bf16 roundings on that path each suffice to kill it,
measured at margin 14.45 (fraction of rows whose ``dS_imax`` is *exactly* 0):

    D = sum_j p_j dP_j, all fp32          0.4%   <- survives
    P cast to V.dtype before the PV dot  77.2%
    O stored in V.dtype                  86.9%
    both (what the kernel does)          86.9%

So fixing only the ``O`` store is **not** sufficient; the forward's
``P_block.to(V_block.dtype)`` poisons ``D`` on its own. Once the non-winner
mass falls below bf16 ULP, ``D`` collapses onto ``dP_imax``, ``dS_imax`` is
exactly 0 while the other ``K - 1`` entries stay non-zero, and the rows stop
summing to zero. The kernel therefore collapsed at the **bf16** cliff
``c_16 ~= 5.5`` even though every arithmetic op after the load is fp32,
whereas the eager fp32 reference survives to ``c_32 ~= 16.6``.

Two tiers of fix were characterised. Only the second one works here:
  1. **Measured insufficient.** Recompute ``D = sum_j P_j dP_j`` in the
     backward from the fp32 ``P`` it already rebuilds. That is what lifts an
     *eager* reference to the fp32 cliff 16.6, but the kernel does not hold
     ``P``: it rebuilds it as ``exp2((QK - M) / ln 2)`` from the fp32 LSE, so
     its ``P`` is the true softmax scaled by ``1 +- ulp(|QK|)``. At this test's
     spike height ``|QK| ~ 20`` that ulp is 2.4e-6 against ``e^-14.45 =
     5.3e-7``: ``P_imax`` is *exactly* 1.0 and ``sum_j P_j`` misses 1 by ~1e-6,
     both of which the uncentred ``dP_j - D`` multiplies by ``D = O(1)``.
     Measured on GPU with ``D`` recomputed in fp32: 40.1% of rows still exactly
     zero, median retention 0.000, row sums 18.4x their own peak element.
  2. **Applied.** Centre ``dP`` on the argmax, the paper's eq. 5 move:
     ``u_j = dP_j - dP_imax`` (forced to exactly 0 at imax by index),
     ``dS_j = P_j (u_j - sum_i P_i u_i)``. Nothing is ever subtracted at O(1)
     scale, and the LSE's normalisation drift now couples only to
     ``(D - dP_imax) = O(e^-m)``. ``_attn_bwd_d`` emits the triple
     ``(dP_imax, imax, C)`` per row in one KV pass — no forward change, no
     argmax tensor from the forward — and the three consumers each spend one
     ``tl.where``. 2e-14 is the paper's fp64 figure; in fp32, with
     ``d_pair_bias`` returned cast to the grad dtype, the row-sum invariant
     bottoms out at bf16 rounding (~2e-3 of the peak), which is what this file
     checks.

Run:
    pytest tests/triton/test_evo_attn_softmax_collapse.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA not available - fused Triton kernel requires GPU", allow_module_level=True)

try:
    from deepfold3.model.triton.evo_attention import HAS_TRITON, _EvoformerAttention
except Exception as e:  # pragma: no cover
    pytest.skip(f"Could not import fused evo attention: {e}", allow_module_level=True)

if not HAS_TRITON:
    pytest.skip("Triton not available", allow_module_level=True)


# Shape and seeds of the dismissed failure, so the numbers are comparable to
# tests/triton/_failures/extreme_attention_concentration.json.
B, N_SEQ, N_RES, H, DIM = 1, 2, 256, 8, 32
SEED = 42


def _wta_inputs(spike: float, dtype=torch.bfloat16):
    """Winner-take-all setup: ``+spike`` on one pair_bias column per head."""
    torch.manual_seed(SEED)
    qs = 1.0 / math.sqrt(DIM)
    shape = (B, N_SEQ, N_RES, H, DIM)
    Q = (torch.randn(shape, device="cuda") * qs).to(dtype).detach().requires_grad_(True)
    K = (torch.randn(shape, device="cuda") * qs).to(dtype).detach().requires_grad_(True)
    V = torch.randn(shape, device="cuda").to(dtype).detach().requires_grad_(True)
    res_mask = torch.zeros((B, N_SEQ, 1, 1, N_RES), device="cuda", dtype=torch.float32)

    torch.manual_seed(SEED + 17)
    pair_bias = torch.randn((B, 1, H, N_RES, N_RES), device="cuda", dtype=torch.float32) * 0.1
    for h in range(H):
        pair_bias[:, :, h, :, (h * 37) % N_RES] += spike
    pair_bias = pair_bias.contiguous().detach().requires_grad_(True)

    torch.manual_seed(SEED + 23)
    dO = (torch.randn(shape, device="cuda") * 0.1).to(dtype)
    return Q, K, V, res_mask, pair_bias, dO


def _oracle(Q, K, V, res_mask, pair_bias, dO):
    """fp64 ground truth for (dQ, d_pair_bias) and the mean logit margin."""
    q = Q.detach().double().transpose(-2, -3).requires_grad_(True)
    k = K.detach().double().transpose(-2, -3)
    v = V.detach().double().transpose(-2, -3)
    s = (q @ k.transpose(-1, -2)) * DIM**-0.5 + pair_bias.detach().double() + res_mask.double()
    s.retain_grad()
    out = torch.softmax(s, dim=-1) @ v
    (dq, ds) = torch.autograd.grad(out, [q, s], grad_outputs=dO.detach().double().transpose(-2, -3))

    sd = s.detach()
    imax = sd.argmax(-1, keepdim=True)
    margin = (sd.gather(-1, imax) - torch.logsumexp(sd.scatter(-1, imax, -torch.inf), -1, keepdim=True)).mean()
    return dq.transpose(-2, -3), ds, margin.item(), imax


def _kernel_grads(Q, K, V, res_mask, pair_bias, dO):
    out = _EvoformerAttention.apply(Q, K, V, res_mask, pair_bias, False)
    dq, db = torch.autograd.grad(out, [Q, pair_bias], grad_outputs=dO)
    torch.cuda.synchronize()
    return dq, db


def test_margin_of_the_dismissed_failure_is_not_twenty():
    """``+20`` on one column is a margin of ~14.5, not 20.

    The margin is measured against the log-sum-exp of all 255 other logits,
    which costs ``log(255) ~= 5.5``. 14.5 sits below the fp32 cliff 16.6 and
    far above the bf16 cliff 5.5 — which is exactly why the eager fp32
    reference still holds most of the signal while the kernel does not.
    """
    *inp, dO = _wta_inputs(20.0)
    _, _, margin, _ = _oracle(*inp, dO)
    assert 13.0 < margin < 16.0, f"margin {margin:.2f} outside the expected ~14.5"
    assert margin < 16.6, "margin should sit below the fp32 cliff"


def test_true_dq_is_small_but_not_noise():
    """The true dQ at the dismissed setting is ~1e-7, not 0 and not ~e^-20.

    ``max_abs_ref`` in the recorded failure is 1.195e-07. If that were a noise
    floor the fp64 oracle would be orders of magnitude below it; it is not.
    """
    *inp, dO = _wta_inputs(20.0)
    dq_true, _, _, _ = _oracle(*inp, dO)
    peak = dq_true.abs().max().item()
    assert peak > 1e-8, f"true dQ peak {peak:.3e} — the quantity really is negligible"
    assert 3e-8 < peak < 1e-6, f"true dQ peak {peak:.3e} not the ~1.2e-7 the JSON records as ref"


def test_kernel_dq_matches_fp64_oracle_under_concentration():
    """dQ must track the fp64 oracle, not collapse to a fraction of it."""
    *inp, dO = _wta_inputs(20.0)
    dq_true, _, _, _ = _oracle(*inp, dO)
    dq_k, _ = _kernel_grads(*inp, dO)

    scale = dq_true.abs().max()
    err = (dq_k.double() - dq_true).abs()
    assert err.max() <= 0.1 * scale, (
        f"dQ error {err.max().item():.3e} vs true peak {scale.item():.3e} (kernel peak {dq_k.abs().max().item():.3e})"
    )


def test_rows_of_d_pair_bias_sum_to_zero():
    """The paper's invariant: softmax gradients must sum to zero over classes.

    ``d_pair_bias`` *is* the raw dS (summed over the MSA axis, which preserves
    the invariant), so this reads the invariant straight off a kernel output
    with no index plumbing. The oracle sums to zero to machine precision. Before
    the fix the kernel did not, because one entry per row — the winner's — had
    been zeroed while the rest stayed put.

    The kernel's floor is not zero but bf16: ``d_pair_bias`` is returned cast to
    the grad dtype, so a satisfied invariant shows up as a row sum at bf16
    rounding noise, two to three orders below the peak, not at 7x above it.
    """
    *inp, dO = _wta_inputs(20.0)
    _, ds_true, _, _ = _oracle(*inp, dO)
    _, db_k = _kernel_grads(*inp, dO)

    # Sanity: the oracle really does satisfy the invariant at this margin.
    assert ds_true.sum(-1).abs().max() <= 1e-6 * ds_true.abs().max()

    rowsum = db_k.sum(-1).abs().max()
    peak = db_k.abs().max()
    assert rowsum <= 0.05 * peak, (
        f"dS rows sum to {rowsum.item():.3e} against peak {peak.item():.3e} — "
        "gradients no longer sum to zero (Softmax Collapse)"
    )
