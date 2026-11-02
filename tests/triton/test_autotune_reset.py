"""Autotuning must not leak into a kernel's result.

The benchmark runs a kernel on the caller's real arguments many times before the
call that counts, so a kernel that accumulates into its outputs returned the sum
of every run. The first backward on an untuned shape was wrong by a factor of
hundreds (input gradient cosine 0.04 against stock) and the next was correct.
"""

from __future__ import annotations

import pytest
import torch

from deepfold3.model.triton.cache import _reset_buffers_hook


def test_hook_zeroes_buffers_and_skips_unused_outputs():
    x = torch.ones(3)
    buf = torch.ones(3)
    # dBIAS is the int N when there is no bias; GRAD_MASK is x when there is no mask.
    kwargs = {"X": x, "dB": buf, "dBIAS": 7, "GRAD_MASK": x}
    _reset_buffers_hook(["dB", "dBIAS", "GRAD_MASK"])(kwargs)
    assert buf.eq(0).all()
    assert x.eq(1).all(), "an aliased input must survive"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernels need a GPU")
def test_cold_autotune_backward_matches_warm():
    from deepfold3.model.config import OptimConfig
    from deepfold3.model.ops import TriangleMulOutgoing
    from deepfold3.model.triton import fused_gated_dual_gemm as G
    from deepfold3.model.triton import layernorm_linear as L

    torch.manual_seed(0)
    m = TriangleMulOutgoing(128, optim=OptimConfig(use_fused_triangle_mul=True)).cuda().train()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, 0.05)
    z = torch.randn(1, 48, 48, 128, device="cuda")
    mask = torch.ones(1, 48, 48, device="cuda")
    d_out = torch.randn_like(z)

    def grad() -> torch.Tensor:
        m.zero_grad(set_to_none=True)
        x = z.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(x, mask)
        out.backward(d_out.to(out.dtype))
        return x.grad.float()

    warm = grad()
    # Forget the tuned configs, so the next call benchmarks again.
    L._first_backward_kernel.cache.clear()
    G._fused_dual_gemm_bwd_pregemm_kernel.cache.clear()
    cold = grad()
    torch.testing.assert_close(cold, warm, atol=1e-2 * float(warm.abs().max()), rtol=1e-1)
