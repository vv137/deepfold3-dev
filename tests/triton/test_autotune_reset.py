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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernels need a GPU")
@pytest.mark.timeout(600)
def test_cold_autotune_ada_layernorm_backward_matches_warm():
    from deepfold3.model.triton import ada_layernorm_linear as A

    torch.manual_seed(0)
    x = torch.randn(4096, 64, device="cuda")
    scale, bias = torch.randn_like(x), torch.randn_like(x)
    weight = torch.randn(64, 128, device="cuda")
    d_out = torch.randn(4096, 128, device="cuda")
    m = A.FusedAdaLayerNormLinear(64, 128)

    def grads() -> list[torch.Tensor]:
        inputs = [t.clone().requires_grad_(True) for t in (x, scale, bias, weight)]
        m(*inputs).backward(d_out)
        return [t.grad.float() for t in inputs]

    warm = grads()
    A._ada_second_forward_kernel.cache.clear()
    A._ada_first_backward_kernel.cache.clear()
    for c, w in zip(grads(), warm, strict=True):
        torch.testing.assert_close(c, w, atol=1e-2 * float(w.abs().max()), rtol=1e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernels need a GPU")
@pytest.mark.timeout(600)
def test_cold_autotune_evo_attention_backward_matches_warm():
    from deepfold3.model.triton import evo_attention as E

    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 48, 4, 32, device="cuda") for _ in range(3))
    bias = torch.randn(1, 4, 48, 48, device="cuda")
    mask = torch.ones(2, 48, device="cuda")
    d_out = torch.randn_like(q)

    def grads() -> list[torch.Tensor]:
        inputs = [t.clone().requires_grad_(True) for t in (q, k, v, bias)]
        E.fused_evo_attention(*inputs, mask).backward(d_out)
        return [t.grad.float() for t in inputs]

    warm = grads()
    for kernel in (E._attn_fwd, E._attn_bwd_d, E._attn_bwd_dq, E._attn_bwd_db, E._attn_bwd_dk_dv):
        kernel.cache.clear()
    for c, w in zip(grads(), warm, strict=True):
        torch.testing.assert_close(c, w, atol=1e-2 * float(w.abs().max()), rtol=1e-1)
