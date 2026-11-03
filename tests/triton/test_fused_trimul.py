"""Channel-major triangle multiply: forward and gradient against stock.

Run:
    uv run pytest tests/triton/test_fused_trimul.py -v
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from deepfold3.model import kernel_census
from deepfold3.model.config import OptimConfig
from deepfold3.model.inference_guard import InferenceOnlyKernelError
from deepfold3.model.ops import TriangleMulIncoming, TriangleMulOutgoing

pytestmark = pytest.mark.timeout(600)  # A cold cache compiles and autotunes four kernels per shape.
CHANNEL_MAJOR = OptimConfig(use_channel_major_triangle_mul=True)
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def build(cls, c, device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    stock = cls(c).to(device, dtype)
    for p in stock.parameters():
        if p.dim() == 2:
            torch.nn.init.normal_(p, std=1.0 / c**0.5)
    new = cls(c, optim=CHANNEL_MAJOR).to(device, dtype)
    new.load_state_dict(stock.state_dict())
    return stock, new


def pair(n, c, device="cpu", dtype=torch.float32, b=1):
    torch.manual_seed(1)
    z = torch.randn(b, n, n, c, device=device, dtype=dtype)
    mask = (torch.rand(b, n, n, device=device) > 0.1).to(dtype)
    return z, mask


@pytest.mark.parametrize("cls", [TriangleMulOutgoing, TriangleMulIncoming])
@pytest.mark.parametrize("c", [16, 64, 128])
def test_forward_and_gradient_match_stock(cls, c):
    stock, new = build(cls, c, dtype=torch.float64)
    z, mask = pair(24, c, dtype=torch.float64, b=2)
    weight = torch.randn_like(z)
    grads = []
    for m in (stock, new):
        x = z.clone().requires_grad_()
        out = m(x, mask)
        (out * weight).sum().backward()
        grads.append((out.detach(), x.grad, {k: p.grad for k, p in m.named_parameters()}))
    # Both LayerNorms take fp32 statistics, so fp32 rounding is the floor here.
    torch.testing.assert_close(grads[1][0], grads[0][0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(grads[1][1], grads[0][1], rtol=1e-4, atol=1e-5)
    for k, g in grads[0][2].items():
        torch.testing.assert_close(grads[1][2][k], g, rtol=1e-4, atol=1e-4, msg=k)


def test_mhr_heads_stay_on_the_channel_last_path():
    m = TriangleMulOutgoing(32, num_heads=4, head_norm="head_ln", optim=CHANNEL_MAJOR)
    assert not m._channel_major
    assert "use_channel_major_triangle_mul[C=32]" in kernel_census.inactive()


@cuda
@pytest.mark.parametrize("cls", [TriangleMulOutgoing, TriangleMulIncoming])
@pytest.mark.parametrize("c", [128, 256])
@pytest.mark.parametrize("b", [1, 2])
def test_kernels_match_stock_in_bf16(cls, c, b):
    stock, new = build(cls, c, "cuda")
    z, mask = pair(133, c, "cuda", b=b)
    with torch.no_grad():
        ref = stock(z, mask)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf16_stock = stock(z, mask).float()
            kernel_census.reset_counts()
            fused = new(z, mask).float()
            as_bf16 = new(z.bfloat16(), mask).float()
    assert kernel_census.snapshot()["TriangleMul"]["fused_trimul"] == 2
    bar = 2 * (bf16_stock - ref).pow(2).mean().sqrt()
    assert (fused - ref).pow(2).mean().sqrt() < bar
    assert (as_bf16 - ref).pow(2).mean().sqrt() < 2 * bar


@cuda
def test_fused_backward_matches_stock():
    for cls in (TriangleMulOutgoing, TriangleMulIncoming):
        for c, b in ((128, 1), (256, 1), (128, 2)):
            stock, new = build(cls, c, "cuda")
            z, mask = pair(133, c, "cuda", b=b)
            weight = torch.randn_like(z)
            grads = []
            for m, on in ((stock, False), (stock, True), (new, True)):
                m.zero_grad(set_to_none=True)
                x = z.clone().requires_grad_()
                kernel_census.reset_counts()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=on):
                    out = m(x, mask)
                (out.float() * weight).sum().backward()
                grads.append({"z": x.grad.float(), **{k: p.grad.float() for k, p in m.named_parameters()}})
            assert kernel_census.snapshot()["TriangleMul"] == {"fused_trimul_grad": 1}
            for k, ref in grads[0].items():
                errors = [(g[k] - ref).norm() / ref.norm().clamp_min(1e-30) for g in grads[1:]]
                assert errors[1] < 3 * errors[0] + 1e-3, k


@cuda
def test_autotuned_first_backward_equals_the_next():
    """Autotuning reruns the kernels, which must not accumulate into their outputs."""
    _, new = build(TriangleMulOutgoing, 128, "cuda")
    z, mask = pair(149, 128, "cuda")
    grads = []
    for _ in range(2):
        new.zero_grad(set_to_none=True)
        x = z.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            new(x, mask).float().pow(2).sum().backward()
        grads.append({"z": x.grad.clone(), **{k: p.grad.clone() for k, p in new.named_parameters()}})
    for k, g in grads[0].items():
        torch.testing.assert_close(g, grads[1][k], rtol=1e-2, atol=1e-2, msg=k)


@cuda
def test_second_backward_sees_the_same_saved_tensors():
    _, new = build(TriangleMulOutgoing, 128, "cuda")
    z, mask = pair(133, 128, "cuda")
    x = z.clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = new(x, mask)
    loss = out.float().pow(2).sum()
    first = torch.autograd.grad(loss, x, retain_graph=True)[0]
    torch.testing.assert_close(torch.autograd.grad(loss, x)[0], first, rtol=0, atol=0)


@cuda
def test_activation_checkpointing_gives_the_same_gradient():
    _, new = build(TriangleMulOutgoing, 128, "cuda")
    z, mask = pair(133, 128, "cuda")
    grads = []
    for run in (lambda x: new(x, mask), lambda x: checkpoint(new, x, mask, use_reentrant=False)):
        x = z.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = run(x)
        out.float().pow(2).sum().backward()
        grads.append(x.grad)
    torch.testing.assert_close(grads[1], grads[0], rtol=0, atol=0)


@cuda
def test_kernel_refuses_grad():
    from deepfold3.model.triton.fused_trimul import fused_trimul

    m = TriangleMulOutgoing(128).cuda()
    z, mask = pair(133, 128, "cuda")
    with pytest.raises(InferenceOnlyKernelError):
        fused_trimul(z, mask, True, m.norm_left, m.gate.weight, m.proj.weight, m.norm_center, m.out_proj.weight, m.out_gate.weight)
