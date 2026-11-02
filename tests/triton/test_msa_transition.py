"""Inference-only fused MSA transition: gating, numerics and flag-off identity.

Run:
    uv run pytest tests/triton/test_msa_transition.py -v
"""

from __future__ import annotations

import logging

import pytest
import torch

from deepfold3.model import optim as optim_module
from deepfold3.model.config import OptimConfig
from deepfold3.model.evoformer import EvoformerBlock
from deepfold3.model.inference_guard import InferenceOnlyKernelError
from deepfold3.model.primitives import TransitionBlock
from deepfold3.model.triton import msa_transition as mt

needs_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and mt.HAS_TRITON), reason="fused MSA transition needs CUDA and Triton"
)


def _block(channels: int, device: str = "cpu") -> TransitionBlock:
    torch.manual_seed(0)
    block = TransitionBlock(channels).to(device).eval()
    torch.nn.init.normal_(block.linear2.weight, std=0.02)
    return block


def _evoformer_block(optim: OptimConfig | None) -> EvoformerBlock:
    torch.manual_seed(0)
    return EvoformerBlock(64, 32, optim=optim).eval()


def _call(block: TransitionBlock, x: torch.Tensor, **kwargs) -> torch.Tensor:
    return mt.fused_msa_transition(
        x, block.norm.weight, block.norm.bias, block.linear1.weight, block.linear2.weight, **kwargs
    )


def test_kernel_raises_with_grad_enabled():
    block = _block(64)
    with pytest.raises(RuntimeError, match="inference-only"):
        _call(block, torch.zeros(4, 8, 64))


def test_flag_only_on_in_inference_preset():
    assert OptimConfig.inference().use_fused_msa_transition_inference
    assert not OptimConfig().use_fused_msa_transition_inference
    assert not OptimConfig.none().use_fused_msa_transition_inference


def test_state_dict_keys_unchanged_by_flag():
    off = OptimConfig.inference()
    off.use_fused_msa_transition_inference = False
    assert list(_evoformer_block(off).state_dict()) == list(_evoformer_block(OptimConfig.inference()).state_dict())


def test_flag_off_never_touches_kernel(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("kernel path reached with flag off")

    monkeypatch.setattr("deepfold3.model.evoformer.msa_transition_residual_", boom)
    block = _evoformer_block(OptimConfig())
    msa, pair = torch.randn(1, 3, 8, 64), torch.randn(1, 8, 8, 32)
    with torch.no_grad():
        block(msa, pair, torch.ones(1, 3, 8), torch.ones(1, 8, 8))


def _run_block(flag: bool, grad: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    block = _evoformer_block(OptimConfig(use_fused_msa_transition_inference=flag))
    torch.manual_seed(3)
    msa = torch.randn(1, 3, 8, 64, requires_grad=grad)
    pair = torch.randn(1, 8, 8, 32)
    with torch.set_grad_enabled(grad):
        out_msa, out_pair = block(msa, pair, torch.ones(1, 3, 8), torch.ones(1, 8, 8))
    if grad:
        out_msa.sum().backward()
    return out_msa.detach(), out_pair.detach(), msa.grad


def test_flag_off_is_identical_to_stock_statement():
    block = _evoformer_block(OptimConfig())
    captured = {}
    block.msa_transition.register_forward_hook(lambda m, i, o: captured.update(x=i[0], o=o))
    msa, pair = torch.randn(1, 3, 8, 64), torch.randn(1, 8, 8, 32)
    with torch.no_grad():
        out, _ = block(msa, pair, torch.ones(1, 3, 8), torch.ones(1, 8, 8))
    assert torch.equal(out, captured["x"] + block.residual_out_scale * captured["o"])


def test_grad_enabled_is_refused_with_flag_on():
    """The flag is inference-only: training must not reach it, not silently fall back."""
    block = _evoformer_block(OptimConfig(use_fused_msa_transition_inference=True))
    msa = torch.randn(1, 3, 8, 64, requires_grad=True)
    with pytest.raises(InferenceOnlyKernelError, match="fused_msa_transition"):
        block(msa, torch.randn(1, 8, 8, 32), torch.ones(1, 3, 8), torch.ones(1, 8, 8))


def test_fallback_reason_logged_once(monkeypatch, caplog):
    monkeypatch.setattr(mt, "_warned", set())
    block = _block(64)
    with caplog.at_level(logging.WARNING), torch.no_grad():
        for _ in range(3):
            assert mt.msa_transition_residual_(block, torch.zeros(2, 3, 64), 1.0) is None
    assert len([r for r in caplog.records if "stock path" in r.getMessage()]) == 1


def test_unsupported_width_is_named():
    with torch.no_grad():
        reason = mt.msa_transition_unsupported(_block(96), torch.zeros(2, 3, 96))
    assert reason is not None


def test_kernel_errors_propagate(monkeypatch):
    def oom(*args, **kwargs):
        raise torch.OutOfMemoryError("simulated")

    monkeypatch.setattr(mt, "msa_transition_unsupported", lambda block, x: None)
    monkeypatch.setattr(mt, "fused_msa_transition", oom)
    with torch.no_grad(), pytest.raises(torch.OutOfMemoryError):
        mt.msa_transition_residual_(_block(64), torch.zeros(2, 3, 64), 1.0)


def test_make_transition_block_unchanged():
    """The kernel is wired in EvoformerBlock only: pair and template blocks never see it."""
    cfg = OptimConfig.inference()
    assert type(optim_module.make_transition_block(128, config=cfg)) is type(
        optim_module.make_transition_block(128, config=OptimConfig(use_fused_transition_pair=True))
    )


@needs_gpu
@pytest.mark.parametrize("channels", [64, 128])
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_numerics_match_stock(channels: int, dtype: str):
    block = _block(channels, "cuda")
    torch.manual_seed(1)
    x = torch.randn(257, 311, channels, device="cuda")  # row count not a multiple of any tile
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "bf16"):
        ref = block(x).float()
        got = _call(block, x).float()
    assert got.shape == x.shape
    assert (got - ref).abs().max() <= (2e-2 if dtype == "bf16" else 5e-3) * ref.abs().max()


@needs_gpu
@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_inplace_residual_matches_out_of_place(scale: float):
    block = _block(64, "cuda")
    torch.manual_seed(2)
    x = torch.randn(33, 65, 64, device="cuda")
    with torch.no_grad():
        expected = x + scale * _call(block, x)
        result = mt.msa_transition_residual_(block, x, scale)
    assert result is x
    assert torch.allclose(result, expected, atol=1e-5, rtol=1e-4)


@needs_gpu
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_evoformer_block_flag_on_matches_off(dtype: str):
    off = OptimConfig.inference()
    off.use_fused_msa_transition_inference = False
    outputs = []
    for optim in (off, OptimConfig.inference()):
        torch.manual_seed(0)
        block = EvoformerBlock(64, 32, optim=optim).eval()
        torch.nn.init.normal_(block.msa_transition.linear2.weight, std=0.02)
        block.cuda()
        torch.manual_seed(5)
        msa, pair = torch.randn(1, 9, 16, 64, device="cuda"), torch.randn(1, 16, 16, 32, device="cuda")
        original = msa.clone()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "bf16"):
            out, _ = block(msa, pair, torch.ones(1, 9, 16, device="cuda"), torch.ones(1, 16, 16, device="cuda"))
        assert torch.equal(msa, original), "the caller's tensor must not be overwritten"
        outputs.append(out.float())
    assert (outputs[0] - outputs[1]).abs().max() <= 2e-2 * outputs[0].abs().max()


@needs_gpu
def test_inference_mode_tensor_is_served():
    block = _block(64, "cuda")
    with torch.inference_mode():
        x = torch.randn(5, 7, 64, device="cuda")
        assert mt.msa_transition_residual_(block, x, 1.0) is x
