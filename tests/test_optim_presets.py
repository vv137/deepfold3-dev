"""``--optim`` presets and the refusal of kernels that cannot run."""

import sys

import pytest
import torch

import deepfold3.model.triton as triton_pkg
from deepfold3.model import kernel_census
from deepfold3.model.config import OPTIM_PRESETS, OptimConfig
from deepfold3.model.inference_guard import NotActive
from deepfold3.model.ops import TriangleMulIncoming, TriangleMulOutgoing
from deepfold3.model.optim import make_transition_block


@pytest.fixture(autouse=True)
def _clean_census():
    kernel_census.reset()
    yield
    kernel_census.reset()


def test_every_preset_resolves():
    for name in OPTIM_PRESETS:
        assert isinstance(OptimConfig.from_preset(name, OptimConfig()), OptimConfig)


def test_train_matched_is_the_checkpoints_config():
    recorded = OptimConfig(use_fused_swiglu=True)
    assert OptimConfig.from_preset("train-matched", recorded) is recorded
    with pytest.raises(ValueError, match="train-matched"):
        OptimConfig.from_preset("train-matched", None)


def test_none_is_stock():
    cfg = OptimConfig.none()
    assert not any(v for k, v in vars(cfg).items() if k.startswith("use_"))


def test_cueq_preset_replaces_the_triton_trimul():
    cfg = OptimConfig.inference_cueq()
    assert cfg.use_cueq_triangle_mul and not cfg.use_channel_major_triangle_mul


def test_missing_triton_refuses(monkeypatch):
    monkeypatch.setattr(triton_pkg, "HAS_TRITON", False)
    with pytest.raises(NotActive, match="Triton is not installed"):
        OptimConfig.inference().require_backends()
    OptimConfig().require_backends()  # dropout_add defaults on; not a request


def test_missing_cueq_refuses(monkeypatch):
    monkeypatch.setitem(sys.modules, "cuequivariance_torch", None)
    with pytest.raises(NotActive, match="cueq dependency group"):
        TriangleMulOutgoing(32, optim=OptimConfig(use_cueq_triangle_mul=True))


def test_channel_gate_is_named():
    make_transition_block(256, config=OptimConfig(use_fused_transition_pair=True))
    assert "use_fused_transition_pair[C=256]" in kernel_census.inactive()
    kernel_census.reset_counts()
    assert "NOT ACTIVE use_fused_transition_pair[C=256]" in kernel_census.format_census()


@pytest.mark.parametrize("cls", [TriangleMulOutgoing, TriangleMulIncoming])
def test_cueq_matches_stock_definition(cls):
    """The kernel's own PyTorch path, which runs at this N, must equal our stock path."""
    pytest.importorskip("cuequivariance_ops_torch", exc_type=ImportError)
    torch.manual_seed(0)
    m = cls(32, optim=OptimConfig(use_cueq_triangle_mul=True)).eval()
    for lin in (m.proj, m.gate, m.out_proj, m.out_gate):
        torch.nn.init.normal_(lin.weight, std=0.3)
    z = torch.randn(1, 8, 8, 32)
    mask = (torch.rand(1, 8, 8) > 0.2).float()
    with torch.no_grad():
        got, want = m._cueq_forward(z, mask), m._compute(z, mask)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
