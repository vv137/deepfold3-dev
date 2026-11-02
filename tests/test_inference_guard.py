"""Inference-only kernels refuse a backward, and say so."""

from __future__ import annotations

import pytest
import torch

from deepfold3.model import kernel_census
from deepfold3.model.inference_guard import InferenceOnlyKernelError, NotActive, inference_only


@pytest.fixture(autouse=True)
def _clean():
    kernel_census.reset()
    yield
    kernel_census.reset()


@inference_only("demo")
def _kernel(x: torch.Tensor) -> torch.Tensor:
    return x + 1


def test_refuses_with_grad_enabled_and_counts_it():
    with pytest.raises(InferenceOnlyKernelError, match="demo"):
        _kernel(torch.zeros(2))
    assert kernel_census.snapshot() == {"demo": {"refused:grad_enabled": 1}}


def test_runs_under_no_grad_even_for_a_parameter():
    with torch.no_grad():
        out = _kernel(torch.nn.Parameter(torch.zeros(2)))
    assert out.eq(1).all()
    assert kernel_census.snapshot() == {}


def test_not_active_names_the_kernel_and_is_counted():
    err = NotActive("demo", "uncovered_cell")
    assert str(err) == "NOT ACTIVE: demo: uncovered_cell"
    assert kernel_census.snapshot() == {"demo": {"refused:uncovered_cell": 1}}
