"""Refusal contract for inference-only kernels.

An inference-only kernel has no backward, and its numerics are measured on the
forward alone, so training must be structurally unable to reach it. A flag that
quietly falls back to another path hides which code produced a result, so these
kernels refuse by name instead. The discipline is borrowed from ``opt_core`` in
anthropics/uplifting-biomolecular-modeling (design only, no code copied).

:func:`inference_only` raises as soon as a backward could be built.
:class:`NotActive` names a kernel that cannot serve a call. Both are recorded in
``kernel_census``, so the refusal reaches ``run_manifest.json``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TypeVar

import torch

from deepfold3.model import kernel_census

__all__ = ["InferenceOnlyKernelError", "NotActive", "inference_only"]

_F = TypeVar("_F", bound=Callable)


class InferenceOnlyKernelError(RuntimeError):
    """An inference-only kernel was reached with grad enabled. A wiring bug in the caller."""


class NotActive(Exception):
    """A kernel cannot serve this call.

    Attributes:
        lever: Name of the kernel that refused.
        reason: Why it refused.
    """

    def __init__(self, lever: str, reason: str):
        self.lever = lever
        self.reason = reason
        super().__init__(f"NOT ACTIVE: {lever}: {reason}")
        kernel_census.record(lever, f"refused:{reason}")


def inference_only(lever: str) -> Callable[[_F], _F]:
    """Refuse to run a kernel wherever autograd could build a backward through it.

    Tests ``torch.is_grad_enabled()``, not ``requires_grad``: parameters always
    require grad, so the input flag would also reject real inference, and under
    enabled grad a Triton kernel records nothing, leaving the graph silently wrong.

    Args:
        lever: Kernel name for the error message and the census.

    Returns:
        A decorator for the kernel's entry point.
    """

    def decorate(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if torch.is_grad_enabled():
                kernel_census.record(lever, "refused:grad_enabled")
                raise InferenceOnlyKernelError(
                    f"{lever} is inference-only and has no backward: call it under "
                    "torch.no_grad() or torch.inference_mode(), or turn its flag off for training."
                )
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate
