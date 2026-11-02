"""Collapse-free log-softmax, softmax and cross-entropy.

PyTorch port of Barron, Gnanapragasam & Tschannen, "A Fix for Softmax and
Cross-Entropy" (arXiv:2610.31642), §2 eq. 5-7.

Standard implementations shift the logits by ``z_max``, which makes the
softmax denominator ``1 + e^-m`` where ``m`` is the *logit margin* — how far
the target logit sits above the log-sum-exp of all the others::

    m = z_t - logsumexp_{j != t} z_j

Once ``m`` exceeds a precision cliff ``c_d`` (``c_16 ~= 5.5`` in bf16,
``c_32 ~= 16.6`` in fp32, ``c_64 ~= 36.7`` in fp64) the ``e^-m`` term is lost
to rounding when added to 1.0, the denominator becomes exactly 1.0, and the
target gradient ``g_t = 1/(1 + e^-m) - 1`` evaluates to exactly 0 while the
``K - 1`` non-target gradients stay positive. The gradients then no longer
sum to zero, which is what drives the "Slingshot" loss spikes under Adam.

The fix never adds 1 to ``e^-m``: the argmax slot's shifted logit is set to
exactly 0, excluded from the log-sum-exp, and recombined through
``softplus``, so ``g_t = -e^-m / (1 + e^-m)`` stays accurate all the way down
to exponent underflow (``m ~ 88``) instead of dying at ``c_d``.

These are drop-in replacements for ``F.log_softmax`` / ``F.softmax`` /
``F.cross_entropy``; they are slower than the fused ATen kernels and are
worth using only where logit margins realistically cross ``c_d`` for the
dtype the loss runs in.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["cross_entropy", "log_softmax", "logit_margin", "softmax"]


def log_softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Log-softmax whose gradients stay accurate past the precision cliff.

    Equation 5 of the paper. The argmax slot is pinned to exactly 0 and
    excluded from the log-sum-exp, so the ``1 + e^-m`` denominator is never
    formed.

    Args:
        z: Logits.
        dim: Axis the softmax runs over.

    Returns:
        ``log_softmax(z)`` along ``dim``, same shape and dtype as ``z``.
    """
    i_max = z.argmax(dim=dim, keepdim=True)
    z_max = z.gather(dim, i_max)
    # Exact 0 at the argmax, z_k - z_max elsewhere. Writing the 0 in rather
    # than subtracting keeps it exact even when z_max is huge.
    shifted = (z - z_max).scatter(dim, i_max, 0.0)
    # Exclude the argmax from the sum instead of contributing e^0 = 1 to it.
    others = shifted.scatter(dim, i_max, -torch.inf)
    m = -torch.logsumexp(others, dim=dim, keepdim=True)
    return shifted - F.softplus(-m)


def softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax via :func:`log_softmax`, so the argmax gradient survives.

    Args:
        z: Logits.
        dim: Axis the softmax runs over.

    Returns:
        ``softmax(z)`` along ``dim``.
    """
    return log_softmax(z, dim=dim).exp()


def logit_margin(z: torch.Tensor, target: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """``m = z_t - logsumexp_{j != t} z_j`` — the paper's logit margin (eq. 2).

    Useful as a diagnostic: compare it against the cliff for the dtype the
    loss runs in (5.5 bf16 / 16.6 fp32 / 36.7 fp64) to tell whether a head is
    anywhere near Softmax Collapse.

    Args:
        z: Logits.
        target: Integer class indices, ``z.shape`` without ``dim``.
        dim: Class axis of ``z``.

    Returns:
        The margin, shaped like ``target``.
    """
    idx = target.unsqueeze(dim)
    others = (z - z.gather(dim, idx)).scatter(dim, idx, -torch.inf)
    return -torch.logsumexp(others, dim=dim).squeeze(dim)


def cross_entropy(z: torch.Tensor, target: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Hard-target cross-entropy that keeps summing to zero past the cliff.

    Equation 6 of the paper: with the target slot pinned to the constant 0,
    ``z_t`` enters the loss only through the ``K - 1`` differences
    ``z_j - z_t``, so autograd recovers the target gradient as the negated sum
    of the non-target ones and ``sum_k g_k == 0`` holds exactly.

    Unlike ``F.cross_entropy`` this returns the unreduced per-element loss and
    takes no ``weight`` / ``ignore_index`` / ``reduction``; callers in this
    repo already apply their own masked means.

    Args:
        z: Logits, class axis ``dim``.
        target: Integer class indices, ``z.shape`` without ``dim``.
        dim: Class axis of ``z``.

    Returns:
        ``-log_softmax(z)[target]``, shaped like ``target``.
    """
    idx = target.unsqueeze(dim)
    others = (z - z.gather(dim, idx)).scatter(dim, idx, -torch.inf)
    m = -torch.logsumexp(others, dim=dim).squeeze(dim)
    return F.softplus(-m)
