"""Which kernel each dispatch site actually ran.

``OptimConfig`` records what was *requested*. It does not record what ran, and
the two are not the same: a lever can be enabled and never reach its branch
because an earlier one shadows it. ``use_fused_ln_transpose`` is exactly that
today -- ``GridSelfAttention.forward`` tests ``self._fused_pair_bias`` first,
and every preset that turns the transpose lever on also turns pair bias on, so
the fused LayerNorm+transpose kernel is constructed and never executed.

Reading the dispatch chain to work that out is error-prone, and a benchmark run
leaves no record of the answer at all -- ``run_manifest.json`` stores
``optim: "inference"``, the name of the preset. So when two runs of one
checkpoint on two machines disagree, "which kernels did each one use?" cannot
be answered after the fact.

This module makes it a fact instead of an inference. Each dispatch site calls
:func:`record` with the path it took; :func:`snapshot` returns the tally, which
``io.run_manifest`` writes into every run. A lever that was requested but whose
branch never ran shows up as a site with no hits, and :func:`shadowed` names it
outright.

Counts are per process and are not synchronised: the increment is a plain dict
operation, so a miscount is possible if two threads dispatch at once. That is
accepted -- the value here is *which* paths ran, and an off-by-one in a count
that reaches the tens of thousands changes nothing.
"""

from __future__ import annotations

import threading
from collections import defaultdict

__all__ = ["record", "requested", "shadowed", "snapshot", "reset_counts", "reset", "format_census"]

_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
_requested: dict[str, str] = {}
_warned: set[str] = set()
_warn_lock = threading.Lock()


def record(site: str, path: str) -> None:
    """Note that ``site`` dispatched to ``path`` once.

    Args:
        site: Dotted name of the decision point, e.g. ``"GridSelfAttention.norm"``.
        path: The branch taken, e.g. ``"fused_pair_bias"`` or ``"stock"``.
    """
    _counts[site][path] += 1


def requested(site: str, lever: str) -> None:
    """Note that ``lever`` was enabled for ``site``, whether or not it ever runs.

    Called at construction. A lever recorded here whose path never appears in
    :func:`snapshot` is shadowed, and :func:`shadowed` reports it.
    """
    _requested[f"{site}:{lever}"] = site


def snapshot() -> dict[str, dict[str, int]]:
    """The tally so far, as ``{site: {path: count}}``."""
    return {site: dict(paths) for site, paths in sorted(_counts.items())}


def shadowed() -> list[str]:
    """Levers that were enabled but whose site never dispatched to them.

    A lever is reported only once its site has dispatched at least once --
    before the first forward every lever would look shadowed.
    """
    out = []
    for key, site in sorted(_requested.items()):
        lever = key.split(":", 1)[1]
        paths = _counts.get(site)
        if paths and lever not in paths:
            out.append(key)
    return out


def warn_once(key: str) -> bool:
    """True the first time ``key`` is seen, so a caller can log once per process."""
    with _warn_lock:
        if key in _warned:
            return False
        _warned.add(key)
        return True


def reset_counts() -> None:
    """Clear the tally but keep the requested levers.

    Levers are noted once, at model construction, so a process that loads one
    model and folds many targets calls this between targets.
    """
    _counts.clear()


def reset() -> None:
    """Clear everything. For tests."""
    _counts.clear()
    _requested.clear()
    _warned.clear()


def format_census() -> str:
    """One line per site, plus a line naming any shadowed lever."""
    lines = [
        f"{site}: " + ", ".join(f"{p}={n}" for p, n in sorted(paths.items())) for site, paths in snapshot().items()
    ]
    for key in shadowed():
        lines.append(f"SHADOWED {key} — enabled, but its site never dispatched to it")
    return "\n".join(lines) if lines else "(no kernel dispatch recorded)"
