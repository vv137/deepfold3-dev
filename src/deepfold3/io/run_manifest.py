"""Run manifest: what a ``deepfold3 inference`` run used, written beside its outputs.

The record exists to answer whether two runs are comparable, so collecting it
never fails a run: anything unavailable (no git checkout, no ``nvidia-smi``) is
``null``. The file is replaced atomically; ``status`` stays ``running`` if the
process is killed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

MANIFEST_VERSION = 1


def _run(*cmd: str) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_state() -> dict:
    """Commit and dirty flag of the checkout this package runs from.

    A wheel installed under some other repository's ``.venv`` is not that
    repository's code, so the source file must be tracked for a commit to count.
    """
    source = Path(__file__).resolve()
    git = ("git", "-C", str(source.parent))
    if _run(*git, "ls-files", "--error-unmatch", source.name) is None:
        return {"commit": None, "dirty": None}
    status = _run(*git, "status", "--porcelain")
    return {"commit": _run(*git, "rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def runtime_versions() -> dict:
    """Python, torch, Triton, CUDA toolkit and driver versions."""
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    try:
        cueq_version = importlib.metadata.version("cuequivariance-ops-torch-cu13")
    except importlib.metadata.PackageNotFoundError:
        cueq_version = None
    driver = _run("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton_version,
        "cuequivariance_ops": cueq_version,
        "cuda": torch.version.cuda,
        "cuda_driver": driver.splitlines()[0] if driver else None,
    }


def gpu_name(device: torch.device) -> str | None:
    """Name of the GPU behind ``device``, or ``None`` for CPU."""
    return torch.cuda.get_device_name(device) if device.type == "cuda" else None


def checkpoint_record(path: Path) -> dict:
    """Path, size and mtime of a weights file. ``sha256`` is filled in later."""
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime": stat.st_mtime, "sha256": None}


def sha256_file(path: Path, chunk_size: int = 1 << 24) -> str:
    """SHA-256 of a file, hex. Seconds for a 5 GB checkpoint, so run it off the main thread."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def new_manifest(**fields) -> dict:
    """Manifest with the host fields every run records, plus ``fields``."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "status": "running",
        "command": sys.argv,
        "git": git_state(),
        "versions": runtime_versions(),
        **fields,
    }


def write_manifest(path: Path, manifest: dict) -> None:
    """Atomically replace ``path`` with ``manifest`` as JSON."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
