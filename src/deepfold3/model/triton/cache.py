"""Persistent autotuning cache for Triton kernels.

Adapted from NVIDIA cuEquivariance's cache_manager.py pattern, simplified
to use torch.cuda for GPU fingerprinting (no pynvml dependency).

Cache files are stored as JSON at ~/.cache/deepfold3-triton/ (or
DEEPFOLD3_TRITON_CACHE_DIR). Each file is keyed by kernel name and GPU
compute capability: {kernel_name}.{major}.{minor}.json.

GPU matching uses fuzzy tolerance: when an exact GPU hash is not found,
the cache falls back to GPUs with similar memory/SM count (within 20%).

Environment variables:
    DEEPFOLD3_TRITON_CACHE_DIR: Override cache directory.
    DEEPFOLD3_TRITON_NO_CACHE:  Set to "1" to disable caching entirely.

Keys are exact shapes. Training fixes N and the MSA depth, so a warmed cache
always hits. Inference pads N to a bucket but its MSA depth varies by input, so
a new shape would benchmark from scratch: ``set_nearest_shape_lookup`` makes a
miss reuse the closest cached shape instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FILE_LOCK = threading.Lock()


# ── GPU Fingerprinting ───────────────────────────────────────────────────


def get_gpu_fingerprint() -> dict[str, Any] | None:
    """Get GPU properties via torch.cuda. Returns None if CUDA unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "major": props.major,
            "minor": props.minor,
            "total_memory_gb": math.ceil(props.total_memory / (1024**3)),
            "multi_processor_count": props.multi_processor_count,
        }
    except Exception as e:
        logger.debug(f"Failed to get GPU fingerprint: {e}")
        return None


def _hash_gpu_info(info: dict[str, Any]) -> str:
    """Deterministic hash of GPU properties (excluding name)."""
    key_parts = [
        str(info.get("major", 0)),
        str(info.get("minor", 0)),
        str(info.get("total_memory_gb", 0)),
        str(info.get("multi_processor_count", 0)),
    ]
    key_string = "_".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()[:16]


# ── Cache Directory ──────────────────────────────────────────────────────


def _get_cache_dir() -> Path:
    """Resolve cache directory, creating it if needed.

    Resolution order: ``DEEPFOLD3_TRITON_CACHE_DIR`` → ``XDG_CACHE_HOME`` →
    ``~/.cache``. Honoring ``XDG_CACHE_HOME`` matters in containers where the
    image's real ``$HOME`` (e.g. ``/home/mpiuser``) is ephemeral and the
    persistent volume is mounted elsewhere (e.g. ``/home/jovyan``): the kakao
    trainjob sets ``XDG_CACHE_HOME=/home/jovyan/.cache`` to redirect every
    cache onto the shared PVC, and this cache must follow suit or it lands on
    ephemeral per-pod storage instead of the shared volume.
    """
    env_dir = os.environ.get("DEEPFOLD3_TRITON_CACHE_DIR")
    if env_dir:
        cache_dir = Path(env_dir)
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg) if xdg else Path.home() / ".cache"
        cache_dir = base / "deepfold3-triton"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _is_cache_disabled() -> bool:
    return os.environ.get("DEEPFOLD3_TRITON_NO_CACHE") == "1"


# ── Fuzzy GPU Matching ───────────────────────────────────────────────────


def _within_tolerance(a: int, b: int, tol: float = 0.2) -> bool:
    """Check if two values are within relative tolerance."""
    if a == 0 and b == 0:
        return True
    return abs(a - b) / max(a, b) < tol


def _fuzzy_match(target: dict, candidate: dict) -> int:
    """Score how well a candidate GPU matches the target. Higher = better.

    Returns:
        3: exact match (memory + SM count)
        2: full match (memory AND SM within tolerance)
        1: partial match (memory OR SM within tolerance)
        0: no match
    """
    t_mem = target.get("total_memory_gb", 0)
    c_mem = candidate.get("total_memory_gb", 0)
    t_sm = target.get("multi_processor_count", 0)
    c_sm = candidate.get("multi_processor_count", 0)

    mem_exact = t_mem == c_mem
    sm_exact = t_sm == c_sm
    mem_close = _within_tolerance(t_mem, c_mem)
    sm_close = _within_tolerance(t_sm, c_sm)

    if mem_exact and sm_exact:
        return 3
    if mem_close and sm_close:
        return 2
    if mem_close or sm_close:
        return 1
    return 0


# ── CacheManager ─────────────────────────────────────────────────────────


class CacheManager:
    """Singleton managing persistent Triton autotuning cache.

    Usage:
        mgr = CacheManager.get()
        config = mgr.get_config("my_kernel", "M128_N64_K32")
        if config is None:
            # run autotuning...
            mgr.set_config("my_kernel", "M128_N64_K32", {"BLOCK_M": 64, ...}, time=0.042)
    """

    _instance: CacheManager | None = None
    _init_lock = threading.Lock()

    def __init__(self):
        # Lazy GPU fingerprint: do NOT call torch.cuda here. CacheManager.get()
        # is invoked at module-import time (via @persistent_autotune decorators
        # on Triton JIT functions), which can be BEFORE CUDA is initialized in
        # the host process — especially under torchrun where the device isn't
        # set until ``torch.cuda.set_device(local_rank)`` runs much later.
        # Fingerprinting prematurely would record ``gpu_info=None`` and write
        # caches under the ``nogpu`` key with file suffix ``.0.0.json``,
        # bypassing all per-GPU tuning. Defer until first ``_resolve_gpu()``.
        self._gpu_resolved = False
        self.gpu_info: dict[str, Any] | None = None
        self.gpu_key: str = "nogpu"
        self._caches: dict[str, dict[str, Any]] = {}  # kernel -> {input_key -> {...}}
        self._dirty: set[str] = set()
        self._cache_dir = _get_cache_dir()

    def _resolve_gpu(self) -> None:
        """Fetch GPU fingerprint on first use, retrying until CUDA is ready.

        Triton kernels can be defined (decorator runs) before CUDA is set up.
        Once an actual kernel call happens, CUDA is guaranteed to be live, so
        this is the right moment to record the device identity.
        """
        if self._gpu_resolved:
            return
        info = get_gpu_fingerprint()
        if info is not None:
            self.gpu_info = info
            self.gpu_key = _hash_gpu_info(info)
            self._gpu_resolved = True
            logger.debug(
                f"Triton cache: resolved GPU "
                f"sm_{info.get('major', '?')}{info.get('minor', '?')} "
                f"({info.get('name', '?')}, {info.get('multi_processor_count', '?')} SMs)"
            )

    @classmethod
    def get(cls) -> CacheManager:
        """Get or create the singleton CacheManager."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _json_path(self, kernel_name: str) -> Path:
        """Path to the JSON cache file for a kernel."""
        self._resolve_gpu()
        if self.gpu_info:
            major, minor = self.gpu_info["major"], self.gpu_info["minor"]
        else:
            major, minor = 0, 0
        return self._cache_dir / f"{kernel_name}.{major}.{minor}.json"

    def load(self, kernel_name: str) -> dict[str, Any]:
        """Load cache for a kernel, with fuzzy GPU matching."""
        if kernel_name in self._caches:
            return self._caches[kernel_name]

        gpu_cache: dict[str, Any] = {}
        json_path = self._json_path(kernel_name)

        if _is_cache_disabled() or not json_path.exists():
            self._caches[kernel_name] = gpu_cache
            return gpu_cache

        try:
            with _FILE_LOCK, open(json_path) as f:
                file_data = json.load(f)
        except Exception as e:
            logger.debug(f"Failed to load cache {json_path}: {e}")
            self._caches[kernel_name] = gpu_cache
            return gpu_cache

        # Try exact GPU key match
        if self.gpu_key in file_data:
            gpu_cache = file_data[self.gpu_key]
        else:
            # Fuzzy matching
            best_key = None
            best_score = 0
            for key, entry in file_data.items():
                entry_gpu = entry.get("gpu_info")
                if entry_gpu and self.gpu_info:
                    score = _fuzzy_match(self.gpu_info, entry_gpu)
                    if score > best_score:
                        best_score = score
                        best_key = key
            if best_key is not None:
                gpu_cache = file_data[best_key]
                logger.debug(f"Fuzzy GPU match for {kernel_name} (score={best_score})")
            elif file_data:
                # Fall back to first available entry
                best_key = next(iter(file_data))
                gpu_cache = file_data[best_key]
                logger.debug(f"No GPU match for {kernel_name}, using first entry")

        # Always stamp our GPU info
        gpu_cache["gpu_info"] = self.gpu_info
        self._caches[kernel_name] = gpu_cache
        return gpu_cache

    def save(self, kernel_name: str) -> None:
        """Save cache for a kernel to disk (atomic write)."""
        if _is_cache_disabled():
            return
        if kernel_name not in self._dirty:
            return

        json_path = self._json_path(kernel_name)
        cache_data = self._caches.get(kernel_name, {})

        # Load existing file data to merge
        existing: dict[str, Any] = {}
        if json_path.exists():
            try:
                with _FILE_LOCK, open(json_path) as f:
                    existing = json.load(f)
            except Exception:
                pass

        existing[self.gpu_key] = cache_data
        # Guaranteed-unique temp name. ``os.getpid()`` alone collides across
        # nodes (separate PID namespaces): on a shared PVC two ranks write the
        # same ``.{pid}.tmp`` and one's ``os.replace`` then races on a file the
        # other already consumed (the live "No such file or directory" save
        # errors under 64-rank DDP). ``mkstemp`` is atomic + unique per writer;
        # ``os.replace`` stays atomic so concurrent saves are last-writer-wins
        # rather than erroring. (Warm the cache once before DDP to avoid the
        # write storm entirely; this only hardens the cold-start path.)
        temp_path: Path | None = None
        try:
            with _FILE_LOCK:
                fd, tmp_name = tempfile.mkstemp(dir=str(json_path.parent), prefix=f"{json_path.name}.", suffix=".tmp")
                temp_path = Path(tmp_name)
                with os.fdopen(fd, "w") as f:
                    json.dump(existing, f, indent=2, default=str)
                os.replace(temp_path, json_path)
            self._dirty.discard(kernel_name)
            logger.debug(f"Saved cache for {kernel_name} to {json_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {kernel_name}: {e}")
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def get_config(self, kernel_name: str, input_key: str) -> dict[str, Any] | None:
        """Get cached config for a kernel + input shape."""
        cache = self._caches.get(kernel_name)
        if cache is None:
            cache = self.load(kernel_name)
        entry = cache.get(input_key)
        if entry is not None and isinstance(entry, dict):
            return entry.get("config")
        return None

    def set_config(self, kernel_name: str, input_key: str, config: dict, time: float = 0.0) -> None:
        """Store a config result and mark cache dirty."""
        if kernel_name not in self._caches:
            self.load(kernel_name)
        self._caches[kernel_name][input_key] = {"config": config, "time": time}
        self._dirty.add(kernel_name)

    def save_all(self) -> None:
        """Save all dirty caches."""
        for kernel_name in list(self._dirty):
            self.save(kernel_name)

    def get_nearest_config(self, kernel_name: str, key_args: list[str], args: dict) -> tuple[str, dict] | None:
        """Config of the cached shape closest to ``args``, as ``(input_key, config)``.

        Distance is the summed ``|log a − log b|`` over ``key_args``; only
        entries with the same key form count. ``None`` when nothing is cached.
        """
        target = [args.get(k) for k in key_args]
        if not all(isinstance(v, int) and v > 0 for v in target):
            return None
        form = re.compile("_".join(rf"{re.escape(k)}(\d+)" for k in key_args))
        cache = self._caches.get(kernel_name)
        if cache is None:
            cache = self.load(kernel_name)
        best: tuple[float, str, dict] | None = None
        for input_key, entry in cache.items():
            m = form.fullmatch(input_key)
            if m is None or not isinstance(entry, dict) or "config" not in entry:
                continue
            dist = sum(abs(math.log(max(int(c), 1) / t)) for c, t in zip(m.groups(), target, strict=True))
            if best is None or dist < best[0]:
                best = (dist, input_key, entry["config"])
        return None if best is None else (best[1], best[2])


_nearest_shape_lookup = False


def set_nearest_shape_lookup(enabled: bool) -> None:
    """On a cache miss, reuse the closest cached shape's config instead of benchmarking.

    For single-process inference, where each input is a new shape. The config
    only picks tile sizes, so this trades a little speed on far shapes for no
    autotune stall. Training leaves it off and benchmarks its one fixed shape.
    """
    global _nearest_shape_lookup
    _nearest_shape_lookup = enabled


_fixed_config = False


def set_fixed_config(enabled: bool) -> None:
    """Serve each kernel its first config, ignoring the cache and never benchmarking.

    Tile sizes change the order of floating-point sums, so a benchmark winner
    (which depends on timing) or a cached config (which depends on the host)
    can change the low bits of a result. For ``inference --det``.
    """
    global _fixed_config
    _fixed_config = enabled


# ── Persistent Autotune Decorator ────────────────────────────────────────


def make_input_key(args: dict, key_args: list[str]) -> str:
    """Build a cache key string from the specified argument names."""
    parts = []
    for k in key_args:
        v = args.get(k)
        if v is not None:
            parts.append(f"{k}{v}")
    return "_".join(parts)


def _reset_buffers_hook(names: list[str]):
    """An autotune ``pre_hook`` that zeroes the output buffers a kernel accumulates into.

    Autotuning benchmarks a kernel on the caller's real arguments, once per
    config and several times each, and then runs it again. A kernel that
    ``atomic_add``s into a buffer therefore returns the sum of all those runs,
    so the first backward on an untuned shape was wrong by a factor of hundreds.

    Triton's own ``reset_to_zero`` calls ``kwargs[name].zero_()``, which these
    kernels cannot use: an unused output may be an int, and an unused pointer
    may alias an input (``GRAD_MASK_ptr`` is ``x`` when there is no mask), which
    would erase it. Those are skipped here.
    """

    def hook(kwargs: dict, reset_only: bool = False) -> None:
        others = [v for k, v in kwargs.items() if k not in names]
        for name in names:
            buf = kwargs[name]
            if hasattr(buf, "zero_") and not any(buf is o for o in others):
                buf.zero_()

    return hook


def persistent_autotune(
    configs: list,
    key: list[str],
    kernel_name: str | None = None,
    reset_to_zero: list[str] | None = None,
):
    """Decorator that wraps triton.autotune with persistent cache.

    On first call for a new (kernel, input_shape) combination:
    1. Check cache — if found, use cached config directly
    2. Otherwise, run normal ``triton.autotune`` and persist the winner

    This is NOT a replacement for triton.autotune — it wraps it. The
    triton.autotune mechanism still handles the actual benchmarking; we just
    persist the results across process restarts so a warmed cache (see
    ``deepfold3 warmup-triton``, ideally on shared storage) skips the
    benchmark phase on later runs.

    Args:
        configs: List of triton.Config objects (same as triton.autotune).
        key: List of argument names to use as cache key.
        reset_to_zero: Output arguments the kernel accumulates into. Required
            for any kernel using ``atomic_add``; see :func:`_reset_buffers_hook`.
        kernel_name: Optional override for cache file name.

    Returns:
        Decorator that applies triton.autotune with cache persistence.

    Usage:
        @persistent_autotune(configs=[...], key=["M", "N", "K"])
        @triton.jit
        def my_kernel(...):
            ...
    """
    try:
        import triton
    except ImportError:
        # If triton not available, return identity decorator
        def _identity(fn):
            return fn

        return _identity

    tune_kwargs = {"pre_hook": _reset_buffers_hook(reset_to_zero)} if reset_to_zero else {}

    if _is_cache_disabled():
        # Just use standard triton.autotune
        return triton.autotune(configs=configs, key=key, **tune_kwargs)

    def decorator(fn):
        # Extract name from fn — handles JITFunction, Heuristics, or plain functions
        fn_name = kernel_name
        if fn_name is None:
            obj = fn
            while fn_name is None:
                if hasattr(obj, "__name__"):
                    fn_name = obj.__name__
                elif hasattr(obj, "fn"):
                    obj = obj.fn
                else:
                    fn_name = type(fn).__name__
                    break
        mgr = CacheManager.get()

        # Load any existing cache for this kernel (warm it once via
        # ``deepfold3 warmup-triton`` so the first training run skips autotune).
        mgr.load(fn_name)

        # Apply standard triton.autotune
        autotuned_fn = triton.autotune(configs=configs, key=key, **tune_kwargs)(fn)

        # Wrap to (1) inject from our persistent cache into Triton's
        # internal Autotuner cache *before* the first call so the benchmark
        # phase is skipped, and (2) capture results after autotune fires.
        _original_run = autotuned_fn.run

        def _build_triton_key(args: tuple, kwargs: dict) -> tuple | None:
            """Mirror ``Autotuner.run``'s key construction so we can index
            ``autotuned_fn.cache`` directly. Returns ``None`` on any mismatch."""
            try:
                arg_names = autotuned_fn.arg_names
                nargs = dict(zip(arg_names, args, strict=False))
                all_args = {**nargs, **kwargs}
                _args = {k: v for k, v in all_args.items() if k in arg_names}
                key_tuple = [_args[k] for k in autotuned_fn.keys if k in _args]
                for _, arg in _args.items():
                    if hasattr(arg, "dtype"):
                        key_tuple.append(str(arg.dtype))
                return tuple(key_tuple)
            except Exception:
                return None

        # Track which (fn_name, input_key) tuples we've already persisted so
        # the post-hook is a single dict membership check on the hot path.
        _persisted: set[str] = set()
        # Sentinel installed in ``autotuned_fn.cache`` once we've handled the
        # pre-hook injection for a given Triton key. Lookup on subsequent
        # calls is then a simple ``in autotuned_fn.cache`` check that lets us
        # bail out without rebuilding the key tuple or input string.
        _injected_keys: set[tuple] = set()

        def _cached_run(*args, **kwargs):
            triton_key = _build_triton_key(args, kwargs)

            # Hot path: Triton's own cache already has this key (either from
            # a prior autotune run or from our pre-hook injection on a
            # previous call). Skip the persistent-cache machinery entirely.
            if triton_key is not None and triton_key in autotuned_fn.cache:
                return _original_run(*args, **kwargs)

            if _fixed_config:
                if triton_key is None:
                    raise RuntimeError(f"Triton {fn_name}: cannot fix the config, no cache key for this call")
                autotuned_fn.cache[triton_key] = configs[0]
                return _original_run(*args, **kwargs)

            # ── Pre-hook: inject from persistent cache if Triton hasn't
            # benchmarked this shape yet, so we skip the autotune entirely.
            input_key: str | None = None
            try:
                if triton_key is not None and triton_key not in _injected_keys:
                    arg_names = autotuned_fn.arg_names
                    nargs = dict(zip(arg_names, args, strict=False))
                    all_args = {**nargs, **kwargs}
                    input_key = make_input_key(all_args, key)
                    cached_cfg = mgr.get_config(fn_name, input_key) if input_key else None
                    if cached_cfg is None and input_key and _nearest_shape_lookup:
                        nearest = mgr.get_nearest_config(fn_name, key, all_args)
                        if nearest is not None:
                            logger.info("Triton %s: %s uses the config of %s", fn_name, input_key, nearest[0])
                            cached_cfg = nearest[1]
                            _persisted.add(input_key)  # borrowed, not a measurement of this shape
                    if cached_cfg is not None:
                        # Reconstruct a triton.Config and inject directly so
                        # ``Autotuner.run`` finds it on the cache lookup.
                        meta_keys = {"num_warps", "num_stages", "num_ctas", "maxnreg"}
                        block_kwargs = {k: v for k, v in cached_cfg.items() if k not in meta_keys}
                        injected = triton.Config(
                            block_kwargs,
                            num_warps=cached_cfg.get("num_warps", 4),
                            num_stages=cached_cfg.get("num_stages", 2),
                            **({"num_ctas": cached_cfg["num_ctas"]} if "num_ctas" in cached_cfg else {}),
                            **({"maxnreg": cached_cfg["maxnreg"]} if cached_cfg.get("maxnreg") is not None else {}),
                        )
                        autotuned_fn.cache[triton_key] = injected
                    _injected_keys.add(triton_key)
            except Exception as e:
                logger.warning(
                    "persistent_autotune pre-hook failed for %s — falling back to triton.autotune: %s",
                    fn_name,
                    e,
                )

            result = _original_run(*args, **kwargs)

            # ── Post-hook: persist the autotune result for next process.
            try:
                if input_key and input_key not in _persisted and autotuned_fn.best_config is not None:
                    if mgr.get_config(fn_name, input_key) is None:
                        best = autotuned_fn.best_config
                        config_dict = {**best.kwargs}
                        config_dict["num_warps"] = best.num_warps
                        config_dict["num_stages"] = best.num_stages
                        mgr.set_config(fn_name, input_key, config_dict)
                        mgr.save(fn_name)
                    _persisted.add(input_key)
            except Exception as e:
                logger.warning(
                    "persistent_autotune post-hook failed for %s — autotune result not persisted: %s",
                    fn_name,
                    e,
                )
            return result

        autotuned_fn.run = _cached_run
        return autotuned_fn

    return decorator
