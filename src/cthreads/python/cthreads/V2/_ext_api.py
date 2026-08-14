"""Lazy binding to ``cthreads._ext`` (native extension; not a v1 Python module)."""

from __future__ import annotations

from typing import Any

try:
    from cthreads import _ext as _ext
except ImportError:
    _ext = None  # type: ignore[assignment]


def _require_ext():
    if _ext is None:
        raise RuntimeError(
            "cthreads._ext is not available — build/install the native extension"
        )
    return _ext


def load_kernels(path: str) -> None:
    _require_ext().load_kernels(path)


def unload_kernels() -> None:
    _require_ext().unload_kernels()


def kernel_path() -> str | None:
    if _ext is None:
        return None
    fn = getattr(_ext, "kernel_path", None)
    return fn() if fn is not None else None


def thread(fn, *args: Any, **kwargs: Any):
    return _require_ext().thread(fn, *args, **kwargs)


def spawn(fn, *args: Any, **kwargs: Any):
    from .job import wrap_job

    return wrap_job(thread(fn, *args, **kwargs))
