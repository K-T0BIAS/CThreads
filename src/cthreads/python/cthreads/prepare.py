"""
High-level prepare + thread entry with optional force-recompile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .build import build
from .compile import compile
from .job import Job, wrap_job


def prepare(force: bool = False) -> Path:
    """
    codegen + native build (hash-cached unless force=True).
    Returns path to the kernel shared library.
    """
    info = compile(force=force)
    # Windows locks a loaded DLL; drop it before relink.
    try:
        from . import _ext

        _ext.unload_kernels()
    except ImportError:
        pass
    return build(project_root=info["root"], force=force)


def thread(fn, *args: Any, force: bool = False, **kwargs: Any) -> Job:
    """
    Ensure kernels are up to date, then return an awaitable Job.

    Usage::

        job = cthreads.thread(fn, ...)
        result = await job              # FastAPI / asyncio
        # or
        job.start(); job.join(); job.result()   # sync
    """
    from . import _ext

    binary = prepare(force=force)
    _ext.load_kernels(str(binary))
    return wrap_job(_ext.thread(fn, *args, **kwargs))
