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
    Codegen + native build (hash-cached unless force=True).

    Does **not** unload a loaded kernel library. On Windows a force-relink
    over an already-loaded DLL will fail — call ``unload_kernels()`` first,
    then ``prepare(force=True)`` / ``load_kernels``.
    """
    info = compile(force=force)
    return build(project_root=info["root"], force=force)


def thread(fn, *args: Any, force: bool = False, **kwargs: Any) -> Job:
    """
    Return an awaitable Job on the currently loaded kernels.

    - If kernels are already loaded and ``force`` is False: spawn only
      (no prepare / load / unload — safe under concurrent callers).
    - If nothing is loaded: ``prepare`` (cache-checked) then ``load_kernels``,
      then spawn.
    - If ``force`` is True while kernels are loaded: raises — unload first.

    Usage::

        job = cthreads.thread(fn, ...)
        result = await job
    """
    from . import _ext

    loaded = _ext.kernel_path()
    if force:
        if loaded:
            raise RuntimeError(
                "cthreads.thread(force=True): kernels are still loaded. "
                "Call cthreads.unload_kernels() before force-rebuild, then "
                "thread(..., force=True) or prepare(force=True)+load_kernels."
            )
        binary = prepare(force=True)
        _ext.load_kernels(str(binary))
    elif not loaded:
        binary = prepare(force=False)
        _ext.load_kernels(str(binary))

    return wrap_job(_ext.thread(fn, *args, **kwargs))
