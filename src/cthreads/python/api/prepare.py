"""
High-level prepare + thread entry with optional force-recompile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import CONFIG
from .build import build
from .compile import compile


def prepare(force: bool = False) -> Path:
    """
    codegen + native build (hash-cached unless force=True).
    Returns path to the kernel shared library.
    """
    info = compile(force=force)
    # Windows locks a loaded DLL; drop it before relink.
    try:
        import cthreads

        if hasattr(cthreads, "unload_kernels"):
            cthreads.unload_kernels()
    except ImportError:
        pass
    return build(project_root=info["root"], force=force)


def thread(fn, *args: Any, force: bool = False, **kwargs: Any):
    """
    Ensure kernels are up to date, then spawn a cthreads.Thread.

    Args:
        force: if True, force codegen + relink before dispatch.
    """
    import cthreads

    binary = prepare(force=force)
    cthreads.load_kernels(str(binary))
    return cthreads.thread(fn, *args, **kwargs)
