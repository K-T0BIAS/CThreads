"""High-level prepare + thread entry for V2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .build import build
from .compile.orchestrator import CompileSession
from .job import Job, wrap_job


def compile(force: bool = False) -> dict:
    """
    Drain the V2 registry into units, emit C++, write tbuffer runtime.

    ``force`` is accepted for API parity; unit emit always regenerates today.
    """
    del force  # reserved for source-fingerprint skip later
    return CompileSession.compile()


def prepare(force: bool = False) -> Path:
    """
    Codegen + native build.

    Does **not** unload a loaded kernel library. On Windows a force-relink
    over an already-loaded DLL will fail — call ``unload_kernels()`` first.
    """
    info = compile(force=force)
    return build(project_root=info["root"], force=force)


def thread(fn, *args: Any, force: bool = False, **kwargs: Any) -> Job:
    """
    Return an awaitable Job on the currently loaded kernels.

    Pack/unpack still goes through ``cthreads.marshal`` (imported by ``_ext``)
    until the public package cutover points C++ at ``cthreads.V2.marshal``.
    """
    from . import _ext_api

    loaded = _ext_api.kernel_path()
    if force:
        if loaded:
            raise RuntimeError(
                "cthreads.V2.thread(force=True): kernels are still loaded. "
                "Call unload_kernels() before force-rebuild, then "
                "thread(..., force=True) or prepare(force=True)+load_kernels."
            )
        binary = prepare(force=True)
        _ext_api.load_kernels(str(binary))
    elif not loaded:
        binary = prepare(force=False)
        _ext_api.load_kernels(str(binary))

    return wrap_job(_ext_api.thread(fn, *args, **kwargs))
