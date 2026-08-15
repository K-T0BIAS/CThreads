"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

cthreads — Python frontend + native runtime.

Public surface:
  @cthreads.Threadable / @cthreads.Thread
  cthreads.compile / build / prepare / thread
  cthreads.Job (awaitable native thread handle)
  cthreads.sync / cthreads.math / …  (pybind submodules on _ext)

Jobs::

    job = cthreads.thread(fn, ...)
    result = await job                 # preferred async path (auto-starts)
    job.start(); job.join(); job.result()   # sync still works
    job.sync_state() / sync_state(job)      # mid-run Threadable writeback
    __sync_state() in @Thread bodies        # compiled kernel barrier

``thread()`` spawns on already-loaded kernels (no per-call unload). First use
runs cache-checked ``prepare`` + ``load_kernels``. Call ``unload_kernels()``
yourself before a force-rebuild, or at process exit.

Native API pattern (sync, math, future libs)
--------------------------------------------
Pybind defines submodules on ``cthreads._ext`` (``def_submodule("sync")``, …).
We import ``_ext`` **once** and re-export each submodule on ``cthreads``:

    from cthreads import sync, math
    sync.Lock()
    math.abs(-1.0)

Do **not** ``sys.modules["cthreads.sync"] = …`` or add shadow ``sync.py`` /
``math.py`` wrappers — that can double-init the extension and break pybind
types (``Lock`` already defined).
"""

from __future__ import annotations

from typing import Any

from .Threadable.wrapper import Threadable
from .Thread.wrapper import Thread
from .compile import compile
from .build import build
from .prepare import prepare, thread
from .job import Job, wrap_job, sync_state
from .pyTypes import TBuffer
from .tbuffer_host import (
    TBufferHandle,
    create_tbuffer,
    destroy_tbuffer,
    tbuffer_generation,
    tbuffer_read_copy_ptr,
    tbuffer_free_read_copy,
)
from . import CONFIG

# Kernel-only barrier: importable so ``from cthreads import __sync_state`` works
# and codegen can resolve the name. Calling it from normal Python is an error —
# it only runs after AST lowering to ``cthreads::detail::__sync_state()``
# (kernel bridge → ``_ext`` TLS writeback).
def __sync_state() -> None:
    raise RuntimeError(
        "cthreads.__sync_state() is only valid inside @Thread bodies "
        "(it is compiled to cthreads::detail::__sync_state())"
    )


# --- _ext once; re-export submodules (same pattern for sync, math, ...) --------
try:
    from . import _ext as _ext # try getting the extension module (c++ code)
except ImportError:
    _ext = None  # type: ignore[assignment]

if _ext is not None:
    sync = _ext.sync # get the sync module
    math = getattr(_ext, "math", None)  # older wheels may lack math
    linalg = getattr(_ext, "linalg", None)
    load_kernels = _ext.load_kernels # get the load_kernels function
    unload_kernels = _ext.unload_kernels # get the unload_kernels function
    host_os = _ext.host_os # get the host_os function
    kernel_path = getattr(_ext, "kernel_path", lambda: None) # get the kernel_path function

    def spawn(fn, *args: Any, **kwargs: Any) -> Job:
        """Low-level: bind args and return an awaitable Job (no prepare/compile)."""
        return wrap_job(_ext.thread(fn, *args, **kwargs))

else:
    sync = None  # type: ignore[assignment,misc]
    math = None  # type: ignore[assignment]
    linalg = None  # type: ignore[assignment]
    load_kernels = None  # type: ignore[assignment,misc]
    unload_kernels = None  # type: ignore[assignment,misc]
    host_os = None  # type: ignore[assignment,misc]
    spawn = None  # type: ignore[assignment,misc]

    def kernel_path():
        return None

__all__ = [
    "Threadable",
    "Thread",
    "compile",
    "build",
    "prepare",
    "thread",
    "spawn",
    "Job",
    "sync_state",
    "__sync_state",
    "TBuffer",
    "TBufferHandle",
    "create_tbuffer",
    "destroy_tbuffer",
    "tbuffer_generation",
    "tbuffer_read_copy_ptr",
    "tbuffer_free_read_copy",
    "sync",
    "math",
    "linalg",
    "load_kernels",
    "unload_kernels",
    "kernel_path",
    "host_os",
    "BINARY_PATH",
    "STORE",
    "KERNELS",
    "VERSION",
]


def __getattr__(name: str) -> Any:
    # Live bindings — CONFIG.* is mutated by compile/build.
    if name == "BINARY_PATH":
        return CONFIG.BINARY_PATH
    if name == "STORE":
        return CONFIG.STORE
    if name == "KERNELS":
        return CONFIG.KERNELS
    if name == "VERSION":
        return CONFIG.VERSION
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
