"""
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
from .job import Job, wrap_job
from . import CONFIG

# --- _ext once; re-export submodules (same pattern for sync, math, …) --------
try:
    from . import _ext as _ext
except ImportError:
    _ext = None  # type: ignore[assignment]

if _ext is not None:
    sync = _ext.sync
    math = getattr(_ext, "math", None)  # older wheels may lack math
    load_kernels = _ext.load_kernels
    unload_kernels = _ext.unload_kernels
    host_os = _ext.host_os
    kernel_path = getattr(_ext, "kernel_path", lambda: None)

    def spawn(fn, *args: Any, **kwargs: Any) -> Job:
        """Low-level: bind args and return an awaitable Job (no prepare/compile)."""
        return wrap_job(_ext.thread(fn, *args, **kwargs))

else:
    sync = None  # type: ignore[assignment,misc]
    math = None  # type: ignore[assignment]
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
    "sync",
    "math",
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
