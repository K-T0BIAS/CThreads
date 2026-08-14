"""
cthreads.V2 — standalone frontend + compile + build + dispatch.

Public surface mirrors top-level ``cthreads``, but lives entirely under
``cthreads.V2`` except for the native extension ``cthreads._ext``.

Note: ``_ext.thread`` still imports ``cthreads.marshal`` (v1) from C++ until
the public cutover. V2 ships its own ``marshal`` / ``sync`` copies for a
self-contained Python tree and future switch.
"""

from __future__ import annotations

from typing import Any

from .frontend.Threadable import Threadable
from .frontend.Thread import Thread
from .frontend.Registry import REGISTRY
from .prepare import compile, prepare, thread
from .build import build
from .job import Job, wrap_job, sync_state
from . import sync
from .sync import (
    TBuffer,
    TBufferHandle,
    create_tbuffer,
    destroy_tbuffer,
    tbuffer_generation,
    tbuffer_read_copy_ptr,
    tbuffer_free_read_copy,
)
from .kernel_meta import KERNELS
from . import runtime
from . import _ext_api


def __sync_state() -> None:
    raise RuntimeError(
        "cthreads.V2.__sync_state() is only valid inside @Thread bodies "
        "(it is compiled to cthreads::detail::__sync_state())"
    )


load_kernels = _ext_api.load_kernels
unload_kernels = _ext_api.unload_kernels
kernel_path = _ext_api.kernel_path
spawn = _ext_api.spawn

try:
    from cthreads import _ext as _ext
except ImportError:
    _ext = None  # type: ignore[assignment]

if _ext is not None:
    math = getattr(_ext, "math", None)
    linalg = getattr(_ext, "linalg", None)
    host_os = getattr(_ext, "host_os", None)
else:
    math = None  # type: ignore[assignment]
    linalg = None  # type: ignore[assignment]
    host_os = None  # type: ignore[assignment]


__all__ = [
    "Threadable",
    "Thread",
    "compile",
    "build",
    "prepare",
    "thread",
    "spawn",
    "Job",
    "wrap_job",
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
    "KERNELS",
    "VERSION",
    "REGISTRY",
]


def __getattr__(name: str) -> Any:
    if name == "BINARY_PATH":
        return runtime.BINARY_PATH
    if name == "VERSION":
        return REGISTRY.VERSION
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
