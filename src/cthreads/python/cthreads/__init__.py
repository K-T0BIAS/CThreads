"""
cthreads — Python frontend + native runtime.

Public surface:
  @cthreads.Threadable / @cthreads.Thread
  cthreads.compile / build / prepare / thread
  cthreads.Job (native thread handle)
  cthreads.sync (Lock, Event, RWLock)
"""

from __future__ import annotations

from typing import Any

from .Threadable.wrapper import Threadable
from .Thread.wrapper import Thread
from .compile import compile
from .build import build
from .prepare import prepare, thread
from . import CONFIG

from ._ext import (
    Job,
    sync,
    load_kernels,
    unload_kernels,
    host_os,
)

try:
    from ._ext import kernel_path
except ImportError:  # older _ext wheel before marshal refactor
    def kernel_path():
        return None

from ._ext import thread as spawn  # low-level: no compile/build

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
