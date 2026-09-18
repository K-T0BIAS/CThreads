"""
cthreads.sync - host sync / TBuffer API.

- Annotation: `TBuffer[...]` (from types)
- Host alloc: `create_tbuffer` / `TBufferHandle` / …
- Native locks/events: re-exported from `cthreads._ext.sync` when present
- GPU workgroup barrier stub: `__sync_threads`; inside `@Gpu` also
  `Barrier.arrive_and_wait()` (same GLSL lowering, no Barrier(...) call)
"""

from __future__ import annotations

from ..types import TBuffer
from .tbuffer_host import (
    TBufferHandle,
    create_tbuffer,
    destroy_tbuffer,
    tbuffer_free_read_copy,
    tbuffer_generation,
    tbuffer_ptr,
    tbuffer_read_copy_ptr,
)


def __sync_threads() -> None:
    """
    Workgroup barrier stub (CUDA-style).

    Only valid inside `@Gpu` bodies; compiled to GLSL barrier() /
    memoryBarrierShared(). Same device sync as Barrier.arrive_and_wait()
    on the GPU path.

    #### Raises
    - RuntimeError = called from ordinary Python (not compiled @Gpu)
    """
    raise RuntimeError(
        "cthreads.sync.__sync_threads() is only valid inside @Gpu bodies "
        "(workgroup barrier; compiled to GLSL barrier())"
    )


try:
    from cthreads import _ext as _ext
except ImportError:
    _ext = None  # type: ignore[assignment]

_native = getattr(_ext, "sync", None) if _ext is not None else None
if _native is not None:
    Lock = _native.Lock
    Event = _native.Event
    RWLock = getattr(_native, "RWLock", None)
    Barrier = getattr(_native, "Barrier", None)
    TBufferI64 = getattr(_native, "TBufferI64", None)
else:
    Lock = None  # type: ignore[assignment,misc]
    Event = None  # type: ignore[assignment,misc]
    RWLock = None  # type: ignore[assignment,misc]
    Barrier = None  # type: ignore[assignment,misc]
    TBufferI64 = None  # type: ignore[assignment,misc]

__all__ = [
    "TBuffer",
    "TBufferHandle",
    "create_tbuffer",
    "destroy_tbuffer",
    "tbuffer_ptr",
    "tbuffer_generation",
    "tbuffer_read_copy_ptr",
    "tbuffer_free_read_copy",
    "Lock",
    "Event",
    "RWLock",
    "Barrier",
    "TBufferI64",
    "__sync_threads",
]
