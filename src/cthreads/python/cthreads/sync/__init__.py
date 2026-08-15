"""
cthreads.sync — host sync / TBuffer API.

- Annotation: ``TBuffer[...]`` (from types)
- Host alloc: ``create_tbuffer`` / ``TBufferHandle`` / …
- Native locks/events: re-exported from ``cthreads._ext.sync`` when present
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

try:
    from cthreads import _ext as _ext
except ImportError:
    _ext = None  # type: ignore[assignment]

_native = getattr(_ext, "sync", None) if _ext is not None else None
if _native is not None:
    Lock = _native.Lock
    Event = _native.Event
    RWLock = getattr(_native, "RWLock", None)
else:
    Lock = None  # type: ignore[assignment,misc]
    Event = None  # type: ignore[assignment,misc]
    RWLock = None  # type: ignore[assignment,misc]

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
]
