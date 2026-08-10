"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

cthreads.sync.* method tables for @Thread lowering.

Python names match C++ (`acquire` -> `acquire()`). Receiver is a typed
`PyCthreadsInternal` local / param; emit assumes an lvalue / reference
(`({recv}).method(...)`).
"""

from collections.abc import Callable
from typing import NamedTuple

Emit = Callable[[str, list[str]], str]  # (receiver_cpp, arg_cpps) -> expr


class SyncOp(NamedTuple):
    emit: Emit
    min_arity: int
    max_arity: int
    cpp_include: str | None = None


def _method(name: str, min_arity: int = 0, max_arity: int | None = None) -> SyncOp:
    """1:1 Python method -> `(recv).name(args...)`."""
    if max_arity is None:
        max_arity = min_arity

    def emit(recv: str, args: list[str]) -> str:
        if not args:
            return f"({recv}).{name}()"
        return f"({recv}).{name}({', '.join(args)})"

    return SyncOp(emit=emit, min_arity=min_arity, max_arity=max_arity)


LOCK_METHODS: dict[str, SyncOp] = {
    "acquire": _method("acquire", 0),
    "release": _method("release", 0),
    "try_acquire": _method("try_acquire", 0),
}

EVENT_METHODS: dict[str, SyncOp] = {
    "set": _method("set", 0),
    "clear": _method("clear", 0),
    "is_set": _method("is_set", 0),
    "wait": _method("wait", 0),
    "wait_for": _method("wait_for", 1),
}

RWLOCK_METHODS: dict[str, SyncOp] = {
    "acquire_read": _method("acquire_read", 0),
    "release_read": _method("release_read", 0),
    "try_acquire_read": _method("try_acquire_read", 0),
    "acquire_write": _method("acquire_write", 0),
    "release_write": _method("release_write", 0),
    "try_acquire_write": _method("try_acquire_write", 0),
}

# PyCthreadsInternal.name -> method table
SYNC_METHODS: dict[str, dict[str, SyncOp]] = {
    "Lock": LOCK_METHODS,
    "Event": EVENT_METHODS,
    "RWLock": RWLOCK_METHODS,
}
