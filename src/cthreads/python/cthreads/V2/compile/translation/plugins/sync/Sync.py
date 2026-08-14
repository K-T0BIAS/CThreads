"""
cthreads.sync method / barrier lowering.

- Lock / Event / RWLock methods on typed receivers (MethodTablePlugin)
- TBuffer* / TBuffer[T] methods
- ``__sync_state()`` kernel barrier
"""

from __future__ import annotations

import ast

from .....types import (
    PyCThreadsInternalType,
    PyType,
    is_tbuffer_pytype,
)
from ...context import TranslationContext
from ...include import add_include
from ...syntax.Op import Op
from ..base import CallPlugin, MethodOp, MethodTablePlugin, TranslateExpr, method_op


class SyncMethodPlugin(MethodTablePlugin):
    """`lock.acquire()` / `event.wait_for(t)` / RWLock methods."""

    tables = {
        "Lock": {
            "acquire": method_op("acquire", 0),
            "release": method_op("release", 0),
            "try_acquire": method_op("try_acquire", 0),
        },
        "Event": {
            "set": method_op("set", 0),
            "clear": method_op("clear", 0),
            "is_set": method_op("is_set", 0),
            "wait": method_op("wait", 0),
            "wait_for": method_op("wait_for", 1),
        },
        "RWLock": {
            "acquire_read": method_op("acquire_read", 0),
            "release_read": method_op("release_read", 0),
            "try_acquire_read": method_op("try_acquire_read", 0),
            "acquire_write": method_op("acquire_write", 0),
            "release_write": method_op("release_write", 0),
            "try_acquire_write": method_op("try_acquire_write", 0),
        },
    }

    def type_key(self, py_type: PyType) -> str | None:
        if isinstance(py_type, PyCThreadsInternalType) and py_type.name in self.tables:
            return py_type.name
        return None


def _read_at_emit(recv: str, args: list[str]) -> str:
    return f"(({recv}).get_read_slot()[{args[0]}])"


class TBufferMethodPlugin(MethodTablePlugin):
    """`buf.publish()` / `buf.read_at(i)` on TBuffer[...] or TBuffer* internals."""

    tables = {
        "TBuffer": {
            "publish": method_op("publish", 0),
            "generation": method_op("generation", 0),
            "capacity": method_op("capacity", 0),
            "read_at": MethodOp(
                emit=_read_at_emit,
                min_arity=1,
                max_arity=1,
            ),
        },
    }

    def type_key(self, py_type: PyType) -> str | None:
        if is_tbuffer_pytype(py_type):
            return "TBuffer"
        return None


class SyncStatePlugin(CallPlugin):
    """`__sync_state()` -> `cthreads::detail::__sync_state()` + syncState.hpp."""

    def try_lower(
        self,
        node: ast.Call,
        ctx: TranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        if not Op.is_builtin_call(node, "__sync_state"):
            return None
        if node.keywords or node.args:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                "__sync_state() takes no arguments"
            )
        add_include(
            ctx.body_includes,
            ctx.seen_body,
            '#include "sync/syncState.hpp"\n',
        )
        return "cthreads::detail::__sync_state()"
