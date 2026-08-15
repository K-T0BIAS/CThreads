"""Unit tests for cthreads.sync method lowering (Sync plugin + Syntax)."""

from __future__ import annotations

import pytest

from cthreads.compiler.translation.plugins.sync.Sync import SyncMethodPlugin
from cthreads.compiler.translation.syntax import Syntax
from cthreads.types import PyCThreadsInternalType, PyFloat, PyInt, PyThreadable, hint_to_pytype
from helpers import make_ctx, parse_expr, parse_stmt, registered_threadable


def _lock_ty() -> PyCThreadsInternalType:
    return hint_to_pytype(type("Lock", (), {"__cthreads_internal__": True}))


def _event_ty() -> PyCThreadsInternalType:
    return hint_to_pytype(type("Event", (), {"__cthreads_internal__": True}))


def _rwlock_ty() -> PyCThreadsInternalType:
    return hint_to_pytype(type("RWLock", (), {"__cthreads_internal__": True}))


def _sync_ctx(**extra):
    return make_ctx(
        symbols={
            "lock": _lock_ty(),
            "ev": _event_ty(),
            "rw": _rwlock_ty(),
            "t": PyFloat(),
            **extra,
        }
    )


def test_sync_method_tables_cover_core_ops():
    tables = SyncMethodPlugin.tables
    assert set(tables["Lock"]) >= {"acquire", "release", "try_acquire"}
    assert set(tables["Event"]) >= {"set", "clear", "is_set", "wait", "wait_for"}
    assert set(tables["RWLock"]) >= {
        "acquire_read",
        "release_read",
        "try_acquire_read",
        "acquire_write",
        "release_write",
        "try_acquire_write",
    }


def test_call_lock_methods():
    ctx = _sync_ctx()
    assert Syntax.expr(parse_expr("lock.acquire()"), ctx) == "(lock).acquire()"
    assert Syntax.expr(parse_expr("lock.release()"), ctx) == "(lock).release()"
    assert Syntax.expr(parse_expr("lock.try_acquire()"), ctx) == "(lock).try_acquire()"


def test_call_rejects_unknown_lock_method():
    ctx = _sync_ctx()
    with pytest.raises(TypeError, match="unknown method|unsupported"):
        Syntax.expr(parse_expr("lock.foo()"), ctx)


def test_call_lock_arity_and_keywords():
    ctx = _sync_ctx()
    with pytest.raises(TypeError, match="acquire"):
        Syntax.expr(parse_expr("lock.acquire(t)"), ctx)
    with pytest.raises(TypeError, match="keyword"):
        Syntax.expr(parse_expr("lock.acquire(x=1)"), ctx)


def test_call_event_and_rwlock():
    ctx = _sync_ctx()
    assert Syntax.expr(parse_expr("ev.set()"), ctx) == "(ev).set()"
    assert Syntax.expr(parse_expr("ev.wait_for(t)"), ctx) == "(ev).wait_for(t)"
    assert Syntax.expr(parse_expr("rw.acquire_read()"), ctx) == "(rw).acquire_read()"
    assert Syntax.expr(parse_expr("rw.release_write()"), ctx) == "(rw).release_write()"


def test_non_sync_receiver_is_unsupported_or_field():
    ctx = make_ctx(symbols={"n": PyInt(), "lock": _lock_ty()})
    with pytest.raises(TypeError, match="unsupported"):
        Syntax.expr(parse_expr("n.acquire()"), ctx)


def test_expr_stmt_sync_call():
    ctx = _sync_ctx()
    assert Syntax.stmt(parse_stmt("lock.acquire()"), ctx) == [
        "    (lock).acquire();"
    ]


def test_sync_on_threadable_field():
    Body = type(
        "SyncBody",
        (),
        {
            "__threadable": True,
            "__annotations__": {
                "lock": type("Lock", (), {"__cthreads_internal__": True})
            },
        },
    )
    with registered_threadable(Body):
        ctx = make_ctx(
            owner_name="SyncBody",
            symbols={"self": PyThreadable("SyncBody")},
        )
        assert Syntax.expr(parse_expr("self.lock.acquire()"), ctx) == (
            "(this->lock).acquire()"
        )
