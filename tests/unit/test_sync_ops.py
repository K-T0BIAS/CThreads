"""Unit tests for cthreads.sync method lowering (syncOps + call + exprStmt)."""

from __future__ import annotations

import pytest

from cthreads.pyTypes import PyCthreadsInternal, PyFloat, PyInt, PyThreadable
from cthreads.Thread.compile.AstTranslators import call, exprStmt
from cthreads.Thread.compile.syncBindingTranslators import (
    EVENT_METHODS,
    LOCK_METHODS,
    RWLOCK_METHODS,
    resolve_sync_op,
)
from cthreads.Thread.compile.syncBindingTranslators.is_sync import is_sync_op
from helpers import make_ctx, parse_expr, parse_stmt, registered_threadable


def _lock_ty() -> PyCthreadsInternal:
    return PyCthreadsInternal("Lock", "cthreads::sync::Lock", "sync/pyLock.hpp")


def _event_ty() -> PyCthreadsInternal:
    return PyCthreadsInternal("Event", "cthreads::sync::Event", "sync/pyEvent.hpp")


def _rwlock_ty() -> PyCthreadsInternal:
    return PyCthreadsInternal("RWLock", "cthreads::sync::RWLock", "sync/pyRWLock.hpp")


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
    assert set(LOCK_METHODS) >= {"acquire", "release", "try_acquire"}
    assert set(EVENT_METHODS) >= {"set", "clear", "is_set", "wait", "wait_for"}
    assert set(RWLOCK_METHODS) >= {
        "acquire_read",
        "release_read",
        "try_acquire_read",
        "acquire_write",
        "release_write",
        "try_acquire_write",
    }


def test_resolve_lock_methods():
    ctx = _sync_ctx()
    assert resolve_sync_op(parse_expr("lock.acquire()"), ctx) is LOCK_METHODS["acquire"]
    assert resolve_sync_op(parse_expr("lock.release()"), ctx) is LOCK_METHODS["release"]
    assert resolve_sync_op(parse_expr("lock.try_acquire()"), ctx) is LOCK_METHODS["try_acquire"]
    assert is_sync_op(parse_expr("lock.acquire()"), ctx)


def test_resolve_returns_none_for_non_sync():
    ctx = make_ctx(symbols={"n": PyInt(), "lock": _lock_ty()})
    assert resolve_sync_op(parse_expr("n + 1"), ctx) is None
    assert resolve_sync_op(parse_expr("unknown()"), ctx) is None
    assert resolve_sync_op(parse_expr("n.acquire()"), ctx) is None


def test_resolve_rejects_unknown_lock_method():
    ctx = _sync_ctx()
    with pytest.raises(TypeError, match="unsupported Lock method"):
        resolve_sync_op(parse_expr("lock.foo()"), ctx)


def test_resolve_lock_arity_and_keywords():
    ctx = _sync_ctx()
    with pytest.raises(TypeError, match="Lock.acquire"):
        resolve_sync_op(parse_expr("lock.acquire(t)"), ctx)
    with pytest.raises(TypeError, match="keyword"):
        resolve_sync_op(parse_expr("ev.wait_for(seconds=t)"), ctx)


def test_resolve_event_wait_for_arity():
    ctx = _sync_ctx()
    assert resolve_sync_op(parse_expr("ev.wait_for(t)"), ctx) is EVENT_METHODS["wait_for"]
    with pytest.raises(TypeError, match="Event.wait_for"):
        resolve_sync_op(parse_expr("ev.wait_for()"), ctx)


def test_call_lock_event_rwlock():
    ctx = _sync_ctx()
    assert call.translate(parse_expr("lock.acquire()"), ctx) == "(lock).acquire()"
    assert call.translate(parse_expr("lock.release()"), ctx) == "(lock).release()"
    assert call.translate(parse_expr("lock.try_acquire()"), ctx) == "(lock).try_acquire()"

    assert call.translate(parse_expr("ev.set()"), ctx) == "(ev).set()"
    assert call.translate(parse_expr("ev.clear()"), ctx) == "(ev).clear()"
    assert call.translate(parse_expr("ev.is_set()"), ctx) == "(ev).is_set()"
    assert call.translate(parse_expr("ev.wait()"), ctx) == "(ev).wait()"
    assert call.translate(parse_expr("ev.wait_for(t)"), ctx) == "(ev).wait_for(t)"

    assert call.translate(parse_expr("rw.acquire_read()"), ctx) == "(rw).acquire_read()"
    assert call.translate(parse_expr("rw.release_write()"), ctx) == "(rw).release_write()"
    assert (
        call.translate(parse_expr("rw.try_acquire_write()"), ctx)
        == "(rw).try_acquire_write()"
    )


def test_expr_stmt_lock_acquire():
    ctx = _sync_ctx()
    lines = exprStmt.translate(parse_stmt("lock.acquire()"), ctx)
    assert lines == ["    (lock).acquire();"]


def test_attribute_receiver_uses_threadable_field():
    Lock = type("Lock", (), {"__cthreads_internal__": True})
    Holder = type(
        "Holder",
        (),
        {"__threadable": True, "__annotations__": {"lock": Lock}},
    )
    with registered_threadable(Holder):
        ctx = make_ctx(
            owner_name="Holder",
            symbols={"self": PyThreadable("Holder", "x")},
        )
        assert (
            resolve_sync_op(parse_expr("self.lock.acquire()"), ctx)
            is LOCK_METHODS["acquire"]
        )
        assert call.translate(parse_expr("self.lock.acquire()"), ctx) == (
            "(this->lock).acquire()"
        )


def test_untyped_attribute_receiver_is_none():
    ctx = make_ctx(symbols={"t": PyFloat()})
    assert resolve_sync_op(parse_expr("obj.lock.acquire()"), ctx) is None
