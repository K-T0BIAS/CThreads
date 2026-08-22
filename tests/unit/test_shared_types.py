"""Tests for Shared[T] validation in @Thread / @Threadable frontends."""

import pytest

from cthreads import Thread, Threadable, Shared
from cthreads.frontend.Threadable.lib import is_threadable
from cthreads.types import hint_to_pytype, is_shared_pytype


def test_is_threadable_accepts_shared():
    assert is_threadable(Shared[list[int]])
    assert is_threadable(Shared[int])


def test_is_threadable_shared_rejects_bad_inner():
    class Nope:
        pass

    with pytest.raises(TypeError, match="not allowed"):
        is_threadable(Shared[Nope])


def test_is_shared_pytype():
    py = hint_to_pytype(Shared[float])
    assert is_shared_pytype(py)


def test_thread_decorator_accepts_shared_param():
    @Thread
    def worker(head: Shared[list[int]], n: int) -> None:
        pass

    assert worker.__threaded is True


def test_thread_decorator_accepts_shared_return():
    @Thread
    def pick() -> Shared[int]:
        return 1

    assert pick.__threaded is True


def test_threadable_field_shared():
    @Threadable
    class State:
        head: Shared[list[int]]

    assert State.__threadable is True
