"""Unit tests for Threadable / type helpers."""

import pytest

from cthreads.frontend.Threadable.lib import is_threadable
from cthreads.types import is_internal_cthreads_type


def test_is_threadable_whitelist_and_generics():
    assert is_threadable(int) is True
    assert is_threadable(list[int]) is True
    assert is_threadable(dict[str, float]) is True


def test_is_threadable_threadable_class():
    class P:
        pass

    P.__threadable = True
    assert is_threadable(P) is True


def test_is_threadable_rejects_plain_class_and_bad_generic():
    class P:
        pass

    with pytest.raises(TypeError):
        is_threadable(P)

    with pytest.raises(TypeError):
        is_threadable(tuple[int, int])


def test_is_internal_cthreads_type():
    class Lock:
        pass

    assert is_internal_cthreads_type(Lock) is False
    Lock.__cthreads_internal__ = True
    assert is_internal_cthreads_type(Lock) is True
    assert is_threadable(Lock) is True
