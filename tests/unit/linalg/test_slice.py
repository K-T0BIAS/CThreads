"""`linalg.Slice` bindings."""

from __future__ import annotations

from cthreads import linalg


def test_default_slice():
    s = linalg.Slice()
    assert s.start == 0
    assert s.step == 1
    assert s.stop != 0
    assert "None" in repr(s)


def test_stop_only():
    s = linalg.Slice(4)
    assert s.start == 0
    assert s.stop == 4
    assert s.step == 1


def test_start_stop_step():
    s = linalg.Slice(1, 8, 2)
    assert s.start == 1
    assert s.stop == 8
    assert s.step == 2
    assert "1" in repr(s) and "8" in repr(s) and "2" in repr(s)


def test_default_step():
    s = linalg.Slice(2, 5)
    assert s.step == 1


def test_fields_mutable():
    s = linalg.Slice(0, 10, 1)
    s.start = 3
    s.stop = 7
    s.step = 2
    assert (s.start, s.stop, s.step) == (3, 7, 2)
