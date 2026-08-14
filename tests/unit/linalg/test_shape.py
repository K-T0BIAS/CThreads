"""``linalg.Shape`` bindings."""

from __future__ import annotations

import pytest

from cthreads import linalg


def test_from_list_and_len():
    s = linalg.Shape([2, 3, 4])
    assert len(s) == 3
    assert s.ndim() == 3
    assert s.numel() == 24
    assert s[0] == 2
    assert s[1] == 3
    assert s[2] == 4


def test_from_scalar_dim():
    s = linalg.Shape(5)
    assert len(s) == 1
    assert s[0] == 5
    assert s.numel() == 5
    assert s.strides() == [1]


@pytest.mark.parametrize(
    "dims, strides",
    [
        ([3], [1]),
        ([2, 3], [3, 1]),
        ([2, 3, 4], [12, 4, 1]),
        ([1, 1, 1], [1, 1, 1]),
        ([8, 1], [1, 1]),
        ([1, 8], [8, 1]),
        ([2, 2, 2, 2], [8, 4, 2, 1]),
    ],
)
def test_c_order_strides(dims, strides):
    assert linalg.Shape(dims).strides() == strides


def test_empty_shape():
    s = linalg.Shape([])
    assert len(s) == 0
    assert s.ndim() == 0
    assert s.numel() == 0
    assert s.strides() == []


def test_zero_dim_numel():
    assert linalg.Shape([0]).numel() == 0
    assert linalg.Shape([2, 0, 3]).numel() == 0
    assert linalg.Shape([0, 0]).numel() == 0


def test_equality():
    assert linalg.Shape([2, 3]) == linalg.Shape([2, 3])
    assert not (linalg.Shape([2, 3]) == linalg.Shape([3, 2]))
    assert linalg.Shape(4) == linalg.Shape([4])


def test_index_error():
    s = linalg.Shape([2, 3])
    with pytest.raises(IndexError):
        _ = s[2]
    with pytest.raises(IndexError):
        _ = s[99]


def test_repr_contains_dims():
    r = repr(linalg.Shape([2, 3, 4]))
    assert "Shape" in r
    assert "2" in r and "3" in r and "4" in r
