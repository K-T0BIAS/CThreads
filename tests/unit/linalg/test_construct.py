"""Constructors, from_list / to_list, dtypes, empty arrays."""

from __future__ import annotations

import pytest

from cthreads import linalg

from unit.linalg.helpers import ARRAY_PARAMS, assert_close, cast, filled, nest


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_construct_shape_strides(Array):
    a = Array([2, 3])
    assert a.shape == [2, 3]
    assert a.ndim == 2
    assert a.numel == 6
    assert a.strides == [3, 1]
    assert a.is_contiguous() is True
    assert a.offset == 0
    assert len(a) == 2


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_construct_tuple_shape(Array):
    a = Array((3, 4))
    assert a.shape == [3, 4]
    assert a.numel == 12


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_construct_1d(Array):
    a = Array([5])
    assert a.shape == [5]
    assert a.ndim == 1
    assert a.strides == [1]
    assert len(a) == 5


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_construct_3d(Array):
    a = Array([2, 3, 4])
    assert a.shape == [2, 3, 4]
    assert a.strides == [12, 4, 1]
    assert a.numel == 24
    assert len(a) == 2


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_from_list_to_list_roundtrip_2d(Array):
    data = nest([cast(Array, i) for i in range(1, 7)], [2, 3])
    a = Array.from_list(data)
    assert_close(a, data, Array)
    assert a.shape == [2, 3]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_from_list_1d(Array):
    data = [cast(Array, x) for x in (1, 2, 3, 4)]
    a = Array.from_list(data)
    assert a.shape == [4]
    assert_close(a, data, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_from_list_3d(Array):
    data = nest([cast(Array, i) for i in range(1, 25)], [2, 3, 4])
    a = Array.from_list(data)
    assert a.shape == [2, 3, 4]
    assert_close(a, data, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_from_list_tuple_nested(Array):
    a = Array.from_list(((cast(Array, 1), cast(Array, 2)), (cast(Array, 3), cast(Array, 4))))
    assert a.shape == [2, 2]
    assert_close(a, [[cast(Array, 1), cast(Array, 2)], [cast(Array, 3), cast(Array, 4)]], Array)


def test_from_list_rejects_scalar():
    with pytest.raises(TypeError):
        linalg.ArrayF32.from_list(1.0)


def test_from_list_rejects_string():
    with pytest.raises(TypeError):
        linalg.ArrayF32.from_list("abc")


def test_from_list_ragged_raises():
    with pytest.raises(ValueError):
        linalg.ArrayF32.from_list([[1.0, 2.0], [3.0]])


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_from_list_empty_1d(Array):
    a = Array.from_list([])
    assert a.shape == [0]
    assert a.numel == 0
    assert a.to_list() == []


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_zero_leading_dim(Array):
    a = Array([0, 3])
    assert a.shape == [0, 3]
    assert a.numel == 0
    assert a.to_list() == []


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_repr_mentions_shape_and_contiguous(Array):
    a = Array([2, 3])
    r = repr(a)
    assert Array.__name__ in r
    assert "2" in r and "3" in r
    assert "contiguous=True" in r
    t = a.transpose()
    assert "contiguous=False" in repr(t)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_filled_helper_matches(Array):
    a, nested = filled(Array, [2, 2], start=10)
    assert_close(a, nested, Array)
    assert a[0, 0] == cast(Array, 10)
    assert a[1, 1] == cast(Array, 13)


def test_f32_from_ints_promotes():
    a = linalg.ArrayF32.from_list([[1, 2], [3, 4]])
    assert a[0, 0] == pytest.approx(1.0)


def test_i32_keeps_ints():
    a = linalg.ArrayI32.from_list([[1, 2], [3, 4]])
    assert a[0, 0] == 1
    assert isinstance(a[0, 0], int)


def test_f64_precision():
    a = linalg.ArrayF64.from_list([1.0 / 3.0])
    assert a[0] == pytest.approx(1.0 / 3.0)
