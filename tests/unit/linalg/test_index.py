"""Indexing: getitem / setitem, slices, negatives, errors."""

from __future__ import annotations

import pytest

from cthreads import linalg

from unit.linalg.helpers import ARRAY_PARAMS, assert_close, cast, filled


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_scalar_index_1d(Array):
    a, data = filled(Array, [4])
    assert a[0] == data[0]
    assert a[3] == data[3]
    assert a[-1] == data[3]
    assert a[-4] == data[0]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_scalar_index_1d_oob(Array):
    a, _ = filled(Array, [4])
    with pytest.raises(IndexError):
        _ = a[4]
    with pytest.raises(IndexError):
        _ = a[-5]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_setitem_1d(Array):
    a, _ = filled(Array, [3])
    a[1] = cast(Array, 99)
    assert a[1] == cast(Array, 99)
    a[-1] = cast(Array, 7)
    assert a[2] == cast(Array, 7)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_tuple_index_2d(Array):
    a, data = filled(Array, [2, 3])
    assert a[0, 0] == data[0][0]
    assert a[0, 2] == data[0][2]
    assert a[1, 1] == data[1][1]
    assert a[-1, -1] == data[1][2]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_list_index_key(Array):
    a, data = filled(Array, [2, 3])
    assert a[[0, 2]] == data[0][2]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_setitem_2d(Array):
    a, _ = filled(Array, [2, 3])
    a[1, 0] = cast(Array, 42)
    assert a[1, 0] == cast(Array, 42)
    a[-1, -1] = cast(Array, 8)
    assert a[1, 2] == cast(Array, 8)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_tuple_index_3d(Array):
    a, data = filled(Array, [2, 3, 4])
    assert a[1, 2, 3] == data[1][2][3]
    a[0, 1, 2] = cast(Array, 50)
    assert a[0, 1, 2] == cast(Array, 50)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_row_index_squeezes(Array):
    a, data = filled(Array, [2, 3])
    row = a[1]
    assert row.shape == [3]
    assert_close(row, data[1], Array)
    row2 = a[-2]
    assert_close(row2, data[0], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_mixed_int_slice_squeezes(Array):
    a, data = filled(Array, [2, 3, 4])
    plane = a[1, :, :]
    assert plane.shape == [3, 4]
    assert_close(plane, data[1], Array)
    col = a[:, 1]
    assert col.shape == [2, 4]
    vec = a[0, :, 2]
    assert vec.shape == [3]
    assert_close(vec, [data[0][i][2] for i in range(3)], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_full_slice_is_view(Array):
    a, data = filled(Array, [2, 3])
    b = a[:]
    assert b.shape == [2, 3]
    assert_close(b, data, Array)
    b[0, 0] = cast(Array, 99)
    assert a[0, 0] == cast(Array, 99)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_row_slice_contiguous(Array):
    a, data = filled(Array, [3, 4])
    rows = a[1:3]
    assert rows.shape == [2, 4]
    assert rows.is_contiguous() is True
    assert_close(rows, data[1:3], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_column_slice_strided(Array):
    a, data = filled(Array, [3, 4])
    cols = a[:, 0:4:2]
    assert cols.shape == [3, 2]
    assert cols.is_contiguous() is False
    expected = [[data[i][0], data[i][2]] for i in range(3)]
    assert_close(cols, expected, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_step_on_1d(Array):
    a, data = filled(Array, [8])
    b = a[1:7:2]
    assert b.shape == [3]
    assert_close(b, [data[1], data[3], data[5]], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_negative_slice_bounds(Array):
    a, data = filled(Array, [5])
    b = a[-3:-1]
    assert_close(b, data[-3:-1], Array)
    c = a[:-1]
    assert_close(c, data[:-1], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_empty_slice(Array):
    a, _ = filled(Array, [4])
    b = a[2:2]
    assert b.shape == [0]
    assert b.to_list() == []


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_slice_clamps_past_end(Array):
    a, data = filled(Array, [3])
    b = a[1:100]
    assert_close(b, data[1:], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_negative_step_rejected(Array):
    a, _ = filled(Array, [4])
    with pytest.raises(RuntimeError, match="step"):
        _ = a[::-1]
    with pytest.raises(ValueError, match="step"):
        _ = a[::0]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_too_many_slice_axes(Array):
    a, _ = filled(Array, [2, 3])
    with pytest.raises(RuntimeError, match="more axes"):
        _ = a[:, :, :]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_bad_index_type(Array):
    a, _ = filled(Array, [2, 3])
    with pytest.raises(TypeError):
        _ = a["x"]
    with pytest.raises(TypeError):
        _ = a[1.5]
    with pytest.raises(TypeError):
        a[0:1] = cast(Array, 1)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_setitem_requires_full_index(Array):
    a, _ = filled(Array, [2, 3])
    with pytest.raises(TypeError):
        a[0] = cast(Array, 1)
    with pytest.raises(RuntimeError):
        a[0:1, 0] = cast(Array, 1)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_index_oob_2d(Array):
    a, _ = filled(Array, [2, 3])
    with pytest.raises(IndexError):
        _ = a[2, 0]
    with pytest.raises(IndexError):
        _ = a[0, 3]
    with pytest.raises(IndexError):
        a[0, 3] = cast(Array, 1)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_cpp_slice_object_not_valid_key(Array):
    a, _ = filled(Array, [4])
    with pytest.raises(TypeError):
        _ = a[linalg.Slice(2)]
