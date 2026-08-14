"""Storage sharing, contiguous copies, slice offsets."""

from __future__ import annotations

import pytest

from unit.linalg.helpers import ARRAY_PARAMS, assert_close, cast, filled


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_row_view_writes_through(Array):
    a, _ = filled(Array, [2, 3], start=1)
    row = a[0]
    row[1] = cast(Array, 99)
    assert a[0, 1] == cast(Array, 99)
    assert a[1, 1] != cast(Array, 99)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_slice_view_writes_through(Array):
    a, _ = filled(Array, [3, 4], start=1)
    sub = a[1:3, 1:3]
    sub[0, 0] = cast(Array, 70)
    assert a[1, 1] == cast(Array, 70)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_transpose_view_writes_through(Array):
    a, _ = filled(Array, [2, 3], start=1)
    t = a.transpose()
    t[2, 1] = cast(Array, 88)
    assert a[1, 2] == cast(Array, 88)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_contiguous_is_a_copy(Array):
    a, data = filled(Array, [2, 3], start=1)
    t = a.transpose()
    c = t.contiguous()
    assert c.is_contiguous() is True
    assert c.shape == t.shape
    assert_close(c, t, Array)
    c[0, 0] = cast(Array, 0)
    assert t[0, 0] == data[0][0]
    assert a[0, 0] == data[0][0]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_contiguous_of_contiguous_is_independent(Array):
    a, data = filled(Array, [2, 2], start=1)
    b = a.contiguous()
    b[0, 0] = cast(Array, 0)
    assert a[0, 0] == data[0][0]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_inplace_contiguous_on_strided(Array):
    a, data = filled(Array, [2, 3], start=1)
    t = a.transpose()
    assert t.is_contiguous() is False
    t._contiguous()
    assert t.is_contiguous() is True
    expected = [[data[i][j] for i in range(2)] for j in range(3)]
    assert_close(t, expected, Array)
    t[0, 0] = cast(Array, 0)
    assert a[0, 0] == data[0][0]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_inplace_contiguous_noop_on_dense(Array):
    a, data = filled(Array, [2, 3], start=1)
    a._contiguous()
    assert a.is_contiguous() is True
    assert_close(a, data, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_slice_offset(Array):
    a, data = filled(Array, [3, 4], start=1)
    assert a.offset == 0
    rows = a[1:]
    assert rows.offset == 4
    assert_close(rows, data[1:], Array)
    col = a[:, 2:]
    assert col.offset == 2
    first = a[1:, 2:]
    assert first.offset == 6
    assert first[0, 0] == data[1][2]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_stepped_slice_offset_and_stride(Array):
    a, data = filled(Array, [4], start=1)
    b = a[1::2]
    assert b.offset == 1
    assert b.strides == [2]
    assert_close(b, [data[1], data[3]], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_reshape_keeps_offset_of_slice(Array):
    a, data = filled(Array, [2, 3], start=1)
    tail = a[1:]
    r = tail.reshape([3])
    assert r.offset == 3
    assert_close(r, data[1], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_squeeze_keeps_shared_storage(Array):
    a, _ = filled(Array, [2, 3], start=1)
    u = a.unsqueeze(0)
    s = u.squeeze(0)
    s[0, 0] = cast(Array, 99)
    assert a[0, 0] == cast(Array, 99)
    assert u[0, 0, 0] == cast(Array, 99)
