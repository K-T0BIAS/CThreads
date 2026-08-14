"""dot and cross: 1D, ND last-axis, strided, inplace, errors."""

from __future__ import annotations

import pytest

from unit.linalg.helpers import ARRAY_PARAMS, assert_close, cast, filled


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_dot_1d(Array):
    x = Array.from_list([cast(Array, 1), cast(Array, 2), cast(Array, 3)])
    y = Array.from_list([cast(Array, 4), cast(Array, 5), cast(Array, 6)])
    d = x.dot(y)
    assert d.shape == [1]
    assert d[0] == cast(Array, 32)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
@pytest.mark.parametrize("n", [1, 3, 4, 7, 8, 9, 16, 17, 32])
def test_dot_1d_lengths(Array, n):
    a, da = filled(Array, [n], start=1)
    b, db = filled(Array, [n], start=2)
    d = a.dot(b)
    assert d.shape == [1]
    assert_close(d[0], _dot(da, db), Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_dot_2d_contracts_last(Array):
    a, da = filled(Array, [2, 3], start=1)
    b, db = filled(Array, [2, 3], start=1)
    d = a.dot(b)
    assert d.shape == [2]
    expected = [_dot(da[i], db[i]) for i in range(2)]
    assert_close(d, expected, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_dot_3d_contracts_last(Array):
    a, da = filled(Array, [2, 2, 4], start=1)
    d = a.dot(a)
    assert d.shape == [2, 2]
    expected = [[_dot(da[i][j], da[i][j]) for j in range(2)] for i in range(2)]
    assert_close(d, expected, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_dot_strided(Array):
    a, data = filled(Array, [3, 2], start=1)
    t = a.transpose()  # 2x3
    d = t.dot(t)
    assert d.shape == [2]
    rows = [[data[i][j] for i in range(3)] for j in range(2)]
    expected = [_dot(rows[0], rows[0]), _dot(rows[1], rows[1])]
    assert_close(d, expected, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_dot_shape_mismatch(Array):
    a, _ = filled(Array, [2, 3])
    b, _ = filled(Array, [3, 2])
    with pytest.raises(RuntimeError, match="shape mismatch"):
        a.dot(b)
    x, _ = filled(Array, [3])
    y, _ = filled(Array, [4])
    with pytest.raises(RuntimeError, match="numel mismatch"):
        x.dot(y)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_inplace_dot(Array):
    x = Array.from_list([cast(Array, 1), cast(Array, 2), cast(Array, 3)])
    y = Array.from_list([cast(Array, 4), cast(Array, 5), cast(Array, 6)])
    x._dot(y)
    assert x.shape == [1]
    assert x[0] == cast(Array, 32)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_cross_1d(Array):
    a = Array.from_list([cast(Array, 1), cast(Array, 2), cast(Array, 3)])
    b = Array.from_list([cast(Array, 4), cast(Array, 5), cast(Array, 6)])
    c = a.cross(b)
    assert c.shape == [3]
    assert_close(c, _cross([1, 2, 3], [4, 5, 6]), Array)
    assert_close(a, [1, 2, 3], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_cross_batched(Array):
    a, da = filled(Array, [2, 3], start=1)
    b, db = filled(Array, [2, 3], start=4)
    c = a.cross(b)
    assert c.shape == [2, 3]
    expected = [_cross(da[i], db[i]) for i in range(2)]
    assert_close(c, expected, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_cross_3d_batch(Array):
    a, da = filled(Array, [2, 2, 3], start=1)
    b, db = filled(Array, [2, 2, 3], start=2)
    c = a.cross(b)
    assert c.shape == [2, 2, 3]
    expected = [[_cross(da[i][j], db[i][j]) for j in range(2)] for i in range(2)]
    assert_close(c, expected, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_cross_strided(Array):
    a = Array.from_list([
        [cast(Array, 1), cast(Array, 0)],
        [cast(Array, 0), cast(Array, 1)],
        [cast(Array, 0), cast(Array, 0)],
    ])
    t = a.transpose()  # (2, 3) -> [[1,0,0],[0,1,0]]
    c = t.cross(t)
    assert c.shape == [2, 3]
    expected = [_cross([1, 0, 0], [1, 0, 0]), _cross([0, 1, 0], [0, 1, 0])]
    assert_close(c, expected, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_cross_last_dim_not_3(Array):
    a, _ = filled(Array, [2, 4])
    with pytest.raises(RuntimeError, match="last dimension"):
        a.cross(a)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_cross_shape_mismatch(Array):
    a, _ = filled(Array, [2, 3])
    b, _ = filled(Array, [3])
    with pytest.raises(RuntimeError, match="shape mismatch"):
        a.cross(b)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_inplace_cross(Array):
    a = Array.from_list([cast(Array, 1), cast(Array, 0), cast(Array, 0)])
    b = Array.from_list([cast(Array, 0), cast(Array, 1), cast(Array, 0)])
    a._cross(b)
    assert_close(a, [0, 0, 1], Array)
