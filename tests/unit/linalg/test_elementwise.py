"""Elementwise + scalar math, SIMD tails, strided, inplace, numel-1 broadcast."""

from __future__ import annotations

import pytest

from unit.linalg.helpers import (
    ARRAY_PARAMS,
    assert_close,
    cast,
    filled,
    is_int,
    map_nested,
    map_nested_2,
)


OPS = (
    ("add", lambda a, b: a + b, lambda x, y: x + y, "_add"),
    ("sub", lambda a, b: a - b, lambda x, y: x - y, "_sub"),
    ("mul", lambda a, b: a * b, lambda x, y: x * y, "_mul"),
    ("div", lambda a, b: a / b, lambda x, y: x / y, "_div"),
)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
@pytest.mark.parametrize("name,py_op,elem,inplace", OPS)
def test_elementwise_same_shape(Array, name, py_op, elem, inplace):
    a, da = filled(Array, [2, 3], start=10)
    b, db = filled(Array, [2, 3], start=2)
    if name == "div" and is_int(Array):
        expected = map_nested_2(da, db, lambda x, y: int(x / y))
    else:
        expected = map_nested_2(da, db, elem)
    assert_close(py_op(a, b), expected, Array)
    assert_close(a, da, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
@pytest.mark.parametrize("name,py_op,elem,inplace", OPS)
def test_scalar_ops(Array, name, py_op, elem, inplace):
    a, da = filled(Array, [2, 3], start=10)
    s = cast(Array, 2)
    if name == "div" and is_int(Array):
        expected = map_nested(da, lambda x: int(x / s))
    else:
        expected = map_nested(da, lambda x: elem(x, s))
    assert_close(py_op(a, s), expected, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_neg(Array):
    a, da = filled(Array, [2, 3], start=1)
    assert_close(-a, map_nested(da, lambda x: -x), Array)
    assert_close(a, da, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
@pytest.mark.parametrize("n", [1, 3, 4, 7, 8, 9, 15, 16, 17, 32, 33])
def test_simd_tail_add_1d(Array, n):
    a, da = filled(Array, [n], start=1)
    b, db = filled(Array, [n], start=3)
    expected = [x + y for x, y in zip(da, db)]
    assert_close(a + b, expected, Array)
    assert_close(a + cast(Array, 5), [x + cast(Array, 5) for x in da], Array)
    assert_close(-a, [-x for x in da], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_numel1_broadcast(Array):
    a, da = filled(Array, [2, 3], start=4)
    s = Array.from_list([cast(Array, 3)])
    expected = map_nested(da, lambda x: x + cast(Array, 3))
    assert_close(a + s, expected, Array)
    assert_close(a * s, map_nested(da, lambda x: x * cast(Array, 3)), Array)
    assert_close(a - s, map_nested(da, lambda x: x - cast(Array, 3)), Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_shape_mismatch_raises(Array):
    a, _ = filled(Array, [2, 3])
    b, _ = filled(Array, [3, 2])
    with pytest.raises(RuntimeError, match="add"):
        _ = a + b
    with pytest.raises(RuntimeError, match="sub"):
        _ = a - b
    with pytest.raises(RuntimeError, match="mul"):
        _ = a * b
    with pytest.raises(RuntimeError, match="div"):
        _ = a / b


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_no_general_broadcast(Array):
    a = Array.from_list([[cast(Array, 1)], [cast(Array, 2)], [cast(Array, 3)], [cast(Array, 4)]])
    b = Array.from_list([[cast(Array, 10), cast(Array, 20), cast(Array, 30)]])
    with pytest.raises(RuntimeError, match="add"):
        _ = a + b


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_strided_elementwise(Array):
    a, data = filled(Array, [2, 3], start=1)
    t = a.transpose()
    s = t + t
    expected = [[data[i][j] * 2 for i in range(2)] for j in range(3)]
    assert_close(s, expected, Array)
    assert s.is_contiguous() is True
    d = t * cast(Array, 3)
    expected_s = [[data[i][j] * 3 for i in range(2)] for j in range(3)]
    assert_close(d, expected_s, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_inplace_array_ops(Array):
    a, da = filled(Array, [2, 2], start=10)
    b, db = filled(Array, [2, 2], start=1)
    a._add(b)
    assert_close(a, map_nested_2(da, db, lambda x, y: x + y), Array)
    a, da = filled(Array, [2, 2], start=10)
    a._sub(b)
    assert_close(a, map_nested_2(da, db, lambda x, y: x - y), Array)
    a, da = filled(Array, [2, 2], start=10)
    a._mul(b)
    assert_close(a, map_nested_2(da, db, lambda x, y: x * y), Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_inplace_scalar_ops(Array):
    a, da = filled(Array, [3], start=6)
    a._add(cast(Array, 2))
    assert_close(a, [x + cast(Array, 2) for x in da], Array)
    a, da = filled(Array, [3], start=6)
    a._mul(cast(Array, 2))
    assert_close(a, [x * 2 for x in da], Array)
    a, da = filled(Array, [3], start=6)
    a._neg()
    assert_close(a, [-x for x in da], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_inplace_strided_writes_through(Array):
    a, data = filled(Array, [2, 3], start=1)
    t = a.transpose()
    t._add(cast(Array, 1))
    expected = map_nested(data, lambda x: x + cast(Array, 1))
    assert_close(a, expected, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_out_of_place_does_not_alias(Array):
    a, data = filled(Array, [2, 3], start=1)
    b = a + cast(Array, 1)
    b[0, 0] = cast(Array, 0)
    assert a[0, 0] == data[0][0]


def test_float_div_fraction():
    from cthreads import linalg

    a = linalg.ArrayF32.from_list([1.0, 2.0, 3.0])
    assert_close(a / 2.0, [0.5, 1.0, 1.5], linalg.ArrayF32)


def test_int_div_truncates():
    from cthreads import linalg

    a = linalg.ArrayI32.from_list([7, 8, 9])
    assert a.to_list() == [7, 8, 9]
    assert (a / 2).to_list() == [3, 4, 4]


def test_float_div_by_zero_is_inf():
    from cthreads import linalg

    a = linalg.ArrayF32.from_list([1.0, -2.0])
    out = (a / 0.0).to_list()
    assert out[0] == float("inf")
    assert out[1] == float("-inf")
