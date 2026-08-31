"""Shared helpers for `cthreads.linalg` runtime tests."""

from __future__ import annotations

import pytest

from cthreads import linalg

ARRAYS = (linalg.ArrayF32, linalg.ArrayF64, linalg.ArrayI32)
ARRAY_IDS = ("f32", "f64", "i32")
ARRAY_PARAMS = [pytest.param(cls, id=name) for cls, name in zip(ARRAYS, ARRAY_IDS)]


def is_int(Array) -> bool:
    return Array is linalg.ArrayI32


def cast(Array, value):
    return int(value) if is_int(Array) else float(value)


def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n if shape else 0


def nest(flat, shape):
    if not shape:
        return list(flat)
    if len(shape) == 1:
        return list(flat)
    step = numel(shape[1:])
    return [nest(flat[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]


def flatten(nested):
    if not isinstance(nested, list):
        return [nested]
    out = []
    for item in nested:
        out.extend(flatten(item))
    return out


def filled(Array, shape, start=1):
    n = numel(shape)
    data = [cast(Array, start + i) for i in range(n)]
    nested = nest(data, shape)
    return Array.from_list(nested), nested


def assert_close(got, expected, Array=None, rel=1e-5, abs_=1e-6):
    if hasattr(got, "to_list"):
        got = got.to_list()
    if hasattr(expected, "to_list"):
        expected = expected.to_list()
    if isinstance(expected, (list, tuple)):
        assert isinstance(got, list)
        assert len(got) == len(expected)
        for g, e in zip(got, expected):
            assert_close(g, e, Array=Array, rel=rel, abs_=abs_)
        return
    if Array is not None and is_int(Array):
        assert got == expected
        return
    assert got == pytest.approx(expected, rel=rel, abs=abs_)


def matmul_2d(a, b):
    m = len(a)
    k = len(a[0])
    n = len(b[0])
    out = [[type(a[0][0])(0) for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = type(a[0][0])(0)
            for t in range(k):
                s += a[i][t] * b[t][j]
            out[i][j] = s
    return out


def apply_last_axis(nested, fn):
    if nested and isinstance(nested[0], list) and nested[0] and not isinstance(nested[0][0], list):
        return [fn(row) for row in nested]
    if nested and isinstance(nested[0], list):
        return [apply_last_axis(row, fn) for row in nested]
    return fn(nested)


def map_nested_2(a, b, fn):
    if isinstance(a, list):
        return [map_nested_2(x, y, fn) for x, y in zip(a, b)]
    return fn(a, b)


def map_nested(a, fn):
    if isinstance(a, list):
        return [map_nested(x, fn) for x in a]
    return fn(a)
