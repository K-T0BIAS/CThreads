"""Per-dtype behavior and mixed-type rejection."""

from __future__ import annotations

import pytest

from cthreads import linalg


def test_cannot_add_f32_and_f64():
    a = linalg.ArrayF32.from_list([1.0, 2.0])
    b = linalg.ArrayF64.from_list([1.0, 2.0])
    with pytest.raises(TypeError):
        _ = a + b


def test_cannot_add_f32_and_i32():
    a = linalg.ArrayF32.from_list([1.0, 2.0])
    b = linalg.ArrayI32.from_list([1, 2])
    with pytest.raises(TypeError):
        _ = a + b


def test_cannot_matmul_mixed():
    a = linalg.ArrayF32.from_list([[1.0, 2.0], [3.0, 4.0]])
    b = linalg.ArrayI32.from_list([[1, 2], [3, 4]])
    with pytest.raises(TypeError):
        a.matmul(b)


def test_f32_scalar_accepts_python_int():
    a = linalg.ArrayF32.from_list([1.0, 2.0, 3.0])
    b = a + 1
    assert b.to_list() == pytest.approx([2.0, 3.0, 4.0])


def test_i32_rejects_float_scalar():
    a = linalg.ArrayI32.from_list([1, 2, 3])
    with pytest.raises(TypeError):
        _ = a + 1.5


def test_three_dtypes_independent_storage():
    f = linalg.ArrayF32.from_list([1.0])
    d = linalg.ArrayF64.from_list([1.0])
    i = linalg.ArrayI32.from_list([1])
    f[0] = 9.0
    assert d[0] == pytest.approx(1.0)
    assert i[0] == 1
