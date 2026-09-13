"""Unit tests for GPU Glsl type helpers."""

from __future__ import annotations

import pytest

from cthreads.gpu.compiler.translation.Glsl import (
    align_bytes,
    elem_type_name,
    size_bytes,
    type_name,
)
from cthreads.types import PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyThreadable


@pytest.mark.parametrize(
    "py, name, nbytes",
    [
        (PyInt(), "int", 4),
        (PyFloat(), "float", 4),
        (PyBool(), "bool", 4),
    ],
)
def test_scalar_type_name_and_size(py, name, nbytes):
    assert type_name(py) == name
    assert size_bytes(py) == nbytes
    assert align_bytes(py) == nbytes


def test_float_is_glsl_float_not_double():
    assert type_name(PyFloat()) == "float"
    assert size_bytes(PyFloat()) == 4


@pytest.mark.parametrize(
    "inner, elem",
    [
        (PyInt(), "int"),
        (PyFloat(), "float"),
        (PyBool(), "bool"),
    ],
)
def test_elem_type_name(inner, elem):
    assert elem_type_name(PyList(inner)) == elem


def test_type_name_rejects_list():
    with pytest.raises(TypeError, match="list is not a scalar"):
        type_name(PyList(PyFloat()))


def test_size_bytes_rejects_list():
    with pytest.raises(TypeError, match="list has no single scalar size"):
        size_bytes(PyList(PyInt()))


def test_align_bytes_rejects_list():
    with pytest.raises(TypeError, match="list has no single scalar size"):
        align_bytes(PyList(PyBool()))


def test_elem_type_name_rejects_non_list():
    with pytest.raises(TypeError, match="expects PyList"):
        elem_type_name(PyInt())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [
        PyString(),
        PyDict(PyString(), PyInt()),
        PyThreadable("T"),
    ],
)
def test_unsupported_scalar_types(bad):
    with pytest.raises(TypeError, match="not supported"):
        type_name(bad)


@pytest.mark.parametrize(
    "bad",
    [
        PyString(),
        PyDict(PyString(), PyInt()),
        PyThreadable("T"),
    ],
)
def test_unsupported_size_bytes(bad):
    with pytest.raises(TypeError, match="unsupported|not supported"):
        size_bytes(bad)


def test_nested_list_elem_rejected():
    with pytest.raises(TypeError):
        elem_type_name(PyList(PyList(PyInt())))


def test_type_name_inner_of_list_ok():
    assert type_name(PyList(PyFloat()).inner_type) == "float"
