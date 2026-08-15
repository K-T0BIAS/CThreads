"""Unit tests for cthreads.types."""

import pytest

from cthreads.types import (
    PyBool,
    PyCThreadsInternalType,
    PyDict,
    PyFloat,
    PyInt,
    PyList,
    PyString,
    PyThreadable,
    hint_to_pytype,
)


def test_primitive_to_cpp_decls():
    assert PyInt().to_cpp("x")[0] == "int x;"
    assert PyInt().to_cpp("x", "3")[0] == "int x = 3;"
    assert PyFloat().cpp_name == "double"
    assert PyBool().to_cpp("f", "true")[0] == "bool f = true;"
    decl, inc = PyString().to_cpp("s", '"hi"')
    assert decl == 'std::string s = "hi";'
    assert "string" in inc


def test_list_and_dict_cpp_names_and_includes():
    lst = PyList(PyInt())
    assert lst.cpp_name == "std::vector<int>"
    assert "vector" in lst.build_include()

    d = PyDict(PyString(), PyFloat())
    assert "std::unordered_map<std::string, double>" == d.cpp_name
    inc = d.build_include()
    assert "unordered_map" in inc
    assert "string" in inc


def test_threadable_identity_only():
    t = PyThreadable("Particle")
    assert t.cpp_name == "Particle"
    assert t.build_include() == ""


def test_hint_to_pytype_primitives_and_generics():
    assert isinstance(hint_to_pytype(int), PyInt)
    assert isinstance(hint_to_pytype(float), PyFloat)
    assert isinstance(hint_to_pytype(bool), PyBool)
    assert isinstance(hint_to_pytype(str), PyString)
    assert isinstance(hint_to_pytype(list[int]), PyList)
    assert hint_to_pytype(list[float]).inner_type.cpp_name == "double"
    assert isinstance(hint_to_pytype(dict[str, int]), PyDict)


def test_hint_to_pytype_threadable():
    class Box:
        pass

    Box.__threadable = True
    py = hint_to_pytype(Box)
    assert isinstance(py, PyThreadable)
    assert py.cpp_name == "Box"


def test_hint_to_pytype_set_rejected():
    with pytest.raises(TypeError, match="set is not supported"):
        hint_to_pytype(set[int])


def test_hint_to_pytype_unknown_rejected():
    class Nope:
        pass

    with pytest.raises(TypeError, match="Cannot map type"):
        hint_to_pytype(Nope)


def test_hint_to_pytype_tbuffer_internal():
    class TBufferF64:
        __cthreads_internal__ = True

    py = hint_to_pytype(TBufferF64)
    assert isinstance(py, PyCThreadsInternalType)
    assert py.cpp_name == "cthreads::sync::tripple_buffer<double>"
    inc = py.build_include()
    assert '#include "sync/t_buffer.hpp"' in inc

    class TBufferListF64:
        __cthreads_internal__ = True

    py_list = hint_to_pytype(TBufferListF64)
    assert "std::vector<double>" in py_list.cpp_name
    assert "#include <vector>" in py_list.build_include()


def test_hint_to_pytype_linalg_internal():
    class ArrayF32:
        __cthreads_internal__ = True

    py = hint_to_pytype(ArrayF32)
    assert isinstance(py, PyCThreadsInternalType)
    assert py.cpp_name == "cthreads::linalg::Array<float>"
    inc = py.build_include()
    assert '#include "linalg/array.hpp"' in inc

    class ArrayBool:
        __cthreads_internal__ = True

    py_bool = hint_to_pytype(ArrayBool)
    assert py_bool.cpp_name == "cthreads::linalg::Array<uint8_t>"
    assert "#include <cstdint>" in py_bool.build_include()

    class Shape:
        __cthreads_internal__ = True

    py_shape = hint_to_pytype(Shape)
    assert py_shape.cpp_name == "cthreads::linalg::Shape"
    assert '#include "linalg/shape.hpp"' in py_shape.build_include()

    class Slice:
        __cthreads_internal__ = True

    py_slice = hint_to_pytype(Slice)
    assert py_slice.cpp_name == "cthreads::linalg::Slice"
    assert '#include "linalg/slice.hpp"' in py_slice.build_include()


def test_hint_to_pytype_live_linalg_classes():
    from cthreads import linalg

    if linalg is None:
        pytest.skip("cthreads.linalg not built")
    py = hint_to_pytype(linalg.ArrayF32)
    assert isinstance(py, PyCThreadsInternalType)
    assert py.cpp_name == "cthreads::linalg::Array<float>"
    assert hint_to_pytype(linalg.Shape).cpp_name == "cthreads::linalg::Shape"
