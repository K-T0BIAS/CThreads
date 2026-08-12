"""Unit tests for cthreads.pyTypes."""

import pytest

from cthreads.pyTypes import (
    PyBool,
    PyCthreadsInternal,
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


def test_threadable_include_quotes():
    t = PyThreadable("Particle", "__Threadable__/Particle.hpp")
    assert t.build_include() == '#include "__Threadable__/Particle.hpp"\n'


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
    assert isinstance(py, PyCthreadsInternal)
    assert py.cpp_name == "cthreads::sync::tripple_buffer<double>"
    inc = py.build_include()
    assert '#include "sync/t_buffer.hpp"' in inc

    class TBufferListF64:
        __cthreads_internal__ = True

    py_list = hint_to_pytype(TBufferListF64)
    assert "std::vector<double>" in py_list.cpp_name
    assert "#include <vector>" in py_list.build_include()
