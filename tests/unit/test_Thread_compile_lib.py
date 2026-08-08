"""Unit tests for Thread.compile.lib helpers."""

import ast
import textwrap

import pytest

from cthreads.pyTypes import PyFloat, PyInt, PyList, PyString, PyThreadable
from cthreads.Thread.compile.lib import (
    add_include,
    cpp_literal,
    include_for,
    parse_function_def,
    resolve_annotation,
)


def test_add_include_dedupes():
    bucket: list[str] = []
    seen: set[str] = set()
    add_include(bucket, seen, "#include <vector>\n#include <string>\n")
    add_include(bucket, seen, "#include <vector>\n")
    assert bucket == ["#include <vector>\n", "#include <string>\n"]


def test_include_for_stdlib_and_threadable():
    assert include_for(PyInt()) == ""
    assert "vector" in include_for(PyList(PyInt()))
    assert include_for(PyThreadable("P", "x")) == '#include "../__Threadable__/P.hpp"\n'


def test_cpp_literal_types_and_escape():
    assert cpp_literal(True) == "true"
    assert cpp_literal(False) == "false"
    assert cpp_literal(12) == "12"
    assert cpp_literal(1.5) == "1.5"
    assert cpp_literal('a"b\\c') == '"a\\"b\\\\c"'
    with pytest.raises(TypeError, match="None"):
        cpp_literal(None)


def test_resolve_annotation_and_rejects_bad():
    g = {"int": int, "list": list}

    assert resolve_annotation(ast.parse("int", mode="eval").body, g) is int
    assert resolve_annotation(ast.parse("list[int]", mode="eval").body, g) == list[int]

    class Bad:
        pass

    with pytest.raises(TypeError):
        resolve_annotation(ast.parse("Bad", mode="eval").body, {"Bad": Bad})


def test_parse_function_def_strips_decorator():
    def deco(f):
        return f

    @deco
    def sample(x: int) -> int:
        return x

    node = parse_function_def(sample)
    assert isinstance(node, ast.FunctionDef)
    assert node.name == "sample"
    assert len(node.args.args) == 1
