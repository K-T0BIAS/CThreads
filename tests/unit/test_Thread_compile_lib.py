"""Unit tests for translation helpers (include / literal / source)."""

import ast
from pathlib import Path

import pytest

from cthreads.compiler.orchestrator.units import Handle, ThreadableUnit
from cthreads.compiler.translation import Cpp, Source, add_include, include_for
from cthreads.frontend.Registry import REGISTRY
from cthreads.types import PyInt, PyList, PyThreadable


def test_add_include_dedupes():
    bucket: list[str] = []
    seen: set[str] = set()
    add_include(bucket, seen, "#include <vector>\n#include <string>\n")
    add_include(bucket, seen, "#include <vector>\n")
    assert bucket == ["#include <vector>\n", "#include <string>\n"]


def test_include_for_stdlib_and_threadable(tmp_path):
    this_file = tmp_path / "out" / "x.hpp"
    this_file.parent.mkdir(parents=True)
    assert include_for(PyInt(), this_file) == ""
    assert "vector" in include_for(PyList(PyInt()), this_file)

    cls = type("P", (), {"__threadable": True})
    REGISTRY.register_threadable(cls)
    hpp = tmp_path / "__Threadable__" / "P.hpp"
    hpp.parent.mkdir(parents=True)
    hpp.write_text("struct P {};\n", encoding="utf-8")
    REGISTRY.threadable_units["P"] = ThreadableUnit(
        handle=Handle(name="P", path="p.py", target=cls),
        fields={},
        hpp_path=hpp,
        cpp_path=hpp.with_suffix(".cpp"),
    )
    text = include_for(PyThreadable("P"), this_file)
    assert "P.hpp" in text
    assert text.startswith("#include")


def test_include_for_shared_threadable(tmp_path):
    from cthreads.types import PyShared, Shared, hint_to_pytype

    this_file = tmp_path / "out" / "worker.hpp"
    this_file.parent.mkdir(parents=True)

    cls = type("Box", (), {"__threadable": True})
    REGISTRY.register_threadable(cls)
    hpp = tmp_path / "__Threadable__" / "Box.hpp"
    hpp.parent.mkdir(parents=True)
    hpp.write_text("struct Box { int n; };\n", encoding="utf-8")
    REGISTRY.threadable_units["Box"] = ThreadableUnit(
        handle=Handle(name="Box", path="box.py", target=cls),
        fields={},
        hpp_path=hpp,
        cpp_path=hpp.with_suffix(".cpp"),
    )

    py = hint_to_pytype(Shared[cls])
    assert isinstance(py, PyShared)
    assert "shared_host.hpp" in py.build_include()
    assert "Box.hpp" not in py.build_include()

    text = include_for(py, this_file)
    assert "shared_host.hpp" in text
    assert "Box.hpp" in text


def test_cpp_literal_types_and_escape():
    assert Cpp.literal(True) == "true"
    assert Cpp.literal(False) == "false"
    assert Cpp.literal(12) == "12"
    assert Cpp.literal(1.5) == "1.5"
    assert Cpp.literal('a"b\\c') == '"a\\"b\\\\c"'
    with pytest.raises(TypeError, match="None"):
        Cpp.literal(None)


def test_resolve_annotation_and_rejects_bad():
    g = {"int": int, "list": list}

    assert Source.resolve_annotation(ast.parse("int", mode="eval").body, g) is int
    assert Source.resolve_annotation(ast.parse("list[int]", mode="eval").body, g) == list[int]

    class Bad:
        pass

    with pytest.raises(TypeError):
        Source.resolve_annotation(ast.parse("Bad", mode="eval").body, {"Bad": Bad})


def test_parse_function_def_strips_decorator():
    def deco(f):
        return f

    @deco
    def sample(x: int) -> int:
        return x

    node = Source.parse_function(sample)
    assert isinstance(node, ast.FunctionDef)
    assert node.name == "sample"
    assert len(node.args.args) == 1
