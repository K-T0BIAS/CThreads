"""Unit tests for free @Thread -> free @Thread call lowering."""

from __future__ import annotations

from pathlib import Path

import pytest

from cthreads.compiler.orchestrator.units import Handle, ThreadUnit, ThreadableUnit
from cthreads.compiler.translation.syntax import Syntax
from cthreads.frontend.Registry import REGISTRY
from cthreads.types import PyFloat, PyInt, PyList
from helpers import make_ctx, parse_expr


def _threaded(name: str = "helper"):
    def fn(x: float) -> float:
        return x

    fn.__name__ = name
    fn.__qualname__ = name
    fn.__threaded = True
    return fn


def _register_free(fn, *, hpp: Path, cpp: Path | None = None) -> ThreadUnit:
    unit = ThreadUnit(
        handle=Handle(name=fn.__qualname__, path="dummy.py", target=fn),
        owner=None,
        params=[],
        return_type=None,
        hpp_path=hpp,
        cpp_path=cpp or hpp.with_suffix(".cpp"),
    )
    REGISTRY.thread_units[fn.__qualname__] = unit
    return unit


def test_free_thread_call_emits_direct_cpp_and_include(tmp_path):
    helper = _threaded("helper")
    hpp = tmp_path / "__Thread__" / "helper.hpp"
    hpp.parent.mkdir()
    hpp.write_text("", encoding="utf-8")
    _register_free(helper, hpp=hpp)

    ctx = make_ctx(
        func_name="main",
        symbols={"x": PyFloat()},
        globals_extra={"helper": helper},
        this_file=tmp_path / "__Thread__" / "main.hpp",
    )
    assert Syntax.expr(parse_expr("helper(x)"), ctx) == "helper(x)"
    assert any("helper.hpp" in line for line in ctx.body_includes)


def test_same_file_call_skips_self_include(tmp_path):
    helper = _threaded("helper")
    hpp = tmp_path / "__Thread__" / "helper.hpp"
    hpp.parent.mkdir()
    hpp.write_text("", encoding="utf-8")
    _register_free(helper, hpp=hpp)

    ctx = make_ctx(
        func_name="helper",
        symbols={"x": PyFloat()},
        globals_extra={"helper": helper},
        this_file=hpp,
    )
    assert Syntax.expr(parse_expr("helper(x)"), ctx) == "helper(x)"
    assert ctx.body_includes == []


def test_non_threaded_name_falls_through_to_len():
    ctx = make_ctx(symbols={"xs": PyList(PyInt())})
    assert Syntax.expr(parse_expr("len(xs)"), ctx) == "(xs).size()"


def test_plain_python_function_is_not_lowered():
    def helper(x: float) -> float:
        return x

    ctx = make_ctx(
        func_name="main",
        symbols={"x": PyFloat()},
        globals_extra={"helper": helper},
    )
    with pytest.raises(TypeError, match="unsupported call"):
        Syntax.expr(parse_expr("helper(x)"), ctx)


def test_unknown_name_falls_through_then_unsupported():
    ctx = make_ctx(func_name="main", symbols={"x": PyFloat()})
    with pytest.raises(TypeError, match="unsupported call"):
        Syntax.expr(parse_expr("missing(x)"), ctx)


def test_keyword_args_rejected(tmp_path):
    helper = _threaded("helper")
    hpp = tmp_path / "helper.hpp"
    hpp.write_text("", encoding="utf-8")
    _register_free(helper, hpp=hpp)
    ctx = make_ctx(
        func_name="main",
        symbols={"x": PyFloat()},
        globals_extra={"helper": helper},
        this_file=tmp_path / "main.hpp",
    )
    with pytest.raises(TypeError, match="keyword args"):
        Syntax.expr(parse_expr("helper(x=x)"), ctx)


def test_threaded_but_unregistered_raises():
    helper = _threaded("helper")
    ctx = make_ctx(
        func_name="main",
        symbols={"x": PyFloat()},
        globals_extra={"helper": helper},
    )
    with pytest.raises(TypeError, match="not registered"):
        Syntax.expr(parse_expr("helper(x)"), ctx)


def test_method_thread_unit_is_left_to_attr_plugin():
    cls = type("Particle", (), {"__threadable": True})
    REGISTRY.threadables["Particle"] = cls
    owner = ThreadableUnit(
        handle=Handle(name="Particle", path="dummy.py", target=cls),
        fields={},
        hpp_path=Path("__Threadable__") / "Particle.hpp",
        cpp_path=Path("__Threadable__") / "Particle.cpp",
    )
    REGISTRY.threadable_units["Particle"] = owner

    def step(self, dt: float) -> None:
        return None

    step.__threaded = True
    unit = ThreadUnit(
        handle=Handle(name=step.__qualname__, path="dummy.py", target=step),
        owner=owner,
        params=[],
        return_type=None,
        hpp_path=None,
        cpp_path=None,
    )
    REGISTRY.thread_units[step.__qualname__] = unit

    ctx = make_ctx(
        func_name="main",
        symbols={"x": PyFloat()},
        globals_extra={"step": step},
    )
    with pytest.raises(TypeError, match="unsupported call"):
        Syntax.expr(parse_expr("step(x)"), ctx)
