"""Unit tests for expression typeof (Name / Threadable fields / list subscript)."""

from __future__ import annotations

from cthreads.pyTypes import (
    PyCthreadsInternal,
    PyInt,
    PyList,
    PyThreadable,
    hint_to_pytype,
)
from cthreads.Thread.compile.AstTranslators.typeof import typeof
from helpers import make_ctx, parse_expr, registered_threadable


_ArrayF32 = type("ArrayF32", (), {"__cthreads_internal__": True})
_Body = type(
    "TypeofBody",
    (),
    {"__threadable": True, "__annotations__": {"a": _ArrayF32, "xs": list[int]}},
)


def test_typeof_name():
    ctx = make_ctx(symbols={"n": PyInt()})
    assert isinstance(typeof(parse_expr("n"), ctx), PyInt)
    assert typeof(parse_expr("missing"), ctx) is None


def test_typeof_threadable_field():
    with registered_threadable(_Body):
        ctx = make_ctx(
            owner_name="TypeofBody",
            symbols={
                "self": PyThreadable("TypeofBody", "x"),
                "p": PyThreadable("TypeofBody", "x"),
            },
        )
        a = typeof(parse_expr("self.a"), ctx)
        assert isinstance(a, PyCthreadsInternal)
        assert a.cpp_name == "cthreads::linalg::Array<float>"
        xs = typeof(parse_expr("p.xs"), ctx)
        assert isinstance(xs, PyList)
        assert isinstance(xs.inner_type, PyInt)
        assert typeof(parse_expr("self.nope"), ctx) is None


def test_typeof_list_subscript():
    ctx = make_ctx(symbols={"xs": PyList(PyInt()), "i": PyInt()})
    inner = typeof(parse_expr("xs[i]"), ctx)
    assert isinstance(inner, PyInt)


def test_typeof_nested_list_then_field():
    with registered_threadable(_Body):
        ctx = make_ctx(
            symbols={
                "bodies": PyList(PyThreadable("TypeofBody", "x")),
                "i": PyInt(),
            }
        )
        ty = typeof(parse_expr("bodies[i].a"), ctx)
        assert isinstance(ty, PyCthreadsInternal)
        assert ty.name == "ArrayF32"


def test_hint_to_pytype_matches_typeof_field():
    with registered_threadable(_Body):
        py = hint_to_pytype(_ArrayF32)
        ctx = make_ctx(
            owner_name="TypeofBody",
            symbols={"self": PyThreadable("TypeofBody", "x")},
        )
        assert typeof(parse_expr("self.a"), ctx).cpp_name == py.cpp_name
