"""Unit tests for expression Typeof (Name / Threadable fields / list subscript)."""

from __future__ import annotations

from cthreads.compiler.translation.Typeof import Typeof
from cthreads.types import (
    PyCThreadsInternalType,
    PyInt,
    PyList,
    PyThreadable,
    hint_to_pytype,
)
from helpers import make_ctx, parse_expr, registered_threadable


_ArrayF32 = type("ArrayF32", (), {"__cthreads_internal__": True})
_Body = type(
    "TypeofBody",
    (),
    {"__threadable": True, "__annotations__": {"a": _ArrayF32, "xs": list[int]}},
)


def test_typeof_name():
    ctx = make_ctx(symbols={"n": PyInt()})
    assert isinstance(Typeof.of(parse_expr("n"), ctx), PyInt)
    assert Typeof.of(parse_expr("missing"), ctx) is None


def test_typeof_threadable_field():
    with registered_threadable(_Body):
        ctx = make_ctx(
            owner_name="TypeofBody",
            symbols={
                "self": PyThreadable("TypeofBody"),
                "p": PyThreadable("TypeofBody"),
            },
        )
        a = Typeof.of(parse_expr("self.a"), ctx)
        assert isinstance(a, PyCThreadsInternalType)
        assert a.cpp_name == "cthreads::linalg::Array<float>"
        xs = Typeof.of(parse_expr("p.xs"), ctx)
        assert isinstance(xs, PyList)
        assert isinstance(xs.inner_type, PyInt)
        assert Typeof.of(parse_expr("self.nope"), ctx) is None


def test_typeof_list_subscript():
    ctx = make_ctx(symbols={"xs": PyList(PyInt()), "i": PyInt()})
    inner = Typeof.of(parse_expr("xs[i]"), ctx)
    assert isinstance(inner, PyInt)


def test_typeof_nested_list_then_field():
    with registered_threadable(_Body):
        ctx = make_ctx(
            symbols={
                "bodies": PyList(PyThreadable("TypeofBody")),
                "i": PyInt(),
            }
        )
        ty = Typeof.of(parse_expr("bodies[i].a"), ctx)
        assert isinstance(ty, PyCThreadsInternalType)
        assert ty.name == "ArrayF32"


def test_hint_to_pytype_matches_typeof_field():
    with registered_threadable(_Body):
        py = hint_to_pytype(_ArrayF32)
        ctx = make_ctx(
            owner_name="TypeofBody",
            symbols={"self": PyThreadable("TypeofBody")},
        )
        assert Typeof.of(parse_expr("self.a"), ctx).cpp_name == py.cpp_name
