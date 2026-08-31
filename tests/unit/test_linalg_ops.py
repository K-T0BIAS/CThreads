"""Unit tests for cthreads.linalg ctor / method / property lowering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cthreads.compiler.translation.plugins.linalg.ops import (
    ARRAY_HPP,
    ARRAY_METHODS,
    ARRAY_NUMERIC_METHODS,
    ARRAY_PROPS,
    LINALG_CTORS,
    LINALG_METHODS,
    SHAPE_HPP,
    SHAPE_METHODS,
    SLICE_HPP,
)
from cthreads.compiler.translation.syntax import Syntax
from cthreads.types import PyFloat, PyInt, PyThreadable, hint_to_pytype
from helpers import make_ctx, parse_expr, registered_threadable


def _array_ty(name: str = "ArrayF32"):
    return hint_to_pytype(type(name, (), {"__cthreads_internal__": True}))


def _shape_ty():
    return hint_to_pytype(type("Shape", (), {"__cthreads_internal__": True}))


def _marked(name: str):
    return type(name, (), {"__cthreads_internal__": True, "__name__": name})


def _linalg_mod(*names: str):
    classes = {n: _marked(n) for n in names}
    return SimpleNamespace(
        __name__="cthreads.linalg",
        __cthreads_internal__=True,
        **classes,
    )


def _linalg_ctx(**extra):
    la = _linalg_mod(
        "ArrayF32", "ArrayF64", "ArrayI32", "ArrayBool", "Shape", "Slice"
    )
    return make_ctx(
        symbols={
            "a": _array_ty("ArrayF32"),
            "b": _array_ty("ArrayF32"),
            "mask": _array_ty("ArrayBool"),
            "sh": _shape_ty(),
            "n": PyInt(),
            "x": PyFloat(),
            **extra,
        },
        globals_extra={
            "linalg": la,
            "ArrayF32": la.ArrayF32,
            "ArrayF64": la.ArrayF64,
            "ArrayI32": la.ArrayI32,
            "ArrayBool": la.ArrayBool,
            "Shape": la.Shape,
            "Slice": la.Slice,
        },
    )


def test_linalg_tables_cover_core_ops():
    assert set(ARRAY_METHODS) >= {
        "view",
        "reshape",
        "flatten",
        "transpose",
        "contiguous",
        "masked_select",
        "masked_fill",
        "masked_scatter",
        "count",
        "any",
        "all",
    }
    assert set(ARRAY_NUMERIC_METHODS) >= {"matmul", "dot", "cross", "_add", "_neg"}
    assert set(SHAPE_METHODS) >= {"ndim", "numel", "strides"}
    assert set(ARRAY_PROPS) >= {"shape", "strides", "ndim", "numel", "offset"}
    assert set(LINALG_CTORS) == {
        "ArrayF32",
        "ArrayF64",
        "ArrayI32",
        "ArrayBool",
        "Shape",
        "Slice",
    }
    assert LINALG_CTORS["ArrayF32"].cpp_include == ARRAY_HPP
    assert LINALG_CTORS["Shape"].cpp_include == SHAPE_HPP
    assert LINALG_CTORS["Slice"].cpp_include == SLICE_HPP
    assert "matmul" in LINALG_METHODS["ArrayF32"]
    assert "matmul" not in LINALG_METHODS["ArrayBool"]


def test_call_array_methods():
    ctx = _linalg_ctx()
    assert Syntax.expr(parse_expr("a.transpose()"), ctx) == "(a).transpose()"
    assert Syntax.expr(parse_expr("a.matmul(b)"), ctx) == "(a).matmul(b)"
    assert Syntax.expr(parse_expr("a.contiguous()"), ctx) == "(a).contiguous()"
    assert Syntax.expr(parse_expr("a.masked_fill(mask, x)"), ctx) == (
        "(a).masked_fill(mask, x)"
    )
    assert Syntax.expr(parse_expr("a.count()"), ctx) == "(a).count_nonzero()"
    assert Syntax.expr(parse_expr("a.any()"), ctx) == "((a).count_nonzero() != 0)"
    assert Syntax.expr(parse_expr("a.all()"), ctx) == (
        "((a).count_nonzero() == (a).shape().numel())"
    )
    assert any(ARRAY_HPP in line for line in ctx.body_includes)


def test_call_rejects_unknown_and_bool_numeric():
    ctx = _linalg_ctx()
    with pytest.raises(TypeError, match="unknown method"):
        Syntax.expr(parse_expr("a.foo()"), ctx)
    with pytest.raises(TypeError, match="unknown method"):
        Syntax.expr(parse_expr("mask.matmul(a)"), ctx)


def test_call_method_arity_and_keywords():
    ctx = _linalg_ctx()
    with pytest.raises(TypeError, match="matmul"):
        Syntax.expr(parse_expr("a.matmul()"), ctx)
    with pytest.raises(TypeError, match="transpose"):
        Syntax.expr(parse_expr("a.transpose(b)"), ctx)
    with pytest.raises(TypeError, match="keyword"):
        Syntax.expr(parse_expr("a.matmul(other=b)"), ctx)


def test_non_linalg_receiver_is_unsupported():
    ctx = make_ctx(symbols={"n": PyInt(), "a": _array_ty()})
    with pytest.raises(TypeError, match="unsupported"):
        Syntax.expr(parse_expr("n.transpose()"), ctx)
    with pytest.raises(TypeError, match="unsupported"):
        Syntax.expr(parse_expr("unknown()"), ctx)


def test_call_shape_methods():
    ctx = _linalg_ctx()
    assert Syntax.expr(parse_expr("sh.ndim()"), ctx) == "(sh).ndim()"
    assert Syntax.expr(parse_expr("sh.numel()"), ctx) == "(sh).numel()"
    assert Syntax.expr(parse_expr("sh.strides()"), ctx) == "(sh).strides()"
    assert any(SHAPE_HPP in line for line in ctx.body_includes)


def test_attribute_array_props():
    ctx = _linalg_ctx()
    assert Syntax.expr(parse_expr("a.shape"), ctx) == "(a).shape()"
    assert Syntax.expr(parse_expr("a.ndim"), ctx) == "(a).ndim()"
    assert Syntax.expr(parse_expr("a.numel"), ctx) == "(a).shape().numel()"
    assert Syntax.expr(parse_expr("a.offset"), ctx) == "(a).offset()"
    assert Syntax.expr(parse_expr("a.strides"), ctx) == "(a).strides()"
    assert any(ARRAY_HPP in line for line in ctx.body_includes)


def test_attribute_receiver_uses_threadable_field():
    ArrayF32 = type("ArrayF32", (), {"__cthreads_internal__": True})
    Body = type(
        "Body",
        (),
        {"__threadable": True, "__annotations__": {"a": ArrayF32}},
    )
    with registered_threadable(Body):
        ctx = _linalg_ctx(self=PyThreadable("Body"))
        # owner unit is required for `self` -> this-> lowering
        from cthreads.frontend.Registry import REGISTRY

        ctx.owner = REGISTRY.threadable_units["Body"]
        assert Syntax.expr(parse_expr("self.a.matmul(b)"), ctx) == (
            "(this->a).matmul(b)"
        )
        assert Syntax.expr(parse_expr("self.a.shape"), ctx) == "(this->a).shape()"


def test_untyped_attribute_receiver_is_unsupported():
    ctx = _linalg_ctx()
    with pytest.raises(TypeError, match="unsupported"):
        Syntax.expr(parse_expr("obj.a.transpose()"), ctx)


def test_call_ctors_emit_cpp_types():
    ctx = _linalg_ctx()
    assert Syntax.expr(parse_expr("ArrayF32(sh)"), ctx) == (
        "cthreads::linalg::Array<float>(sh)"
    )
    assert Syntax.expr(parse_expr("linalg.ArrayBool(sh)"), ctx) == (
        "cthreads::linalg::Array<uint8_t>(sh)"
    )
    assert Syntax.expr(parse_expr("Shape(n)"), ctx) == "cthreads::linalg::Shape(n)"
    assert Syntax.expr(parse_expr("Shape([2, 3])"), ctx) == (
        "cthreads::linalg::Shape(std::vector<int>{2, 3})"
    )
    assert Syntax.expr(parse_expr("ArrayF32([2, 3])"), ctx) == (
        "cthreads::linalg::Array<float>(std::vector<int>{2, 3})"
    )
    assert Syntax.expr(parse_expr("a.reshape([2, 3])"), ctx) == (
        "(a).reshape(std::vector<int>{2, 3})"
    )
    assert Syntax.expr(parse_expr("Slice()"), ctx) == "cthreads::linalg::Slice()"
    assert Syntax.expr(parse_expr("Slice(n, n)"), ctx) == (
        "cthreads::linalg::Slice(n, n)"
    )
    assert any(ARRAY_HPP in line for line in ctx.body_includes)
    assert any(SHAPE_HPP in line for line in ctx.body_includes)
    assert any("cstdint" in line for line in ctx.body_includes)


def test_ctor_rejects_arity_keywords_and_unmarked():
    ctx = _linalg_ctx()
    with pytest.raises(TypeError, match="ArrayF32\\(\\) expects"):
        Syntax.expr(parse_expr("ArrayF32()"), ctx)
    with pytest.raises(TypeError, match="ArrayF32\\(\\) expects"):
        Syntax.expr(parse_expr("ArrayF32(sh, n)"), ctx)
    with pytest.raises(TypeError, match="keyword"):
        Syntax.expr(parse_expr("ArrayF32(shape=sh)"), ctx)

    plain = type("ArrayF32", (), {"__name__": "ArrayF32"})
    other = SimpleNamespace(__name__="other", ArrayF32=plain)
    unmarked = make_ctx(
        symbols={"sh": _shape_ty()},
        globals_extra={"ArrayF32": plain, "other": other},
    )
    with pytest.raises(TypeError, match="unsupported"):
        Syntax.expr(parse_expr("ArrayF32(sh)"), unmarked)
    with pytest.raises(TypeError, match="unsupported"):
        Syntax.expr(parse_expr("other.ArrayF32(sh)"), unmarked)
