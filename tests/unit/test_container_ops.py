"""Unit tests for list/dict method lowering (containerOps + call + exprStmt)."""

from __future__ import annotations

import pytest

from cthreads.pyTypes import PyDict, PyInt, PyList, PyString, PyThreadable
from cthreads.Thread.compile.AstTranslators import call, exprStmt
from cthreads.Thread.compile.pythonContainerLibTranslators import (
    DICT_METHODS,
    LIST_METHODS,
    resolve_container_op,
)
from cthreads.Thread.compile.pythonContainerLibTranslators.is_containerOp import (
    is_container_op,
)
from helpers import make_ctx, parse_expr, parse_stmt, registered_threadable


def _list_ctx(**extra):
    symbols = {
        "xs": PyList(PyInt()),
        "ys": PyList(PyInt()),
        "v": PyInt(),
        "i": PyInt(),
        **extra,
    }
    return make_ctx(symbols=symbols)


def _dict_ctx(**extra):
    symbols = {
        "d": PyDict(PyString(), PyInt()),
        "k": PyString(),
        "default": PyInt(),
        **extra,
    }
    return make_ctx(symbols=symbols)


def test_list_method_tables_cover_core_ops():
    assert set(LIST_METHODS) >= {"append", "clear", "pop", "insert", "extend"}
    assert set(DICT_METHODS) >= {"get", "clear", "pop"}


def test_resolve_list_append_and_clear():
    ctx = _list_ctx()
    assert resolve_container_op(parse_expr("xs.append(v)"), ctx) is LIST_METHODS["append"]
    assert resolve_container_op(parse_expr("xs.clear()"), ctx) is LIST_METHODS["clear"]
    assert is_container_op(parse_expr("xs.append(v)"), ctx)


def test_resolve_returns_none_for_non_container():
    ctx = make_ctx(symbols={"n": PyInt(), "xs": PyList(PyInt())})
    assert resolve_container_op(parse_expr("n + 1"), ctx) is None
    assert resolve_container_op(parse_expr("len(xs)"), ctx) is None
    assert resolve_container_op(parse_expr("unknown(v)"), ctx) is None
    # int has no container methods → None (not a list/dict receiver)
    assert resolve_container_op(parse_expr("n.append(v)"), ctx) is None


def test_resolve_rejects_unknown_list_method():
    ctx = _list_ctx()
    with pytest.raises(TypeError, match="unsupported list method"):
        resolve_container_op(parse_expr("xs.remove(v)"), ctx)


def test_resolve_list_arity_errors():
    ctx = _list_ctx()
    with pytest.raises(TypeError, match="list.append"):
        resolve_container_op(parse_expr("xs.append()"), ctx)
    with pytest.raises(TypeError, match="list.clear"):
        resolve_container_op(parse_expr("xs.clear(v)"), ctx)
    with pytest.raises(TypeError, match="list.insert"):
        resolve_container_op(parse_expr("xs.insert(i)"), ctx)


def test_resolve_rejects_keywords():
    ctx = _list_ctx()
    with pytest.raises(TypeError, match="keyword"):
        resolve_container_op(parse_expr("xs.append(v=1)"), ctx)


def test_call_list_append_clear_insert_extend():
    ctx = _list_ctx()
    assert call.translate(parse_expr("xs.append(v)"), ctx) == "(xs).push_back(v)"
    assert call.translate(parse_expr("xs.clear()"), ctx) == "(xs).clear()"
    assert (
        call.translate(parse_expr("xs.insert(i, v)"), ctx)
        == "(xs).insert((xs).begin() + (i), v)"
    )
    assert (
        call.translate(parse_expr("xs.extend(ys)"), ctx)
        == "(xs).insert((xs).end(), (ys).begin(), (ys).end())"
    )


def test_call_list_pop_no_arg():
    ctx = _list_ctx()
    got = call.translate(parse_expr("xs.pop()"), ctx)
    assert "auto v = (xs).back();" in got
    assert "(xs).pop_back();" in got
    assert "return v;" in got
    assert got.startswith("([&]()")
    assert got.endswith("())")


def test_call_list_pop_at_index():
    ctx = _list_ctx()
    got = call.translate(parse_expr("xs.pop(i)"), ctx)
    assert "auto& c = (xs);" in got
    assert "auto it = c.begin() + (i);" in got
    assert "auto v = *it;" in got
    assert "c.erase(it);" in got
    assert "return v;" in got


def test_resolve_dict_get_pop_require_default():
    ctx = _dict_ctx()
    with pytest.raises(TypeError, match="dict.get"):
        resolve_container_op(parse_expr("d.get(k)"), ctx)
    with pytest.raises(TypeError, match="dict.pop"):
        resolve_container_op(parse_expr("d.pop(k)"), ctx)


def test_call_dict_get_clear_pop():
    ctx = _dict_ctx()
    got_get = call.translate(parse_expr("d.get(k, default)"), ctx)
    assert "(d).find(k)" in got_get
    assert "it->second" in got_get
    assert "(default)" in got_get

    assert call.translate(parse_expr("d.clear()"), ctx) == "(d).clear()"

    got_pop = call.translate(parse_expr("d.pop(k, default)"), ctx)
    assert "(d).find(k)" in got_pop
    assert "(d).erase(it)" in got_pop
    assert "return v;" in got_pop


def test_resolve_rejects_unknown_dict_method():
    ctx = _dict_ctx()
    with pytest.raises(TypeError, match="unsupported dict method"):
        resolve_container_op(parse_expr("d.update(k)"), ctx)


def test_expr_stmt_list_append():
    ctx = _list_ctx()
    lines = exprStmt.translate(parse_stmt("xs.append(v)"), ctx)
    assert lines == ["    (xs).push_back(v);"]


def test_expr_stmt_still_ignores_docstring_and_rejects_binop():
    ctx = make_ctx()
    assert exprStmt.translate(parse_stmt('"docstring"'), ctx) == []
    lines = exprStmt.translate(parse_stmt("1 + 2"), ctx)
    assert "unsupported statement: Expr" in lines[0]


def test_attribute_receiver_uses_threadable_field():
    """obj.items.append waits on field typing via REGISTRY, not a handwritten map."""
    Box = type(
        "Box",
        (),
        {"__threadable": True, "__annotations__": {"items": list[int]}},
    )
    with registered_threadable(Box):
        ctx = make_ctx(
            owner_name="Box",
            symbols={
                "self": PyThreadable("Box", "x"),
                "v": PyInt(),
            },
        )
        assert resolve_container_op(parse_expr("self.items.append(v)"), ctx) is LIST_METHODS["append"]
        assert call.translate(parse_expr("self.items.append(v)"), ctx) == (
            "(this->items).push_back(v)"
        )


def test_untyped_attribute_receiver_is_none():
    ctx = make_ctx(symbols={"v": PyInt()})
    assert resolve_container_op(parse_expr("obj.items.append(v)"), ctx) is None
