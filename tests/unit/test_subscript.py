"""Subscript expression lowering."""

import ast

from cthreads.compiler.translation.syntax import Syntax
from cthreads.types import PyFloat, PyInt, PyList
from helpers import make_ctx


def test_subscript_list_index():
    ctx = make_ctx(symbols={"xs": PyList(PyFloat()), "i": PyInt()})
    node = ast.Subscript(
        value=ast.Name(id="xs", ctx=ast.Load()),
        slice=ast.Name(id="i", ctx=ast.Load()),
        ctx=ast.Load(),
    )
    assert Syntax.expr(node, ctx) == "(xs[i])"
