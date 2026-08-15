"""
Python container builtins / methods → STL.

- `len(xs)` -> `(xs).size()` (TBuffer -> `(xs).capacity()`)
- list: append/clear/pop/insert/extend
- dict: get/clear/pop (default required)
"""

import ast
from .....types import PyDict, PyList, PyType, is_tbuffer_pytype
from ...Typeof import Typeof
from ...context import TranslationContext
from ...syntax.Op import Op
from ..base import CallPlugin, MethodOp, MethodTablePlugin, TranslateExpr, method_op


class LenPlugin(CallPlugin):
    """`len(xs)` -> `(xs).size()`; TBuffer -> `(xs).capacity()`."""

    def try_lower(
        self,
        node: ast.Call,
        ctx: TranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        if not Op.is_builtin_call(node, "len"):
            return None
        if node.keywords:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                "len() keyword args are not supported"
            )
        if len(node.args) != 1:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"len() expects 1 arg, got {len(node.args)}"
            )

        arg_node = node.args[0]
        ty = Typeof.of(arg_node, ctx)
        arg = translate_expr(arg_node, ctx)
        if ty is not None and is_tbuffer_pytype(ty):
            return f"({arg}).capacity()"
        return f"({arg}).size()"


def _list_pop(recv: str, args: list[str]) -> str:
    if not args:
        return (
            f"([&]() {{"
            f"   auto v = ({recv}).back();"
            f"   ({recv}).pop_back();"
            f"   return v;"
            f"}}())"
        )
    return (
        f"([&]() {{"
        f"   auto& c = ({recv});"
        f"   auto it = c.begin() + ({args[0]});"
        f"   auto v = *it;"
        f"   c.erase(it);"
        f"   return v;"
        f"}}())"
    )


class ContainerMethodPlugin(MethodTablePlugin):
    """`xs.append(v)` / `d.get(k, default)` on typed list/dict receivers."""

    tables = {
        "list": {
            "append": MethodOp(
                emit=lambda recv, args: f"({recv}).push_back({args[0]})",
                min_arity=1,
                max_arity=1,
            ),
            "clear": method_op("clear", 0),
            "pop": MethodOp(emit=_list_pop, min_arity=0, max_arity=1),
            "insert": MethodOp(
                emit=lambda recv, args: (
                    f"({recv}).insert(({recv}).begin() + ({args[0]}), {args[1]})"
                ),
                min_arity=2,
                max_arity=2,
            ),
            "extend": MethodOp(
                emit=lambda recv, args: (
                    f"({recv}).insert(({recv}).end(), "
                    f"({args[0]}).begin(), ({args[0]}).end())"
                ),
                min_arity=1,
                max_arity=1,
            ),
        },
        "dict": {
            "get": MethodOp(
                emit=lambda recv, args: (
                    f"([&]() {{ "
                    f"    auto it = ({recv}).find({args[0]}); "
                    f"    return it != ({recv}).end() ? it->second : ({args[1]}); "
                    f"}}())"
                ),
                min_arity=2,
                max_arity=2,
            ),
            "clear": method_op("clear", 0),
            "pop": MethodOp(
                emit=lambda recv, args: (
                    f"([&]() {{ "
                    f"    auto it = ({recv}).find({args[0]}); "
                    f"    if (it == ({recv}).end()) return ({args[1]}); "
                    f"    auto v = it->second; "
                    f"    ({recv}).erase(it); "
                    f"    return v; "
                    f"}}())"
                ),
                min_arity=2,
                max_arity=2,
            ),
        },
    }

    def type_key(self, py_type: PyType) -> str | None:
        if isinstance(py_type, PyList):
            return "list"
        if isinstance(py_type, PyDict):
            return "dict"
        return None
