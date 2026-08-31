"""`Shared[list|dict]` method lowering (reuses container STL tables)."""

from __future__ import annotations

import ast

from .....types import PyDict, PyList, PyShared, PyType, is_shared_pytype
from ...context import TranslationContext
from ...include import add_include
from ..base import CallPlugin, TranslateExpr
from ..stdlib.Containers import ContainerMethodPlugin


def shared_container_key(py_type: PyType) -> str | None:
    if not isinstance(py_type, PyShared):
        return None
    inner = py_type.inner_type
    if isinstance(inner, PyList):
        return "list"
    if isinstance(inner, PyDict):
        return "dict"
    return None


class SharedMethodPlugin(CallPlugin):
    tables = ContainerMethodPlugin.tables

    def try_lower(
        self,
        node: ast.Call,
        ctx: TranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        if not isinstance(node.func, ast.Attribute):
            return None
        if not isinstance(node.func.value, ast.Name):
            return None

        raw = ctx.symbols.get(node.func.value.id)
        if not is_shared_pytype(raw):
            return None

        key = shared_container_key(raw)
        if key is None:
            return None

        methods = self.tables.get(key)
        if methods is None:
            return None

        op = methods.get(node.func.attr)
        if op is None:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"unknown method {node.func.attr!r} on Shared[{key}]"
            )

        if node.keywords:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"Shared[{key}].{node.func.attr} keyword args are not supported"
            )

        n_args = len(node.args)
        if n_args < op.min_arity or n_args > op.max_arity:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"Shared[{key}].{node.func.attr} expects {op.min_arity}"
                f"{'' if op.min_arity == op.max_arity else f'..{op.max_arity}'} "
                f"args, got {n_args}"
            )

        for line in op.includes:
            add_include(ctx.body_includes, ctx.seen_body, line)

        recv = translate_expr(node.func.value, ctx)
        args = [translate_expr(a, ctx) for a in node.args]
        return op.emit(recv, args)
