"""`ArrayF32(shape)` / `linalg.Shape(n)` / `Slice(...)` constructors."""

from __future__ import annotations

import ast
import sys
from typing import Any

from ...context import TranslationContext
from ...include import add_include
from ..base import CallPlugin, TranslateExpr
from .ops import LINALG_CTORS


def _globals(ctx: TranslationContext) -> dict[str, Any]:
    g = getattr(ctx.fn, "__globals__", None)
    return g if isinstance(g, dict) else {}


def _owner_is_cthreads_linalg(obj: Any, parent_mod: Any) -> bool:
    if parent_mod is not None and getattr(parent_mod, "__cthreads_internal__", False):
        return True
    if getattr(obj, "__cthreads_internal__", False):
        return True
    mod_name = getattr(obj, "__module__", None)
    if isinstance(mod_name, str):
        mod = sys.modules.get(mod_name)
        if mod is not None and getattr(mod, "__cthreads_internal__", False):
            return True
    return False


def _resolve_ctor_obj(node: ast.Call, ctx: TranslationContext) -> tuple[Any, Any]:
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        mod = _globals(ctx).get(func.value.id)
        if mod is None:
            return None, None
        return getattr(mod, func.attr, None), mod
    if isinstance(func, ast.Name):
        return _globals(ctx).get(func.id), None
    return None, None


class LinalgCtorPlugin(CallPlugin):
    def try_lower(
        self,
        node: ast.Call,
        ctx: TranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        obj, parent = _resolve_ctor_obj(node, ctx)
        if obj is None:
            return None
        name = getattr(obj, "__name__", None)
        if not isinstance(name, str) or name not in LINALG_CTORS:
            return None
        if not _owner_is_cthreads_linalg(obj, parent):
            return None

        if node.keywords:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                "linalg constructor keyword args are not supported"
            )

        ctor = LINALG_CTORS[name]
        n = len(node.args)
        if n < ctor.min_arity or n > ctor.max_arity:
            if ctor.min_arity == ctor.max_arity:
                expect = str(ctor.min_arity)
            else:
                expect = f"{ctor.min_arity}..{ctor.max_arity}"
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"{name}() expects {expect} arg(s), got {n}"
            )

        add_include(
            ctx.body_includes,
            ctx.seen_body,
            f'#include "{ctor.cpp_include}"\n',
        )
        for inc in ctor.extra_includes:
            add_include(ctx.body_includes, ctx.seen_body, f"#include <{inc}>\n")

        args = [translate_expr(a, ctx) for a in node.args]
        if not args:
            return f"{ctor.cpp_type}()"
        return f"{ctor.cpp_type}({', '.join(args)})"
