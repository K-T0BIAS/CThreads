"""
Fallback field / Threadable-method lowering (v1 attribute.py tail).

Must register **last** so math consts, linalg, sync, and container methods win first.
V1 printed unknown attributes as `base.attr`. Calls on a @Threadable receiver
are the same idea: `(obj).method(args)` / `this->method(args)`.
"""

import ast

from .....types import PyThreadable
from ...Typeof import Typeof
from ...context import TranslationContext
from ..base import AttrPlugin, CallPlugin, TranslateExpr


class FieldAttrPlugin(AttrPlugin):
    """`self.x` -> `this->x`; otherwise `(base).attr`."""

    def try_lower(
        self,
        node: ast.Attribute,
        ctx: TranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and ctx.owner is not None
        ):
            return f"this->{node.attr}"
        base = translate_expr(node.value, ctx)
        return f"{base}.{node.attr}"


class ThreadableMethodPlugin(CallPlugin):
    """`obj.method(args)` on a @Threadable -> `(obj).method(args)`."""

    def try_lower(
        self,
        node: ast.Call,
        ctx: TranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        if not isinstance(node.func, ast.Attribute):
            return None
        recv_ty = Typeof.of(node.func.value, ctx)
        if not isinstance(recv_ty, PyThreadable):
            return None
        if node.keywords:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"{recv_ty.name}.{node.func.attr} keyword args are not supported"
            )
        args = [translate_expr(a, ctx) for a in node.args]
        arg_csv = ", ".join(args)
        if (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and ctx.owner is not None
        ):
            if arg_csv:
                return f"this->{node.func.attr}({arg_csv})"
            return f"this->{node.func.attr}()"
        recv = translate_expr(node.func.value, ctx)
        if arg_csv:
            return f"({recv}).{node.func.attr}({arg_csv})"
        return f"({recv}).{node.func.attr}()"
