"""
Plugin bases for Call / Attribute lowering.

Contract (stdlib math and native sync/linalg share this):
  try_lower(node, ctx, translate_expr) -> C++ expr string | None
  Includes go on ctx.body_includes via add_include.

MethodTablePlugin: receiver Typeof + per-type method tables (list, Lock, Array, …).
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar, NamedTuple

from ....types import PyType
from ..Typeof import Typeof
from ..context import TranslationContext
from ..include import add_include

# Already-translated subexpressions; typically Syntax.expr
TranslateExpr = Callable[[ast.expr, TranslationContext], str]

# (receiver_cpp, arg_cpps) -> C++ expr
MethodEmit = Callable[[str, list[str]], str]


class MethodOp(NamedTuple):
    """One method row for MethodTablePlugin tables."""

    emit: MethodEmit
    min_arity: int
    max_arity: int
    # Full `#include ...\n` lines (angle or quote). Empty if headers come from the type.
    includes: tuple[str, ...] = ()


class CallPlugin(ABC):
    """Handles `ast.Call` (free calls, ctors, or methods - subclass decides)."""

    @abstractmethod
    def try_lower(
        self,
        node: ast.Call,
        ctx: TranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        """
        Return a C++ expression if this plugin handles `node`, else None.

        May append to `ctx.body_includes`. Use `translate_expr` for nested
        expressions (never call Syntax from here at import time).
        """


class AttrPlugin(ABC):
    """Handles `ast.Attribute` (fields, props, module constants - not calls)."""

    @abstractmethod
    def try_lower(
        self,
        node: ast.Attribute,
        ctx: TranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        """
        Return a C++ expression if this plugin handles `node`, else None.

        May append to `ctx.body_includes`.
        """


class MethodTablePlugin(CallPlugin):
    """
    `recv.method(args)` where `Typeof.of(recv)` selects a method table.

    Subclasses set `tables` and implement `type_key`.
    Unknown method on a known type_key raises TypeError (same as v1 sync/list).
    """

    tables: ClassVar[dict[str, dict[str, MethodOp]]] = {}

    @abstractmethod
    def type_key(self, py_type: PyType) -> str | None:
        """Map a receiver PyType to a key in `tables`, or None if not ours."""

    def try_lower(
        self,
        node: ast.Call,
        ctx: TranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        if not isinstance(node.func, ast.Attribute):
            return None

        recv_ty = Typeof.of(node.func.value, ctx)
        if recv_ty is None:
            return None

        key = self.type_key(recv_ty)
        if key is None:
            return None

        methods = self.tables.get(key)
        if methods is None:
            return None

        op = methods.get(node.func.attr)
        if op is None:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"unknown method {node.func.attr!r} on {key}"
            )

        if node.keywords:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"{key}.{node.func.attr} keyword args are not supported"
            )

        n_args = len(node.args)
        if n_args < op.min_arity or n_args > op.max_arity:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"{key}.{node.func.attr} expects {op.min_arity}"
                f"{'' if op.min_arity == op.max_arity else f'..{op.max_arity}'} "
                f"args, got {n_args}"
            )

        for line in op.includes:
            add_include(ctx.body_includes, ctx.seen_body, line)

        recv = translate_expr(node.func.value, ctx)
        args = [translate_expr(a, ctx) for a in node.args]
        return op.emit(recv, args)


def method_op(
    name: str,
    min_arity: int = 0,
    max_arity: int | None = None,
    includes: tuple[str, ...] = (),
) -> MethodOp:
    """1:1 Python method -> `(recv).name(args...)`."""
    if max_arity is None:
        max_arity = min_arity

    def emit(recv: str, args: list[str]) -> str:
        if not args:
            return f"({recv}).{name}()"
        return f"({recv}).{name}({', '.join(args)})"

    return MethodOp(
        emit=emit,
        min_arity=min_arity,
        max_arity=max_arity,
        includes=includes,
    )
