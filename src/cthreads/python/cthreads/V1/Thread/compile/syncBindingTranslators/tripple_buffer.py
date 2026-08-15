"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

``TBuffer*`` / ``TBuffer[Threadable]`` method lowering for @Thread codegen.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import NamedTuple, Optional

from ....pyTypes import PyCthreadsInternal, PyTBuffer, is_tbuffer_pytype
from ..AstTranslators.context import TranslateContext
from ..AstTranslators.typeof import expr_src, typeof

Emit = Callable[[str, list[str]], str]


class TBufferOp(NamedTuple):
    emit: Emit
    min_arity: int
    max_arity: int


def _method(name: str, min_arity: int = 0, max_arity: int | None = None) -> TBufferOp:
    if max_arity is None:
        max_arity = min_arity

    def emit(recv: str, args: list[str]) -> str:
        if not args:
            return f"({recv}).{name}()"
        return f"({recv}).{name}({', '.join(args)})"

    return TBufferOp(emit=emit, min_arity=min_arity, max_arity=max_arity)


def _read_at_emit(recv: str, args: list[str]) -> str:
    if len(args) != 1:
        raise ValueError("read_at expects one index")
    return f"(({recv}).get_read_slot()[{args[0]}])"


TBUFFER_METHODS: dict[str, TBufferOp] = {
    "publish": _method("publish", 0),
    "generation": _method("generation", 0),
    "capacity": _method("capacity", 0),
    "read_at": TBufferOp(emit=_read_at_emit, min_arity=1, max_arity=1),
}


def resolve_tbuffer_op(node: ast.AST, ctx: TranslateContext) -> Optional[TBufferOp]:
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Attribute):
        return None

    recv = node.func.value
    ty = typeof(recv, ctx)
    if ty is None or not is_tbuffer_pytype(ty):
        return None
    if node.keywords:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            "triple-buffer method keyword args are not supported"
        )

    name = node.func.attr
    op = TBUFFER_METHODS.get(name)
    if op is None:
        label = ty.name if isinstance(ty, (PyTBuffer, PyCthreadsInternal)) else "TBuffer"
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported {label} method {name!r} on {expr_src(recv)!r}"
        )

    n = len(node.args)
    if n < op.min_arity or n > op.max_arity:
        if op.min_arity == op.max_arity:
            expect = str(op.min_arity)
        else:
            expect = f"{op.min_arity}..{op.max_arity}"
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"TBuffer.{name}() expects {expect} arg(s), got {n}"
        )
    return op


def is_tbuffer_op(node: ast.AST, ctx: TranslateContext) -> bool:
    return resolve_tbuffer_op(node, ctx) is not None
