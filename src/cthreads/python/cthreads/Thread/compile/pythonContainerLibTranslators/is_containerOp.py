"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Detect list/dict method calls for lowering.

Unlike math (resolved via `ctx.fn.__globals__`), container ops are
`recv.method(...)` on a typed local. Receiver must be a bare `Name`
present in `ctx.symbols` as `PyList` / `PyDict` (same limit as
`for x in xs`).
"""

from __future__ import annotations

import ast
from typing import Optional

from ....pyTypes import PyDict, PyList
from ..AstTranslators.context import TranslateContext
from .containerOps import DICT_METHODS, LIST_METHODS, ContainerOp


def resolve_container_op(node: ast.AST, ctx: TranslateContext) -> Optional[ContainerOp]:
    """
    Resolve `xs.append(...)` / `d.get(k, default)` to a `ContainerOp` like `cs.push_back(v)`

    #### Args
    - node: ast.AST - usually an `ast.Call`
    - ctx: TranslateContext - needs `symbols` for the receiver type

    #### Returns
    - Optional[ContainerOp] - the op if this is a whitelisted container
      method call, otherwise None

    #### Raises
    - TypeError - known list/dict method with bad arity, or unknown method
      on a typed list/dict receiver
    """
    # xs.append(v)  ->  Call(func=Attribute(value=Name('xs'), attr='append'), ...)
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Attribute):
        return None
    if node.keywords:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            "container method keyword args are not supported"
        )

    recv = node.func.value
    # Start with Name-only receivers (xs.append); self.items.append later.
    if not isinstance(recv, ast.Name):
        return None

    ty = ctx.symbols.get(recv.id)
    if isinstance(ty, PyList):
        table = LIST_METHODS
        kind = "list"
    elif isinstance(ty, PyDict):
        table = DICT_METHODS
        kind = "dict"
    else:
        return None

    name = node.func.attr
    op = table.get(name)
    if op is None:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported {kind} method {name!r} on {recv.id!r}"
        )

    n = len(node.args)
    if n < op.min_arity or n > op.max_arity:
        if op.min_arity == op.max_arity:
            expect = str(op.min_arity)
        else:
            expect = f"{op.min_arity}..{op.max_arity}"
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"{kind}.{name}() expects {expect} arg(s), got {n}"
        )

    return op


def is_container_op(node: ast.AST, ctx: TranslateContext) -> bool:
    """True if ``resolve_container_op`` would return a ``ContainerOp``."""
    return resolve_container_op(node, ctx) is not None
