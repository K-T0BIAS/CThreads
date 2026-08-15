"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Detect cthreads.sync method calls for lowering.

Like container ops: `recv.method(...)` on a typed receiver (`PyCthreadsInternal`
Lock / Event / RWLock). The receiver is typed by `typeof` (a name in
`ctx.symbols` or a Threadable field).
"""

import ast
from typing import Optional

from ....pyTypes import PyCthreadsInternal
from ..AstTranslators.context import TranslateContext
from ..AstTranslators.typeof import expr_src, typeof
from .syncOps import SYNC_METHODS, SyncOp


def resolve_sync_op(node: ast.AST, ctx: TranslateContext) -> Optional[SyncOp]:
    """
    Resolve `lock.acquire()` / `event.wait_for(t)` to a `SyncOp`.

    #### Args
    - node: ast.AST - usually an `ast.Call`
    - ctx: TranslateContext - needs `symbols` for the receiver type

    #### Returns
    - Optional[SyncOp] - the op if this is a whitelisted sync method call,
      otherwise None

    #### Raises
    - TypeError - known sync type with bad arity, keywords, or unknown method
    """
    # lock.acquire() -> Call(Attribute(Name('lock'), 'acquire'), [])
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Attribute):
        return None

    recv = node.func.value
    ty = typeof(recv, ctx)
    if not isinstance(ty, PyCthreadsInternal):
        return None

    table = SYNC_METHODS.get(ty.name)
    if table is None:
        return None
    if node.keywords:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            "sync method keyword args are not supported"
        )

    name = node.func.attr
    op = table.get(name)
    if op is None:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported {ty.name} method {name!r} on {expr_src(recv)!r}"
        )

    n = len(node.args)
    if n < op.min_arity or n > op.max_arity:
        if op.min_arity == op.max_arity:
            expect = str(op.min_arity)
        else:
            expect = f"{op.min_arity}..{op.max_arity}"
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"{ty.name}.{name}() expects {expect} arg(s), got {n}"
        )

    return op


def is_sync_op(node: ast.AST, ctx: TranslateContext) -> bool:
    """True if `resolve_sync_op` would return a `SyncOp`."""
    return resolve_sync_op(node, ctx) is not None
