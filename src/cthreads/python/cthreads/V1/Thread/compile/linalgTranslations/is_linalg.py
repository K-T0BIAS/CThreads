"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Detect ``cthreads.linalg`` ctors / methods / properties for lowering.

Methods: ``recv.method(...)`` on a typed local (same as sync / containers).
Ctors: ``ArrayF32(shape)`` / ``linalg.ArrayF32(shape)`` via ``ctx.fn.__globals__``
(same as ``cthreads.math``). The *module* and pybind *classes* are marked
``__cthreads_internal__``.
"""

from __future__ import annotations

import ast
import sys
from typing import Any, Optional

from ....pyTypes import PyCthreadsInternal
from ..AstTranslators.context import TranslateContext
from ..AstTranslators.typeof import expr_src, typeof
from .linalgOps import (
    ARRAY_PROPS,
    ARRAY_TYPE_NAMES,
    LINALG_CTORS,
    LINALG_METHODS,
    LinalgAttr,
    LinalgCtor,
    LinalgOp,
)


def _globals(ctx: TranslateContext) -> dict[str, Any]:
    fn = getattr(ctx, "fn", None)
    g = getattr(fn, "__globals__", None)
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


def _resolve_ctor_obj(node: ast.Call, ctx: TranslateContext) -> tuple[Any, Any]:
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        mod = _globals(ctx).get(func.value.id)
        if mod is None:
            return None, None
        return getattr(mod, func.attr, None), mod
    if isinstance(func, ast.Name):
        return _globals(ctx).get(func.id), None
    return None, None


def resolve_linalg_ctor(node: ast.AST, ctx: TranslateContext) -> Optional[LinalgCtor]:
    """
    Resolve ``ArrayF32(shape)`` / ``linalg.Shape(n)`` to a ``LinalgCtor``.

    #### Raises
    - TypeError - known linalg ctor with bad arity or keywords
    """
    if not isinstance(node, ast.Call):
        return None

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
    return ctor


def resolve_linalg_method(node: ast.AST, ctx: TranslateContext) -> Optional[LinalgOp]:
    """
    Resolve ``a.matmul(b)`` / ``self.a.transpose()`` on a typed Array/Shape receiver.

    #### Raises
    - TypeError - known linalg type with bad arity, keywords, or unknown method
    """
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Attribute):
        return None

    recv = node.func.value
    ty = typeof(recv, ctx)
    if not isinstance(ty, PyCthreadsInternal):
        return None

    table = LINALG_METHODS.get(ty.name)
    if table is None:
        return None
    if node.keywords:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            "linalg method keyword args are not supported"
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


def resolve_linalg_attr(node: ast.AST, ctx: TranslateContext) -> Optional[LinalgAttr]:
    """Resolve ``a.shape`` / ``a.ndim`` on a typed Array local to C++ methods."""
    if not isinstance(node, ast.Attribute):
        return None
    ty = typeof(node.value, ctx)
    if not isinstance(ty, PyCthreadsInternal):
        return None
    if ty.name not in ARRAY_TYPE_NAMES:
        return None
    return ARRAY_PROPS.get(node.attr)


def is_linalg_op(node: ast.AST, ctx: TranslateContext) -> bool:
    """True if this call is a linalg ctor or instance method."""
    if resolve_linalg_method(node, ctx) is not None:
        return True
    return resolve_linalg_ctor(node, ctx) is not None
