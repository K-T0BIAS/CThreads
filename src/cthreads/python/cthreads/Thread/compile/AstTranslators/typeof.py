"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Resolve the PyType of an expression for method / property lowering.

``ctx.symbols`` types names. Attribute chains use the Threadable class
annotations already registered on ``REGISTRY`` (same source as C++ struct
fields). List subscripts use ``PyList.inner_type``.

Method *tables* stay in the list / sync / linalg resolvers — this only
answers “what is the receiver?”.
"""

from __future__ import annotations

import ast
from typing import get_type_hints

from ....pyTypes import PyList, PyThreadable, PyType, hint_to_pytype
from .context import TranslateContext


def expr_src(node: ast.AST) -> str:
    """Best-effort source for error messages."""
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError):
        return type(node).__name__


def typeof(node: ast.expr, ctx: TranslateContext) -> PyType | None:
    """
    Type of ``node`` if it can be decided from symbols / Threadable fields.

    #### Args
    - node: ast.expr - the expression to type
    - ctx: TranslateContext - needs ``symbols`` (and REGISTRY for fields)

    #### Returns
    - PyType | None - the type, or None if it cannot be determined
    """
    if isinstance(node, ast.Name):
        ty = ctx.symbols.get(node.id)
        return ty if isinstance(ty, PyType) else None

    if isinstance(node, ast.Attribute):
        base = typeof(node.value, ctx)
        if not isinstance(base, PyThreadable):
            return None
        return _threadable_fields(base).get(node.attr)

    if isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Slice):
            return None
        base = typeof(node.value, ctx)
        if isinstance(base, PyList):
            return base.inner_type
        return None

    return None


def _threadable_fields(ty: PyThreadable) -> dict[str, PyType]:
    """Field name -> PyType from the live @Threadable class, if registered."""
    from ....CONFIG import REGISTRY

    cls = REGISTRY.threadables.get(ty.name)
    if cls is None:
        return {}
    hints = get_type_hints(cls, localns={cls.__name__: cls})
    return {name: hint_to_pytype(hint) for name, hint in hints.items()}
