"""
Ordered plugin lists. Syntax.expr tries these before failing on Call/Attribute.

Concrete plugins register by appending instances (or call register_*).
"""

from __future__ import annotations

import ast

from ..context import TranslationContext
from .base import (
    AttrPlugin,
    CallPlugin,
    MethodOp,
    MethodTablePlugin,
    TranslateExpr,
    method_op,
)

CALL_PLUGINS: list[CallPlugin] = []
ATTR_PLUGINS: list[AttrPlugin] = []


def register_call(plugin: CallPlugin) -> CallPlugin:
    CALL_PLUGINS.append(plugin)
    return plugin


def register_attr(plugin: AttrPlugin) -> AttrPlugin:
    ATTR_PLUGINS.append(plugin)
    return plugin


def lower_call(
    node: ast.Call,
    ctx: TranslationContext,
    translate_expr: TranslateExpr,
) -> str | None:
    for plugin in CALL_PLUGINS:
        out = plugin.try_lower(node, ctx, translate_expr)
        if out is not None:
            return out
    return None


def lower_attr(
    node: ast.Attribute,
    ctx: TranslationContext,
    translate_expr: TranslateExpr,
) -> str | None:
    for plugin in ATTR_PLUGINS:
        out = plugin.try_lower(node, ctx, translate_expr)
        if out is not None:
            return out
    return None


__all__ = [
    "CALL_PLUGINS",
    "ATTR_PLUGINS",
    "CallPlugin",
    "AttrPlugin",
    "MethodTablePlugin",
    "MethodOp",
    "method_op",
    "TranslateExpr",
    "register_call",
    "register_attr",
    "lower_call",
    "lower_attr",
]

# Side-effect: register concrete plugins (Attr: math/linalg before fields fallback)
from . import stdlib as _stdlib  # noqa: E402, F401
from . import shared as _shared  # noqa: E402, F401
from . import cthreads_math as _cthreads_math  # noqa: E402, F401
from . import sync as _sync  # noqa: E402, F401
from . import linalg as _linalg  # noqa: E402, F401
from . import free_fn as _free_fn  # noqa: E402, F401
from . import fields as _fields  # noqa: E402, F401  # last AttrPlugin

