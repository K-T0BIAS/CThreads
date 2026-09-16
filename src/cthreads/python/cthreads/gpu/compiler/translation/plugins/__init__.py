"""
Ordered GPU plugin lists. GpuSyntax.expr tries these for Call / Attribute.
"""

from __future__ import annotations

import ast

from ..context import GpuTranslationContext
from .base import AttrPlugin, CallPlugin, TranslateExpr

CALL_PLUGINS: list[CallPlugin] = []
ATTR_PLUGINS: list[AttrPlugin] = []


def register_call(plugin: CallPlugin) -> CallPlugin:
    """
    Append a CallPlugin to the GPU call registry.

    #### Args:
    - plugin: CallPlugin = plugin instance to register

    #### Returns
    - CallPlugin = the same plugin (for chaining)
    """
    CALL_PLUGINS.append(plugin)
    return plugin


def register_attr(plugin: AttrPlugin) -> AttrPlugin:
    """
    Append an AttrPlugin to the GPU attribute registry.

    #### Args:
    - plugin: AttrPlugin = plugin instance to register

    #### Returns
    - AttrPlugin = the same plugin (for chaining)
    """
    ATTR_PLUGINS.append(plugin)
    return plugin


def lower_call(
    node: ast.Call,
    ctx: GpuTranslationContext,
    translate_expr: TranslateExpr,
) -> str | None:
    """
    Try each CallPlugin until one returns GLSL text.

    #### Args:
    - node: ast.Call = call expression
    - ctx: GpuTranslationContext = current GPU translation state
    - translate_expr: TranslateExpr = nested expression lowerer

    #### Returns
    - str | None = GLSL text, or None if no plugin matched
    """
    for plugin in CALL_PLUGINS:
        out = plugin.try_lower(node, ctx, translate_expr)
        if out is not None:
            return out
    return None


def lower_attr(
    node: ast.Attribute,
    ctx: GpuTranslationContext,
    translate_expr: TranslateExpr,
) -> str | None:
    """
    Try each AttrPlugin until one returns GLSL text.

    #### Args:
    - node: ast.Attribute = attribute expression
    - ctx: GpuTranslationContext = current GPU translation state
    - translate_expr: TranslateExpr = nested expression lowerer

    #### Returns
    - str | None = GLSL text, or None if no plugin matched
    """
    for plugin in ATTR_PLUGINS:
        out = plugin.try_lower(node, ctx, translate_expr)
        if out is not None:
            return out
    return None


__all__ = [
    "CALL_PLUGINS",
    "ATTR_PLUGINS",
    "AttrPlugin",
    "CallPlugin",
    "TranslateExpr",
    "register_call",
    "register_attr",
    "lower_call",
    "lower_attr",
]

# Side-effect: register concrete plugins.
from .indexes import IndexAttrPlugin  # noqa: E402
from .math_calls import MathCallPlugin  # noqa: E402

register_attr(IndexAttrPlugin())
register_call(MathCallPlugin())
