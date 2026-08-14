"""`a.shape` / `a.numel` properties on typed Array receivers."""

from __future__ import annotations

import ast

from .....types import PyCThreadsInternalType
from ...Typeof import Typeof
from ...context import TranslationContext
from ...include import add_include
from ..base import AttrPlugin, TranslateExpr
from .ops import ARRAY_PROPS, ARRAY_TYPE_NAMES


class LinalgPropPlugin(AttrPlugin):
    def try_lower(
        self,
        node: ast.Attribute,
        ctx: TranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        ty = Typeof.of(node.value, ctx)
        if not isinstance(ty, PyCThreadsInternalType):
            return None
        if ty.name not in ARRAY_TYPE_NAMES:
            return None
        prop = ARRAY_PROPS.get(node.attr)
        if prop is None:
            return None

        for line in prop.includes:
            add_include(ctx.body_includes, ctx.seen_body, line)

        recv = translate_expr(node.value, ctx)
        return prop.emit(recv)
