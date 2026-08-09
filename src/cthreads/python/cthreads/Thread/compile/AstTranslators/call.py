"""
Translate `ast.Call` — currently math-stdlib calls only.

Python:  math.sqrt(x)
AST:     Call(func=Attribute(Name('math'), 'sqrt'), args=[Name('x')])
C++:     std::sqrt(x)
"""

from __future__ import annotations

import ast

from ..lib import add_include
from ..mathLibTranslators import CMATH_INCLUDE, resolve_math_call
from .context import TranslateContext


def translate(node: ast.Call, ctx: TranslateContext) -> str:
    from .translate import translate_expr

    op = resolve_math_call(node, ctx)
    if op is None:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported call (only whitelisted math.* is supported)"
        )

    add_include(ctx.body_includes, ctx.seen_body, CMATH_INCLUDE)
    args = ", ".join(translate_expr(a, ctx) for a in node.args)
    return f"{op.cpp_func}({args})"
