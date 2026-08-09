"""
Translate `ast.Call` — builtins + stdlib math.* / cthreads.math.*.

Python:  len(xs)          math.sqrt(x)     math.abs via cthreads.math
C++:     (xs).size()      std::sqrt(x)     cthreads::math::abs(x)
"""

from __future__ import annotations

import ast

from ....pyOps import is_builtin_call
from ..lib import add_include
from ..mathLibTranslators import CMATH_INCLUDE, resolve_math_call
from .context import TranslateContext


def translate(node: ast.Call, ctx: TranslateContext) -> str:
    from .translate import translate_expr

    if is_builtin_call(node, "len"):
        if node.keywords:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                "len() keyword args are not supported"
            )
        if len(node.args) != 1:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"len() expects 1 arg, got {len(node.args)}"
            )
        arg = translate_expr(node.args[0], ctx)
        return f"({arg}).size()"

    op = resolve_math_call(node, ctx)
    if op is None:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported call (only whitelisted builtins / math.* / cthreads.math.*)"
        )

    if op.cpp_include:
        add_include(ctx.body_includes, ctx.seen_body, f'#include "{op.cpp_include}"\n')
    else:
        add_include(ctx.body_includes, ctx.seen_body, CMATH_INCLUDE)

    args = ", ".join(translate_expr(a, ctx) for a in node.args)
    return f"{op.cpp_func}({args})"
