"""
Translate `ast.For`.

Two supported shapes:

1) Container iteration (Python list -> std::vector):

    for x in xs:
        ...

    for (auto& x : xs) {
        ...
    }

2) range(...) — lowered like an index loop, not a range object:

    for i in range(n):          -> for (int i = 0; i < n; i += 1)
    for i in range(a, b):       -> for (int i = a; i < b; i += 1)
    for i in range(a, b, s):    -> for (int i = a; i < b; i += s)

    Step is assumed positive (same common case as arrange-style loops).
"""

import ast

from ....pyOps import is_builtin_call
from ....pyTypes import PyInt, PyList
from .context import TranslateContext


def translate(node: ast.For, ctx: TranslateContext) -> list[str]:
    from .translate import translate_expr, translate_stmt

    if node.orelse:
        raise TypeError(
            f"Thread function {ctx.func_name}: for/else is not supported"
        )
    if not isinstance(node.target, ast.Name):
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            "for-loop target must be a plain name"
        )

    loop_var = node.target.id
    if loop_var in ctx.symbols:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"for-loop rebinds existing name {loop_var!r}"
        )

    def nest(lines: list[str]) -> list[str]:
        return ["    " + line if line.strip() else line for line in lines]

    # --- range(n) / range(a, b) / range(a, b, s) ---
    it = node.iter
    if is_builtin_call(it, "range"):
        assert isinstance(it, ast.Call)
        if it.keywords:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                "range() keyword args are not supported"
            )
        n = len(it.args)
        if n == 1:
            start, stop, step = "0", translate_expr(it.args[0], ctx), "1"
        elif n == 2:
            start = translate_expr(it.args[0], ctx)
            stop = translate_expr(it.args[1], ctx)
            step = "1"
        elif n == 3:
            start = translate_expr(it.args[0], ctx)
            stop = translate_expr(it.args[1], ctx)
            step = translate_expr(it.args[2], ctx)
        else:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"range() expects 1..3 args, got {n}"
            )

        ctx.symbols[loop_var] = PyInt()
        lines = [
            f"    for (int {loop_var} = {start}; "
            f"{loop_var} < {stop}; "
            f"{loop_var} += {step}) {{"
        ]
        for stmt in node.body:
            lines.extend(nest(translate_stmt(stmt, ctx)))
        lines.append("    }")
        del ctx.symbols[loop_var]
        return lines

    # --- for x in container  (list / std::vector) ---
    if not isinstance(it, ast.Name):
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            "for-iter must be a name or range(...)"
        )
    container_ty = ctx.symbols.get(it.id)
    if not isinstance(container_ty, PyList):
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"for-iter {it.id!r} must be a list[...], got {type(container_ty).__name__}"
        )

    container = translate_expr(it, ctx)
    ctx.symbols[loop_var] = container_ty.inner_type
    lines = [f"    for (auto& {loop_var} : {container}) {{"]
    for stmt in node.body:
        lines.extend(nest(translate_stmt(stmt, ctx)))
    lines.append("    }")
    del ctx.symbols[loop_var]
    return lines
