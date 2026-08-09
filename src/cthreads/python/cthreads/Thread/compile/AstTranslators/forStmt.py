"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

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
    """
    Translates for statements to a list of C++ lines
    
    #### Args
    - node: ast.For - the ast.For node to translate
    - ctx: TranslateContext - the translate context of the function
    
    #### Returns
    - list[str] - a list of C++ lines that represent the translated ast.For node
    """
    from .translate import translate_expr, translate_stmt

    if node.orelse: # for/else is not supported
        raise TypeError(
            f"Thread function {ctx.func_name}: for/else is not supported"
        )
    if not isinstance(node.target, ast.Name): # for-loop target must be a plain name
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            "for-loop target must be a plain name"
        )

    # get the loop variable name
    loop_var = node.target.id
    if loop_var in ctx.symbols: # loop variable must not already be in the symbols to ensure the globals stay intact
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"for-loop rebinds existing name {loop_var!r}"
        )

    # helper function to add indentation to each line
    def nest(lines: list[str]) -> list[str]:
        return ["    " + line if line.strip() else line for line in lines]

    # if this is a range based for loop (range(n) / range(a, b) / range(a, b, s))
    it = node.iter
    if is_builtin_call(it, "range"): # is a range based for loop
        assert isinstance(it, ast.Call) # must be a call node (range())
        if it.keywords: # range keyword args are not supported
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                "range() keyword args are not supported"
            )
        # get the um of args in the range 
        n = len(it.args)
        if n == 1: # range(n) -> 0 to n-1
            start, stop, step = "0", translate_expr(it.args[0], ctx), "1"
        elif n == 2: # range(a, b) -> a to b-1
            start = translate_expr(it.args[0], ctx)
            stop = translate_expr(it.args[1], ctx)
            step = "1"
        elif n == 3: # range(a, b, s) -> a to b-1 by s
            start = translate_expr(it.args[0], ctx)
            stop = translate_expr(it.args[1], ctx)
            step = translate_expr(it.args[2], ctx)
        else: # range() expects 1..3 args, got n this is an error
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"range() expects 1..3 args, got {n}"
            )

        # build a normal for loop to iterate over the range
        ctx.symbols[loop_var] = PyInt()
        lines = [
            f"    for (int {loop_var} = {start}; "
            f"{loop_var} < {stop}; "
            f"{loop_var} += {step}) {{"
        ]
        # translate the body of the for loop
        for stmt in node.body:
            lines.extend(nest(translate_stmt(stmt, ctx)))
        lines.append("    }")
        # remove the loop variable from the symbols (its not global only local to the loop 
        # (was added since the loop body uses it like global))
        del ctx.symbols[loop_var] 
        return lines

    # translate conainer loops like for x in xs:
    if not isinstance(it, ast.Name): # for-iter must be a name
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            "for-iter must be a name or range(...)"
        )
    # get the type of the container
    container_ty = ctx.symbols.get(it.id)
    if not isinstance(container_ty, PyList): # container must be a list (so that it can become std::vector<T>)
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"for-iter {it.id!r} must be a list[...], got {type(container_ty).__name__}"
        )
    # translate the container expression
    container = translate_expr(it, ctx)
    ctx.symbols[loop_var] = container_ty.inner_type # set the loop variable type to the inner type of the container
    # build the for loop
    lines = [f"    for (auto& {loop_var} : {container}) {{"]
    # translate the body of the for loop
    for stmt in node.body:
        lines.extend(nest(translate_stmt(stmt, ctx)))
    lines.append("    }")
    # remove the loop variable from the symbols (its not global only local to the loop 
    # (was added since the loop body uses it like global))
    del ctx.symbols[loop_var]
    # return the lines
    return lines
