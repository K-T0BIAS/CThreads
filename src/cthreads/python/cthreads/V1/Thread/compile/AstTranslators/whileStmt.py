"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.While` — while loop.

Python:
    while cond:
        ...

AST:
    While(test=..., body=[...], orelse=[...])

C++:
    while (<test>) {
        ...
    }

Python while/else is not supported (orelse must be empty).
"""

import ast

from .context import TranslateContext


def translate(node: ast.While, ctx: TranslateContext) -> list[str]:
    """
    Translates an ast.While node to a list of C++ lines

    #### Args
    - node: ast.While - the ast.While node to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - list[str] - a list of C++ lines that represent the translated ast.While node

    #### Example

    ```python
    while (condition):
        statement
    ```

    ----

    ```cpp
    while (condition) {
        statement
    }
    ```
    """
    from .translate import translate_expr, translate_stmt

    if node.orelse: # while/else is not supported
        raise TypeError(
            f"Thread function {ctx.func_name}: while/else is not supported"
        )

    # translate the loop condition to c++
    test = translate_expr(node.test, ctx)
    # start the loop with the condition
    lines = [f"    while ({test}) {{"]

    for stmt in node.body:
        # translate the body of the loop to c++
        nested = translate_stmt(stmt, ctx)
        lines.extend(
            ["    " + line if line.strip() else line for line in nested]
        )
    lines.append("    }")
    # return the list of C++ lines
    return lines
