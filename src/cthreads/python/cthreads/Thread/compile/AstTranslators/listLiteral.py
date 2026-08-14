"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.List` — a list display.

Python:  [1, 2, 3]          [x, y]         [[1, 2], [3, 4]]      []
C++:     std::vector<int>{1, 2, 3}   std::vector<int>{x, y}  ...   {}

Element type is inferred from constants, typed names, and nested lists.
Empty `[]` emits `{}` so `xs: list[int] = []` can take the type from the
annotation. Starred elts (`[*xs]`) are not supported.
"""

from __future__ import annotations

import ast

from ....pyTypes import PyType
from ..lib import add_include
from .context import TranslateContext


def _elem_cpp_type(node: ast.expr, ctx: TranslateContext) -> str | None:
    """
    Infer the C++ type of one list element, or None if it cannot be typed
    (empty nested list, call, binop, ...).
    """
    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, bool):
            return "bool"
        if isinstance(val, int):
            return "int"
        if isinstance(val, float):
            return "double"
        if isinstance(val, str):
            return "std::string"
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported list element literal {type(val).__name__}"
        )

    if isinstance(node, ast.Name):
        ty = ctx.symbols.get(node.id)
        if isinstance(ty, PyType):
            return ty.cpp_name
        return None

    if isinstance(node, ast.List):
        inner = None
        for elt in node.elts:
            got = _elem_cpp_type(elt, ctx)
            if got is None:
                continue
            if inner is None:
                inner = got
            elif inner != got:
                raise TypeError(
                    f"Thread function {ctx.func_name}: "
                    f"mixed nested list types {inner} and {got}"
                )
        if inner is None:
            return None
        return f"std::vector<{inner}>"

    return None


def translate(node: ast.List, ctx: TranslateContext) -> str:
    """
    Translates a list display to a C++ `std::vector<T>{...}` expression
    (or `{}` for an empty list).

    #### Args
    - node: ast.List - the list display to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - str - a C++ expression that represents the translated list
    """
    from .translate import translate_expr

    for elt in node.elts:
        if isinstance(elt, ast.Starred):
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                "starred list elements are not supported"
            )

    if not node.elts:
        add_include(ctx.body_includes, ctx.seen_body, "#include <vector>\n")
        return "{}"

    elem_ty = None
    for elt in node.elts:
        got = _elem_cpp_type(elt, ctx)
        if got is None:
            continue
        if elem_ty is None:
            elem_ty = got
        elif elem_ty != got:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"mixed list element types {elem_ty} and {got}"
            )

    if elem_ty is None:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            "cannot infer list element type "
            "(use a typed name, a literal, or an annotated assignment)"
        )

    add_include(ctx.body_includes, ctx.seen_body, "#include <vector>\n")
    if "std::string" in elem_ty:
        add_include(ctx.body_includes, ctx.seen_body, "#include <string>\n")

    parts = [translate_expr(elt, ctx) for elt in node.elts]
    return f"std::vector<{elem_ty}>{{{', '.join(parts)}}}"
