from __future__ import annotations

import ast

from ....types import PyType
from ..Cpp import Cpp
from ..context import TranslationContext
from ..include import add_include


class Literal:
    """ast.Constant and ast.List."""

    @staticmethod
    def constant(node: ast.Constant, ctx: TranslationContext) -> str:
        return Cpp.literal(node.value)

    @staticmethod
    def list_display(node: ast.List, ctx: TranslationContext) -> str:
        from .Syntax import Syntax

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
            got = Literal._elem_cpp_type(elt, ctx)
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

        parts = [Syntax.expr(elt, ctx) for elt in node.elts]
        return f"std::vector<{elem_ty}>{{{', '.join(parts)}}}"

    @staticmethod
    def _elem_cpp_type(node: ast.expr, ctx: TranslationContext) -> str | None:
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
                got = Literal._elem_cpp_type(elt, ctx)
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
