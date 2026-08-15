from __future__ import annotations

import ast

from ....types import hint_to_pytype
from ..Cpp import Cpp
from ..Source import Source
from ..context import TranslationContext
from ..include import add_include, include_for


class Assign:
    """ast.AnnAssign / Assign / AugAssign."""

    @staticmethod
    def ann_assign(node: ast.AnnAssign, ctx: TranslationContext) -> list[str]:
        from .Syntax import Syntax

        if not isinstance(node.target, ast.Name):
            raise TypeError(
                f"Thread function {ctx.func_name}: AnnAssign target must be a plain name"
            )
        var_name = node.target.id
        if var_name in ctx.symbols:
            raise TypeError(
                f"Thread function {ctx.func_name}: redeclaration of {var_name!r}"
            )
        hint = Source.resolve_annotation(node.annotation, ctx.fn.__globals__)
        py_type = hint_to_pytype(hint)
        ctx.symbols[var_name] = py_type
        add_include(
            ctx.body_includes, ctx.seen_body, include_for(py_type, ctx.this_file)
        )
        if node.value is None:
            decl, _ = py_type.to_cpp(var_name)
            return [f"    {decl}"]
        rhs = Syntax.expr(node.value, ctx)
        decl, _ = py_type.to_cpp(var_name, rhs)
        return [f"    {decl}"]

    @staticmethod
    def assign(node: ast.Assign, ctx: TranslationContext) -> list[str]:
        from .Syntax import Syntax

        if len(node.targets) != 1:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                "only single-target assignment is supported"
            )
        target = node.targets[0]
        if isinstance(target, ast.Name):
            if target.id not in ctx.symbols:
                raise TypeError(
                    f"Thread function {ctx.func_name}: "
                    f"assign to unknown name {target.id!r} "
                    "(declare it with an annotated assignment first)"
                )
        elif isinstance(target, ast.Subscript):
            if isinstance(target.slice, ast.Slice):
                raise TypeError(
                    f"Thread function {ctx.func_name}: "
                    "slice assignment is not supported"
                )
        elif not isinstance(target, ast.Attribute):
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"unsupported assign target {type(target).__name__}"
            )
        lhs = Syntax.expr(target, ctx)
        rhs = Syntax.expr(node.value, ctx)
        return [f"    {lhs} = {rhs};"]

    @staticmethod
    def aug_assign(node: ast.AugAssign, ctx: TranslationContext) -> list[str]:
        from .Op import Op
        from .Syntax import Syntax

        target = Syntax.expr(node.target, ctx)
        value = Syntax.expr(node.value, ctx)
        if isinstance(node.op, ast.Pow):
            add_include(ctx.body_includes, ctx.seen_body, Cpp.CMATH)
            return [f"    {target} = std::pow({target}, {value});"]
        op = Op.BINOPS.get(type(node.op))
        if not op:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"unsupported aug-assign operator {type(node.op).__name__}"
            )
        return [f"    {target} {op}= {value};"]
