"""
Dispatch AST nodes to per-type translator units.

Expressions return a C++ expr string; statements return C++ lines.

Example walk — Python:

    b: int = a + 10

AST:

    AnnAssign(
      target=Name('b'),
      annotation=Name('int'),
      value=BinOp(left=Name('a'), op=Add(), right=Constant(10)),
    )

Call chain:

    translate_stmt(AnnAssign)
      -> annAssign.translate
           -> translate_expr(BinOp)                  # Op root of the RHS
                -> translate_expr(Name 'a')  -> "a"
                -> translate_expr(Constant 10) -> "10"
                -> "(a + 10)"
           -> "    int b = (a + 10);"

C++:

    int b = (a + 10);
"""

import ast
from dataclasses import dataclass
from typing import Any, Optional

from . import (
    annAssign,
    assign,
    attribute,
    augAssign,
    binOp,
    boolOp,
    breakStmt,
    compare,
    constant,
    continueStmt,
    exprStmt,
    forStmt,
    ifStmt,
    name,
    passStmt,
    returnStmt,
    subscript,
    unaryOp,
    whileStmt,
)
from .context import ExprTranslator, StmtTranslator, TranslateContext
from .signature import SignatureParts, translate_signature

EXPR_TRANSLATORS: dict[type, ExprTranslator] = {
    ast.Constant: constant.translate,
    ast.Name: name.translate,
    ast.Attribute: attribute.translate,
    ast.Subscript: subscript.translate,
    ast.BinOp: binOp.translate,
    ast.UnaryOp: unaryOp.translate,
    ast.Compare: compare.translate,
    ast.BoolOp: boolOp.translate,
}

STMT_TRANSLATORS: dict[type, StmtTranslator] = {
    ast.AnnAssign: annAssign.translate,
    ast.Assign: assign.translate,
    ast.AugAssign: augAssign.translate,
    ast.Pass: passStmt.translate,
    ast.Return: returnStmt.translate,
    ast.Expr: exprStmt.translate,
    ast.If: ifStmt.translate,
    ast.For: forStmt.translate,
    ast.While: whileStmt.translate,
    ast.Break: breakStmt.translate,
    ast.Continue: continueStmt.translate,
}


@dataclass
class TranslateResult:
    return_type: str
    func_name: str
    params_csv: str
    sig_includes: list[str]
    body: str
    body_includes: list[str]

    def free_signature(self) -> str:
        # CTHREADS_API -> extern "C" + dllexport on MSVC (see cthreads_export.hpp)
        return f"CTHREADS_API {self.return_type} {self.func_name}({self.params_csv})"

    def method_decl(self) -> str:
        return f"    {self.return_type} {self.func_name}({self.params_csv});"

    def method_def_signature(self, owner_name: str) -> str:
        return f"{self.return_type} {owner_name}::{self.func_name}({self.params_csv})"


def translate_expr(node: ast.expr, ctx: TranslateContext) -> str:
    translator = EXPR_TRANSLATORS.get(type(node))
    if translator is None:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported expression {type(node).__name__}"
        )
    return translator(node, ctx)


def translate_stmt(node: ast.stmt, ctx: TranslateContext) -> list[str]:
    translator = STMT_TRANSLATORS.get(type(node))
    if translator is None:
        return [f"    // unsupported statement: {type(node).__name__}"]
    return translator(node, ctx)


def translate_function(
    fn,
    func_def: ast.FunctionDef,
    hints: dict[str, Any],
    owner_name: Optional[str] = None,
) -> TranslateResult:
    ctx = TranslateContext(
        fn=fn,
        func_name=func_def.name,
        hints=hints,
        owner_name=owner_name,
    )
    parts: SignatureParts = translate_signature(func_def, ctx)

    body_lines: list[str] = []
    for stmt in func_def.body:
        body_lines.extend(translate_stmt(stmt, ctx))

    body = "\n".join(body_lines)
    if body:
        body = body + "\n"

    return TranslateResult(
        return_type=parts.return_type,
        func_name=parts.func_name,
        params_csv=parts.params_csv,
        sig_includes=ctx.sig_includes,
        body=body,
        body_includes=ctx.body_includes,
    )
