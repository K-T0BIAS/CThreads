"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Dispatch AST nodes to per-type translator units.

Expressions return a C++ expr string; statements return C++ lines.

Example walk - Python:

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

# get all the translators for the different types of nodes
from . import (
    annAssign,
    assign,
    attribute,
    augAssign,
    binOp,
    boolOp,
    breakStmt,
    call,
    compare,
    constant,
    continueStmt,
    exprStmt,
    forStmt,
    ifStmt,
    listLiteral,
    name,
    passStmt,
    returnStmt,
    subscript,
    unaryOp,
    whileStmt,
)
from .context import ExprTranslator, StmtTranslator, TranslateContext
from .signature import SignatureParts, translate_signature

# dict of the translators for the different types of expressions
EXPR_TRANSLATORS: dict[type, ExprTranslator] = {
    ast.Constant: constant.translate,
    ast.Name: name.translate,
    ast.Attribute: attribute.translate,
    ast.Subscript: subscript.translate,
    ast.BinOp: binOp.translate,
    ast.UnaryOp: unaryOp.translate,
    ast.Compare: compare.translate,
    ast.BoolOp: boolOp.translate,
    ast.Call: call.translate,
    ast.List: listLiteral.translate,
}

# dict of the translators for the different types of statements
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
    """
    A dataclass to store the result of translating a function

    #### Attributes
    - return_type: str - the return type of the function
    - func_name: str - the name of the function
    - params_csv: str - the parameters of the function as a comma separated string
    - sig_includes: list[str] - a list of include statements for the function signature
    - body: str - the body of the function as a string
    - body_includes: list[str] - a list of include statements for the function body

    #### Methods
    - free_signature() -> str: returns the c++ function signature as a string
    - method_decl() -> str: returns the c++ function declaration as a string
    - method_def_signature(owner_name: str) -> str: returns the c++ function definition signature as a string
    """
    return_type: str # the return type of the function
    func_name: str # the name of the function
    params_csv: str # the parameters of the function as a comma separated string
    sig_includes: list[str] # a list of include statements for the function signature
    body: str # the body of the function as a string
    body_includes: list[str] # a list of include statements for the function body
    def free_signature(self) -> str:
        """
        Returns the c++ function signature as a string

        #### Returns
        - str - the c++ function signature as a string

        #### Example
        ```cpp
        CTHREADS_API int add(int a, int b);
        ```
        """
        # CTHREADS_API -> extern "C" + dllexport on MSVC (see cthreads_export.hpp)
        return f"CTHREADS_API {self.return_type} {self.func_name}({self.params_csv})"

    def method_decl(self) -> str:
        """
        Returns the c++ function declaration as a string (used if the function is a method)

        NOTE: since this is a method declaration, it does not have the CTHREADS_API prefix

        #### Returns
        - str - the c++ function declaration as a string

        #### Example
        ```cpp
        int add(int a, int b);
        ```
        """
        return f"    {self.return_type} {self.func_name}({self.params_csv});"

    def method_def_signature(self, owner_name: str) -> str:
        """
        Returns the c++ function definition signature as a string (used if the function is a method)

        #### Args
        - owner_name: str - the name of the owner of the function

        #### Returns
        - str - the c++ function definition signature as a string

        #### Example
        ```cpp
        int MyClass::add(int a, int b);
        ```
        """
        return f"{self.return_type} {owner_name}::{self.func_name}({self.params_csv})"


def translate_expr(node: ast.expr, ctx: TranslateContext) -> str:
    """
    Translates any ast.expr node to a C++ expression.
    Internally resolves the correct translator for the node type and calls it.
    If no translator is found for the node type, an error is raised.

    #### Args
    - node: ast.expr - the ast.expr node to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - str - a C++ expression that represents the translated ast.expr node
    """
    translator = EXPR_TRANSLATORS.get(type(node)) # try to get the translator for this node type
    if translator is None: # no translator found so unsupported
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported expression {type(node).__name__}"
        )
    return translator(node, ctx) # call the translator and return the result


def translate_stmt(node: ast.stmt, ctx: TranslateContext) -> list[str]:
    """
    Translates any ast.stmt node to a list of C++ lines.
    Internally resolves the correct translator for the node type and calls it.
    If no translator is found for the node type, an error is raised.

    #### Args
    - node: ast.stmt - the ast.stmt node to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - list[str] - a list of C++ lines that represent the translated ast.stmt node
    """
    translator = STMT_TRANSLATORS.get(type(node)) # try to get the translator for this node type
    if translator is None: # no translator found so unsupported
        return [f"    // unsupported statement: {type(node).__name__}"]
    return translator(node, ctx) # call the translator and return the result


def translate_function(
    fn,
    func_def: ast.FunctionDef,
    hints: dict[str, Any],
    owner_name: Optional[str] = None,
) -> TranslateResult:
    """
    Translates a function to a TranslateResult object.
    Works for free functions and methods.

    #### Args
    - fn: function - the function to translate
    - func_def: ast.FunctionDef - the function definition to translate
    - hints: dict[str, Any] - the hints for the function
    - owner_name: Optional[str] - the name of the owner of the function (may be None for free functions)
    
    #### Returns
    - TranslateResult - a TranslateResult object that contains the translated function
    """
    # init the translate context
    ctx = TranslateContext(
        fn=fn,
        func_name=func_def.name,
        hints=hints,
        owner_name=owner_name,
    )
    # translate the signature of the function
    parts: SignatureParts = translate_signature(func_def, ctx)
    # translate the body of the function
    body_lines: list[str] = []
    for stmt in func_def.body:
        # iterate over the body of the function and translate each statement
        body_lines.extend(translate_stmt(stmt, ctx))
    # join the body lines into a single string
    body = "\n".join(body_lines)
    if body: # if the body is not empty, add a newline
        body = body + "\n"
    # return the TranslateResult object
    return TranslateResult(
        return_type=parts.return_type, # the return type of the function
        func_name=parts.func_name,     # the name of the function
        params_csv=parts.params_csv,   # the parameters of the function as a comma separated string
        sig_includes=ctx.sig_includes, # the includes for the function signature
        body=body,                     # the body of the function as a string
        body_includes=ctx.body_includes, # the includes for the function body
    )
