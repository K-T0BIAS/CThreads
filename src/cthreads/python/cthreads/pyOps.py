"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Operator tables and builtin-call whitelist for AST lowering.

In the AST, `a * b` is not the character `*` — it is:

    BinOp(left=..., op=ast.Mult(), right=...)

When we emit C++, we look up `type(node.op)` in these maps.

Builtin calls (`range`, `len`, ...) are matched by bare name — same style as
``for i in range(n)`` — not via pybind helpers.
"""

from __future__ import annotations

import ast

# Bare-name builtins lowered in codegen (not cthreads.* / math.* modules).
BUILTINS: frozenset[str] = frozenset({"range", "len"})


def is_builtin_call(node: ast.AST, name: str) -> bool:
    """True for `name(...)` when `name` is a whitelisted builtin."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        and name in BUILTINS
    )


# Binary operators: used by ast.BinOp and as the op inside ast.AugAssign
BINOPS: dict[type, str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",          # Python 3 `/` is true division -> usually C++ `/` on doubles
    ast.FloorDiv: "/",     # `//` — only valid for ints in our subset; revisit later
    ast.Mod: "%",
    # ast.Pow handled specially in binOp / augAssign → std::pow(...)
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
}

UNARYOPS: dict[type, str] = {
    ast.UAdd: "+",
    ast.USub: "-",
    ast.Not: "!",          # logical not; Python `not`
    ast.Invert: "~",       # bitwise invert
}

# Compare ops live on ast.Compare.ops (a list); values are C++ tokens
CMPOPS: dict[type, str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    # ast.Is / ast.In etc. have no direct C++ map for our numeric subset
}

# BoolOps live on ast.BoolOp.op; values are C++ tokens (short-circuit)
BOOLOPS: dict[type, str] = {
    ast.And: "&&",
    ast.Or: "||",
}
