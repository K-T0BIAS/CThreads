"""
Operator tables for future expression lowering (BinOp / UnaryOp / AugAssign).

In the AST, `a * b` is not the character `*` — it is:

    BinOp(left=..., op=ast.Mult(), right=...)

When we emit C++, we look up `type(node.op)` in these maps.
"""

import ast

# Binary operators: used by ast.BinOp and as the op inside ast.AugAssign
BINOPS: dict[type, str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",          # Python 3 `/` is true division → usually C++ `/` on doubles
    ast.FloorDiv: "/",     # `//` — only valid for ints in our subset; revisit later
    ast.Mod: "%",
    ast.Pow: "",           # no C++ `**`; use std::pow later or reject
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
