"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

cthreads.linalg method / ctor tables for @Thread lowering.

Python names match the pybind surface. Receivers are typed
``PyCthreadsInternal`` locals (``ArrayF32`` / ``Shape`` / …).
Ctors resolve through ``ctx.fn.__globals__`` like ``cthreads.math``.
"""

from collections.abc import Callable
from typing import NamedTuple

Emit = Callable[[str, list[str]], str]  # (receiver_cpp, arg_cpps) -> expr
CtorEmit = Callable[[list[str]], str]  # (arg_cpps) -> expr


class LinalgOp(NamedTuple):
    emit: Emit
    min_arity: int
    max_arity: int
    cpp_include: str | None = "linalg/array.hpp"


class LinalgCtor(NamedTuple):
    cpp_type: str
    min_arity: int
    max_arity: int
    cpp_include: str
    extra_includes: tuple[str, ...] = ()


class LinalgAttr(NamedTuple):
    emit: Callable[[str], str]
    cpp_include: str | None = "linalg/array.hpp"


def _method(name: str, min_arity: int = 0, max_arity: int | None = None) -> LinalgOp:
    """1:1 Python method -> ``(recv).name(args...)``."""
    if max_arity is None:
        max_arity = min_arity

    def emit(recv: str, args: list[str]) -> str:
        if not args:
            return f"({recv}).{name}()"
        return f"({recv}).{name}({', '.join(args)})"

    return LinalgOp(emit=emit, min_arity=min_arity, max_arity=max_arity)


def _emit_count(recv: str, args: list[str]) -> str:
    return f"({recv}).count_nonzero()"


def _emit_any(recv: str, args: list[str]) -> str:
    return f"(({recv}).count_nonzero() != 0)"


def _emit_all(recv: str, args: list[str]) -> str:
    return f"(({recv}).count_nonzero() == ({recv}).shape().numel())"


ARRAY_HPP = "linalg/array.hpp"
SHAPE_HPP = "linalg/shape.hpp"
SLICE_HPP = "linalg/slice.hpp"

ARRAY_TYPE_NAMES: frozenset[str] = frozenset(
    {"ArrayF32", "ArrayF64", "ArrayI32", "ArrayBool"}
)
NUMERIC_ARRAY_NAMES: frozenset[str] = frozenset(
    {"ArrayF32", "ArrayF64", "ArrayI32"}
)

# Shared by every Array* (including ArrayBool).
ARRAY_METHODS: dict[str, LinalgOp] = {
    "view": _method("view", 1),
    "reshape": _method("reshape", 1),
    "flatten": _method("flatten", 0),
    "transpose": _method("transpose", 0),
    "permute": _method("permute", 1),
    "squeeze": _method("squeeze", 1),
    "unsqueeze": _method("unsqueeze", 1),
    "contiguous": _method("contiguous", 0),
    "_contiguous": _method("_contiguous", 0),
    "_view": _method("_view", 1),
    "_reshape": _method("_reshape", 1),
    "_flatten": _method("_flatten", 0),
    "_transpose": _method("_transpose", 0),
    "_permute": _method("_permute", 1),
    "_squeeze": _method("_squeeze", 1),
    "_unsqueeze": _method("_unsqueeze", 1),
    "masked_select": _method("masked_select", 1),
    "masked_fill": _method("masked_fill", 2),
    "masked_scatter": _method("masked_scatter", 2),
    "is_contiguous": _method("is_contiguous", 0),
    "ndim": _method("ndim", 0),
    "offset": _method("offset", 0),
    "count": LinalgOp(emit=_emit_count, min_arity=0, max_arity=0),
    "any": LinalgOp(emit=_emit_any, min_arity=0, max_arity=0),
    "all": LinalgOp(emit=_emit_all, min_arity=0, max_arity=0),
}

# Numeric arrays only (not ArrayBool).
ARRAY_NUMERIC_METHODS: dict[str, LinalgOp] = {
    "matmul": _method("matmul", 1),
    "dot": _method("dot", 1),
    "cross": _method("cross", 1),
    "_add": _method("_add", 1),
    "_sub": _method("_sub", 1),
    "_mul": _method("_mul", 1),
    "_div": _method("_div", 1),
    "_neg": _method("_neg", 0),
    "_matmul": _method("_matmul", 1),
    "_dot": _method("_dot", 1),
    "_cross": _method("_cross", 1),
}

SHAPE_METHODS: dict[str, LinalgOp] = {
    "ndim": LinalgOp(
        emit=lambda recv, args: f"({recv}).ndim()",
        min_arity=0,
        max_arity=0,
        cpp_include=SHAPE_HPP,
    ),
    "numel": LinalgOp(
        emit=lambda recv, args: f"({recv}).numel()",
        min_arity=0,
        max_arity=0,
        cpp_include=SHAPE_HPP,
    ),
    "strides": LinalgOp(
        emit=lambda recv, args: f"({recv}).strides()",
        min_arity=0,
        max_arity=0,
        cpp_include=SHAPE_HPP,
    ),
}

# Python properties on Array -> C++ methods.
ARRAY_PROPS: dict[str, LinalgAttr] = {
    "shape": LinalgAttr(emit=lambda recv: f"({recv}).shape()"),
    "strides": LinalgAttr(emit=lambda recv: f"({recv}).strides()"),
    "ndim": LinalgAttr(emit=lambda recv: f"({recv}).ndim()"),
    "numel": LinalgAttr(emit=lambda recv: f"({recv}).shape().numel()"),
    "offset": LinalgAttr(emit=lambda recv: f"({recv}).offset()"),
}

LINALG_CTORS: dict[str, LinalgCtor] = {
    "ArrayF32": LinalgCtor(
        "cthreads::linalg::Array<float>", 1, 1, ARRAY_HPP
    ),
    "ArrayF64": LinalgCtor(
        "cthreads::linalg::Array<double>", 1, 1, ARRAY_HPP
    ),
    "ArrayI32": LinalgCtor(
        "cthreads::linalg::Array<int>", 1, 1, ARRAY_HPP
    ),
    "ArrayBool": LinalgCtor(
        "cthreads::linalg::Array<uint8_t>",
        1,
        1,
        ARRAY_HPP,
        extra_includes=("cstdint",),
    ),
    "Shape": LinalgCtor("cthreads::linalg::Shape", 1, 1, SHAPE_HPP),
    "Slice": LinalgCtor("cthreads::linalg::Slice", 0, 3, SLICE_HPP),
}

# PyCthreadsInternal.name -> method table
LINALG_METHODS: dict[str, dict[str, LinalgOp]] = {
    "ArrayF32": {**ARRAY_METHODS, **ARRAY_NUMERIC_METHODS},
    "ArrayF64": {**ARRAY_METHODS, **ARRAY_NUMERIC_METHODS},
    "ArrayI32": {**ARRAY_METHODS, **ARRAY_NUMERIC_METHODS},
    "ArrayBool": dict(ARRAY_METHODS),
    "Shape": SHAPE_METHODS,
}

MASK_AND = "cthreads::linalg::mask_and"
MASK_OR = "cthreads::linalg::mask_or"
MASK_XOR = "cthreads::linalg::mask_xor"
MASK_NOT = "cthreads::linalg::mask_not"
