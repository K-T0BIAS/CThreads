"""
cthreads.linalg method / ctor / property tables.

Converted to MethodOp (full #include lines) for MethodTablePlugin.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from ..base import MethodOp, method_op

ARRAY_HPP = "linalg/array.hpp"
SHAPE_HPP = "linalg/shape.hpp"
SLICE_HPP = "linalg/slice.hpp"

ARRAY_TYPE_NAMES: frozenset[str] = frozenset(
    {"ArrayF32", "ArrayF64", "ArrayI32", "ArrayBool"}
)
NUMERIC_ARRAY_NAMES: frozenset[str] = frozenset(
    {"ArrayF32", "ArrayF64", "ArrayI32"}
)


def _inc(hpp: str | None) -> tuple[str, ...]:
    if not hpp:
        return ()
    return (f'#include "{hpp}"\n',)


def _method(
    name: str,
    min_arity: int = 0,
    max_arity: int | None = None,
    hpp: str | None = ARRAY_HPP,
) -> MethodOp:
    if max_arity is None:
        max_arity = min_arity
    base = method_op(name, min_arity, max_arity, includes=_inc(hpp))
    return base


def _emit_count(recv: str, args: list[str]) -> str:
    return f"({recv}).count_nonzero()"


def _emit_any(recv: str, args: list[str]) -> str:
    return f"(({recv}).count_nonzero() != 0)"


def _emit_all(recv: str, args: list[str]) -> str:
    return f"(({recv}).count_nonzero() == ({recv}).shape().numel())"


ARRAY_METHODS: dict[str, MethodOp] = {
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
    "count": MethodOp(
        emit=_emit_count, min_arity=0, max_arity=0, includes=_inc(ARRAY_HPP)
    ),
    "any": MethodOp(
        emit=_emit_any, min_arity=0, max_arity=0, includes=_inc(ARRAY_HPP)
    ),
    "all": MethodOp(
        emit=_emit_all, min_arity=0, max_arity=0, includes=_inc(ARRAY_HPP)
    ),
}

ARRAY_NUMERIC_METHODS: dict[str, MethodOp] = {
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

SHAPE_METHODS: dict[str, MethodOp] = {
    "ndim": _method("ndim", 0, hpp=SHAPE_HPP),
    "numel": _method("numel", 0, hpp=SHAPE_HPP),
    "strides": _method("strides", 0, hpp=SHAPE_HPP),
}

# Python property on Array -> C++ method call
AttrEmit = Callable[[str], str]


class LinalgAttr(NamedTuple):
    emit: AttrEmit
    includes: tuple[str, ...] = _inc(ARRAY_HPP)


ARRAY_PROPS: dict[str, LinalgAttr] = {
    "shape": LinalgAttr(emit=lambda recv: f"({recv}).shape()"),
    "strides": LinalgAttr(emit=lambda recv: f"({recv}).strides()"),
    "ndim": LinalgAttr(emit=lambda recv: f"({recv}).ndim()"),
    "numel": LinalgAttr(emit=lambda recv: f"({recv}).shape().numel()"),
    "offset": LinalgAttr(emit=lambda recv: f"({recv}).offset()"),
}


class LinalgCtor(NamedTuple):
    cpp_type: str
    min_arity: int
    max_arity: int
    cpp_include: str
    extra_includes: tuple[str, ...] = ()


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

LINALG_METHODS: dict[str, dict[str, MethodOp]] = {
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
