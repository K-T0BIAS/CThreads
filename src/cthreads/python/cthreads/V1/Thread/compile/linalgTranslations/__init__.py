"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

cthreads.linalg lowering helpers for @Thread codegen.
"""

from .is_linalg import (
    is_linalg_op,
    resolve_linalg_attr,
    resolve_linalg_ctor,
    resolve_linalg_method,
)
from .linalgOps import (
    ARRAY_HPP,
    ARRAY_METHODS,
    ARRAY_NUMERIC_METHODS,
    ARRAY_PROPS,
    ARRAY_TYPE_NAMES,
    LINALG_CTORS,
    LINALG_METHODS,
    MASK_AND,
    MASK_NOT,
    MASK_OR,
    MASK_XOR,
    NUMERIC_ARRAY_NAMES,
    SHAPE_HPP,
    SHAPE_METHODS,
    SLICE_HPP,
    LinalgAttr,
    LinalgCtor,
    LinalgOp,
)

__all__ = [
    "ARRAY_HPP",
    "ARRAY_METHODS",
    "ARRAY_NUMERIC_METHODS",
    "ARRAY_PROPS",
    "ARRAY_TYPE_NAMES",
    "LINALG_CTORS",
    "LINALG_METHODS",
    "MASK_AND",
    "MASK_NOT",
    "MASK_OR",
    "MASK_XOR",
    "NUMERIC_ARRAY_NAMES",
    "SHAPE_HPP",
    "SHAPE_METHODS",
    "SLICE_HPP",
    "LinalgAttr",
    "LinalgCtor",
    "LinalgOp",
    "is_linalg_op",
    "resolve_linalg_attr",
    "resolve_linalg_ctor",
    "resolve_linalg_method",
]
