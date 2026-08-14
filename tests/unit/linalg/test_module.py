"""Submodule surface: types, flags, re-export."""

from __future__ import annotations

import cthreads
from cthreads import linalg


def test_exported_on_cthreads():
    assert cthreads.linalg is linalg
    assert linalg is not None


def test_marked_internal():
    assert getattr(linalg, "__cthreads_internal__", False) is True
    for name in ("Shape", "Slice", "ArrayBool", "ArrayF32", "ArrayF64", "ArrayI32"):
        cls = getattr(linalg, name)
        assert getattr(cls, "__cthreads_internal__", False) is True, name


def test_required_types_exist():
    for name in ("Shape", "Slice", "ArrayBool", "ArrayF32", "ArrayF64", "ArrayI32"):
        assert hasattr(linalg, name)


def test_array_required_methods():
    required = (
        "from_list",
        "to_list",
        "is_contiguous",
        "view",
        "reshape",
        "flatten",
        "transpose",
        "permute",
        "squeeze",
        "unsqueeze",
        "contiguous",
        "_contiguous",
        "_view",
        "_reshape",
        "_flatten",
        "_transpose",
        "_permute",
        "_squeeze",
        "_unsqueeze",
        "matmul",
        "dot",
        "cross",
        "_add",
        "_sub",
        "_mul",
        "_div",
        "_neg",
        "_matmul",
        "_dot",
        "_cross",
    )
    for cls in (linalg.ArrayF32, linalg.ArrayF64, linalg.ArrayI32):
        for name in required:
            assert hasattr(cls, name), f"{cls.__name__}.{name}"


def test_array_required_properties():
    a = linalg.ArrayF32([2, 3])
    for name in ("shape", "strides", "ndim", "numel", "offset"):
        assert hasattr(a, name)
