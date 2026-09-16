"""Unit tests for GPU index frontend markers."""

from __future__ import annotations

import pytest

from cthreads.gpu import BlockDim, BlockIdx, GlobalIdx, GridDim, ThreadIdx
from cthreads.gpu.frontend.indexes import GpuIndexBuiltin, _Axis


@pytest.mark.parametrize(
    "cls, base",
    [
        (ThreadIdx, "gl_LocalInvocationID"),
        (BlockIdx, "gl_WorkGroupID"),
        (BlockDim, "gl_WorkGroupSize"),
        (GridDim, "gl_NumWorkGroups"),
        (GlobalIdx, "gl_GlobalInvocationID"),
    ],
)
def test_index_classes_glsl_base(cls, base):
    assert issubclass(cls, GpuIndexBuiltin)
    assert cls._glsl_base == base
    assert isinstance(cls.x, _Axis)
    assert isinstance(cls.y, _Axis)
    assert isinstance(cls.z, _Axis)


@pytest.mark.parametrize(
    "cls",
    [ThreadIdx, BlockIdx, BlockDim, GridDim, GlobalIdx],
)
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_axis_repr(cls, axis):
    obj = getattr(cls, axis)
    assert isinstance(obj, _Axis)
    assert f"{cls.__name__}.{axis}" in repr(obj)


def test_indexes_reexported_from_package():
    import cthreads.gpu as g

    assert g.GlobalIdx is GlobalIdx
    assert g.ThreadIdx is ThreadIdx
    assert g.BlockIdx is BlockIdx
    assert g.BlockDim is BlockDim
    assert g.GridDim is GridDim
