"""
CUDA-style index builtins for `@Gpu` kernels.

Frontend markers only. Codegen lowers `GlobalIdx.x` (and friends) via the
GPU AttrPlugin registry to GLSL invocation IDs.

#### Mapping:
- ThreadIdx -> gl_LocalInvocationID
- BlockIdx -> gl_WorkGroupID
- BlockDim -> gl_WorkGroupSize
- GridDim -> gl_NumWorkGroups
- GlobalIdx -> gl_GlobalInvocationID
"""

from __future__ import annotations


class _Axis:
    """
    Placeholder for one axis (`.x` / `.y` / `.z`) of an index builtin.
    """

    __slots__ = ("_label",)

    def __init__(self, label: str) -> None:
        self._label: str = label

    def __repr__(self) -> str:
        return f"<gpu builtin {self._label}>"


class GpuIndexBuiltin:
    """
    Base for ThreadIdx / BlockIdx / GlobalIdx marker classes.

    #### Technical terms:
    - GLSL: shading language used for Vulkan compute shaders
    """

    # Subclasses set the GLSL built-in vector name (without .x/.y/.z).
    _glsl_base: str = ""


class ThreadIdx(GpuIndexBuiltin):
    """
    Local invocation index within a workgroup (`gl_LocalInvocationID`).
    """

    _glsl_base: str = "gl_LocalInvocationID"
    x: _Axis = _Axis("ThreadIdx.x")
    y: _Axis = _Axis("ThreadIdx.y")
    z: _Axis = _Axis("ThreadIdx.z")


class BlockIdx(GpuIndexBuiltin):
    """
    Workgroup index (`gl_WorkGroupID`).
    """

    _glsl_base: str = "gl_WorkGroupID"
    x: _Axis = _Axis("BlockIdx.x")
    y: _Axis = _Axis("BlockIdx.y")
    z: _Axis = _Axis("BlockIdx.z")


class BlockDim(GpuIndexBuiltin):
    """
    Workgroup size (`gl_WorkGroupSize`).
    """

    _glsl_base: str = "gl_WorkGroupSize"
    x: _Axis = _Axis("BlockDim.x")
    y: _Axis = _Axis("BlockDim.y")
    z: _Axis = _Axis("BlockDim.z")


class GridDim(GpuIndexBuiltin):
    """
    Number of workgroups (`gl_NumWorkGroups`).
    """

    _glsl_base: str = "gl_NumWorkGroups"
    x: _Axis = _Axis("GridDim.x")
    y: _Axis = _Axis("GridDim.y")
    z: _Axis = _Axis("GridDim.z")


class GlobalIdx(GpuIndexBuiltin):
    """
    Global invocation index (`gl_GlobalInvocationID`).

    Prefer this for 1D element-wise kernels (for example saxpy).

    #### Example:
    ``py
    from cthreads.gpu import GlobalIdx, Gpu

    @Gpu
    def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = a * x[i] + y[i]
    ``
    """

    _glsl_base: str = "gl_GlobalInvocationID"
    x: _Axis = _Axis("GlobalIdx.x")
    y: _Axis = _Axis("GlobalIdx.y")
    z: _Axis = _Axis("GlobalIdx.z")
