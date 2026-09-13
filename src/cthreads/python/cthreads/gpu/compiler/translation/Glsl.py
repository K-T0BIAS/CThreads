"""
Map cthreads PyTypes to GLSL type names and std430 sizes for GPU Signature.

Python float -> GLSL float (f32). CPU @Thread uses double; GPU v1 does not.
Lists are not a GLSL type string — use type_name on the element type.
"""

from __future__ import annotations

from ....types import (
    PyBool,
    PyDict,
    PyFloat,
    PyInt,
    PyList,
    PyString,
    PyThreadable,
    PyType,
    is_shared_pytype,
    is_sync_pytype,
    is_tbuffer_pytype,
)

# GLSL / host pack sizes (must match gpu_kernel_meta._GPU_SCALAR_BYTES).
_SCALAR: dict[type, tuple[str, int]] = {
    PyInt: ("int", 4),
    PyFloat: ("float", 4),
    PyBool: ("bool", 4),
}


def _scalar_entry(py_type: PyType) -> tuple[str, int]:
    for cls, entry in _SCALAR.items():
        if isinstance(py_type, cls):
            return entry
    raise TypeError(
        f"GPU GLSL: unsupported scalar type {py_type.name!r} "
        f"(allowed: int, float, bool)"
    )


def type_name(py_type: PyType) -> str:
    """
    GLSL type name for a scalar PyType (`int` / `float` / `bool`).

    For lists, pass `py_type.inner_type` (or use `elem_type_name`).
    """
    if isinstance(py_type, PyList):
        raise TypeError(
            "GPU GLSL: list is not a scalar type name; "
            "use elem_type_name(py_type) or type_name(py_type.inner_type)"
        )
    if (
        is_sync_pytype(py_type)
        or is_shared_pytype(py_type)
        or is_tbuffer_pytype(py_type)
        or isinstance(py_type, (PyDict, PyString, PyThreadable))
    ):
        raise TypeError(
            f"GPU GLSL: {py_type.name!r} is not supported on the GPU path"
        )
    return _scalar_entry(py_type)[0]


def elem_type_name(py_type: PyList) -> str:
    """GLSL element type for a `list[...]` PyType (e.g. `float` for list[float])."""
    if not isinstance(py_type, PyList):
        raise TypeError(
            f"GPU GLSL: elem_type_name expects PyList, got {type(py_type)!r}"
        )
    return type_name(py_type.inner_type)


def size_bytes(py_type: PyType) -> int:
    """std430 size in bytes for a scalar (lists: use size_bytes on the element)."""
    if isinstance(py_type, PyList):
        raise TypeError(
            "GPU GLSL: list has no single scalar size; "
            "use size_bytes(py_type.inner_type) for the element"
        )
    return _scalar_entry(py_type)[1]


def align_bytes(py_type: PyType) -> int:
    """std430 alignment for a v1 scalar (same as size for int/float/bool)."""
    return size_bytes(py_type)
