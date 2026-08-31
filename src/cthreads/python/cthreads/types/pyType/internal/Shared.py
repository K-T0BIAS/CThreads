from typing import Any

from ..pyType import PyType


class Shared:
    """
    Annotation helper for cooperative shared kernel params / returns.

    Use `Shared[list[int]]` (etc.) in @Thread signatures. Native storage
    lives on a `SharedHost` (cpp/headers/shared_host.hpp); 
    Python can see the c++ semem (shared memory) state via `job.sync_state()`, `__sync_state__()`
    or `job.result()` if the threaded fn returns the shared memory variable (not live like `TBuffer`).
    """

    __cthreads_shared__ = True # marker to identify shared types

    def __class_getitem__(cls, inner: Any) -> type:
        name = getattr(inner, "__name__", repr(inner))
        stub = type(f"Shared[{name}]", (), {})
        stub.__cthreads_shared__ = True
        stub.__cthreads_shared_inner__ = inner
        return stub


class PyShared(PyType):
    """
    Maps `Shared[inner]` to a pytype for translation into c++.
    
    """

    inner_type: PyType

    def __init__(self, inner_type: PyType) -> None:
        self.inner_type = inner_type
        super().__init__(
            name="Shared",
            cpp_name=inner_type.cpp_name,
            description=f"shared<{inner_type.description}>",
            cpp_include="shared_host.hpp",
            needs_include=True,
        )

    def build_include(self) -> str:
        # Inner headers (vector, Threadable.hpp, …) come from include_for(..., this_file).
        return super().build_include()


def is_shared_pytype(py_type: PyType) -> bool:
    """
    Checks if a pytype is a shared pytype.
    """
    return isinstance(py_type, PyShared)


def peel_shared(py_type: PyType) -> PyType:
    """Unwrap `Shared[inner]` for type-directed lowering (one level)."""
    if isinstance(py_type, PyShared):
        return py_type.inner_type
    return py_type
