from typing import Any

from ..pyType import PyType
from .include_map import SYNC_INTERNAL_NAMES
from .pyCThreadsInternalType import PyCThreadsInternalType

class TBuffer:
    """
    Annotation helper for triple-buffer kernel params.

    Use ``TBuffer[Particle]`` in @Thread signatures; codegen emits
    ``cthreads::sync::tripple_buffer<Particle>``.
    """

    __cthreads_tbuffer__ = True

    def __class_getitem__(cls, inner: Any) -> type:
        name = getattr(inner, "__name__", repr(inner))
        stub = type(f"TBuffer[{name}]", (), {})
        stub.__cthreads_tbuffer__ = True
        stub.__cthreads_tbuffer_inner__ = inner
        return stub


class PyTBuffer(PyType):
    """Maps ``TBuffer[inner]`` to ``cthreads::sync::tripple_buffer<inner>``."""

    inner_type: PyType

    def __init__(self, inner_type: PyType) -> None:
        self.inner_type = inner_type
        cpp_inner = inner_type.cpp_name
        super().__init__(
            name="TBuffer",
            cpp_name=f"cthreads::sync::tripple_buffer<{cpp_inner}>",
            description=f"tripple_buffer<{inner_type.description}>",
            cpp_include="sync/t_buffer.hpp",
            needs_include=True,
        )

    def build_include(self) -> str:
        return super().build_include() + self.inner_type.build_include()


TBUFFER_INTERNAL_NAMES: frozenset[str] = frozenset(
    {
        "TBufferF64",
        "TBufferI64",
        "TBufferBool",
        "TBufferStr",
        "TBufferListF64",
        "TBufferDictStrF64",
        "TBufferObj",
    }
)


def is_tbuffer_pytype(py_type: PyType) -> bool:
    """True for ``TBuffer[...]`` and fixed ``cthreads.sync.TBuffer*`` types."""
    return isinstance(py_type, PyTBuffer) or (
        isinstance(py_type, PyCThreadsInternalType)
        and py_type.name in TBUFFER_INTERNAL_NAMES
    )


def is_sync_pytype(py_type: PyType) -> bool:
    """True for ``Lock`` / ``Event`` / ``RWLock`` kernel params (non-copyable)."""
    return (
        isinstance(py_type, PyCThreadsInternalType)
        and py_type.name in SYNC_INTERNAL_NAMES
    )