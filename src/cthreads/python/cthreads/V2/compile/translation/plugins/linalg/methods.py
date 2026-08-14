"""`a.matmul(b)` / `shape.numel()` on typed Array / Shape receivers."""

from __future__ import annotations

from .....types import PyCThreadsInternalType, PyType
from ..base import MethodTablePlugin
from .ops import LINALG_METHODS


class LinalgMethodPlugin(MethodTablePlugin):
    tables = LINALG_METHODS

    def type_key(self, py_type: PyType) -> str | None:
        if isinstance(py_type, PyCThreadsInternalType) and py_type.name in self.tables:
            return py_type.name
        return None
