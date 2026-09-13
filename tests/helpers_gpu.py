"""
Shared helpers for GPU unit / pipeline tests.
"""

from __future__ import annotations

from types import ModuleType

import cthreads.gpu.runtime as runtime_mod


def prepare_module() -> ModuleType:
    """
    Return `cthreads.gpu.runtime` (holds `_gpu_prepared` / prepare / gpu).
    """
    return runtime_mod
