"""
Shared helpers for GPU unit / pipeline tests.
"""

from __future__ import annotations

import os
from types import ModuleType

import cthreads.gpu.runtime as runtime_mod


def prepare_module() -> ModuleType:
    """
    Return `cthreads.gpu.runtime` (holds `_gpu_prepared` / prepare / gpu).
    """
    return runtime_mod


def on_github_actions() -> bool:
    """True when running under GitHub Actions CI."""
    return os.environ.get("GITHUB_ACTIONS", "").lower() == "true"


def glsl_compiler_available() -> bool:
    """
    True when in-process compile_glsl or glslc can compile a tiny compute shader.

    Always False on GitHub Actions for now (CI builds without CTHREADS_GPU /
    glslang). Broad except: probe must never fail collection.
    """
    if on_github_actions():
        return False
    try:
        from cthreads.gpu.compiler.translation.spirv import compile_glsl_to_spirv

        compile_glsl_to_spirv(
            "#version 450\nlayout(local_size_x = 1) in;\nvoid main() {}\n"
        )
        return True
    except Exception:
        return False
