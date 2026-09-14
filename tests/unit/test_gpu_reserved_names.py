"""Unit tests for GLSL reserved-identifier / keyword collisions."""

from __future__ import annotations

import pytest

from helpers_gpu import glsl_compiler_available

from cthreads.gpu.compiler.translation.spirv import compile_glsl_to_spirv
from cthreads.gpu.compiler.translation.translate import translate_function_for_gpu

pytestmark = pytest.mark.skipif(
    not glsl_compiler_available(),
    reason="no GLSL compiler (skipped on GitHub Actions / CPU-only builds)",
)


@pytest.mark.parametrize(
    "param",
    [
        "out",
        "uniform",
        "buffer",
        "flat",
        "smooth",
        "shared",
    ],
)
def test_reserved_list_param_names_fail_with_hint(param, tmp_module):
    """No keyword denylist: glslang fails; we wrap with an identifier hint."""
    mod = tmp_module(
        f"""
        from cthreads.gpu import GlobalIdx

        def k(n: int, {param}: list[float]) -> None:
            i: int = GlobalIdx.x
            if i >= n:
                return
            {param}[i] = 1.0
        """,
        name=f"reserved_{param}",
    )
    with pytest.raises(RuntimeError, match="reserved words|GLSL compile failed"):
        translate_function_for_gpu(mod.k, compile_spirv=True)


def test_safe_param_names_compile(tmp_module):
    mod = tmp_module(
        """
        from cthreads.gpu import GlobalIdx

        def k(n: int, ys: list[float]) -> None:
            i: int = GlobalIdx.x
            if i >= n:
                return
            ys[i] = 1.0
        """,
        name="safe_param_names",
    )
    r = translate_function_for_gpu(mod.k, compile_spirv=True)
    assert r.spirv is not None
    assert "} ys;" in r.source


def test_compile_error_includes_hint():
    with pytest.raises(RuntimeError, match="Hint:.*reserved"):
        compile_glsl_to_spirv("#version 450\nvoid main() { not_a_type x; }\n")
