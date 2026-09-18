"""Unit tests for GLSL -> SPIR-V (native glslang or glslc fallback)."""

from __future__ import annotations

import pytest

from helpers_gpu import glsl_compiler_available

from cthreads.gpu import GlobalIdx
from cthreads.gpu.compiler.translation.spirv import compile_glsl_to_spirv
from cthreads.gpu.compiler.translation.translate import translate_function_for_gpu

_MIN_COMP = """#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0, std430) buffer Scalars { int n; } scalars;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= scalars.n) { return; }
}
"""


pytestmark = pytest.mark.skipif(
    not glsl_compiler_available(),
    reason="no GLSL compiler (skipped on GitHub Actions / CPU-only builds)",
)


def test_compile_min_comp_magic_and_alignment():
    data = compile_glsl_to_spirv(_MIN_COMP)
    assert len(data) >= 20
    assert len(data) % 4 == 0
    assert data[:4] == b"\x03\x02\x23\x07"


def test_compile_empty_raises():
    with pytest.raises(Exception):
        compile_glsl_to_spirv("")


def test_compile_bad_glsl_raises():
    with pytest.raises(Exception):
        compile_glsl_to_spirv("#version 450\nvoid main() { not_a_type x; }\n")


def test_compile_rejects_uint_to_int_without_cast():
    bad = """#version 450
layout(local_size_x = 1) in;
void main() { int i = gl_GlobalInvocationID.x; }
"""
    with pytest.raises(Exception):
        compile_glsl_to_spirv(bad)


def test_compile_with_cast_ok():
    ok = """#version 450
layout(local_size_x = 1) in;
void main() { int i = int(gl_GlobalInvocationID.x); }
"""
    data = compile_glsl_to_spirv(ok)
    assert data[:4] == b"\x03\x02\x23\x07"


@pytest.mark.parametrize("local", [1, 8, 64, 256])
def test_compile_local_sizes(local):
    src = f"""#version 450
layout(local_size_x = {local}) in;
void main() {{}}
"""
    data = compile_glsl_to_spirv(src)
    assert len(data) % 4 == 0


def test_translate_compile_spirv_flag():
    def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = a * x[i] + y[i]

    r0 = translate_function_for_gpu(saxpy, compile_spirv=False)
    assert r0.spirv is None
    r1 = translate_function_for_gpu(saxpy, compile_spirv=True)
    assert r1.spirv is not None
    assert r1.spirv[:4] == b"\x03\x02\x23\x07"
    assert "#version 450" in r1.source
    assert "void main()" in r1.source


def test_translate_compile_many_kernels():
    def add(n: int, x: list[float], y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = x[i] + y[i]

    def fill(n: int, y: list[int]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = i

    for fn in (add, fill):
        r = translate_function_for_gpu(fn, compile_spirv=True)
        assert r.spirv is not None
        assert r.spirv[:4] == b"\x03\x02\x23\x07"


def test_compile_missing_version_raises():
    with pytest.raises(Exception):
        compile_glsl_to_spirv("void main() {}\n")
