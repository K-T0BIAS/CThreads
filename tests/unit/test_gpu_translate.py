"""Unit tests for translate_function_for_gpu (no device required)."""

from __future__ import annotations

import pytest

from cthreads.gpu import BlockIdx, GlobalIdx, ThreadIdx
from cthreads.gpu.compiler.translation.translate import translate_function_for_gpu
from cthreads.gpu.gpu_kernel_meta import build_gpu_kernel_meta


def test_translate_saxpy_source_shape():
    def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = a * x[i] + y[i]

    r = translate_function_for_gpu(saxpy)
    meta = build_gpu_kernel_meta(saxpy)
    assert r.func_name == "saxpy"
    assert r.binding_count == meta.binding_count
    assert r.scalar_bytes == meta.scalar_bytes
    assert r.source.startswith("#version 450")
    assert "gl_GlobalInvocationID.x" in r.source
    assert "y.data[" in r.source
    assert r.spirv is None


@pytest.mark.parametrize("local", [1, 32, 64, 128])
def test_translate_custom_local_size(local):
    def k(n: int) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return

    r = translate_function_for_gpu(k, local_size_x=local)
    assert f"layout(local_size_x = {local})" in r.source
    assert r.local_size_x == local


def test_translate_rejects_bad_signature():
    def bad(x: dict[str, int]) -> None:
        pass

    with pytest.raises(TypeError):
        translate_function_for_gpu(bad)


def test_translate_pass_only_body():
    def k(n: int) -> None:
        pass

    r = translate_function_for_gpu(k)
    assert "void main()" in r.source
    assert "binding = 0" in r.source


def test_translate_lists_only():
    def k(x: list[int], y: list[int]) -> None:
        i: int = GlobalIdx.x
        y[i] = x[i]

    r = translate_function_for_gpu(k)
    assert "buffer Scalars" not in r.source
    assert "binding = 0" in r.source
    assert "binding = 1" in r.source
    assert r.scalar_bytes == 0
    assert r.binding_count == 2


def test_translate_for_range_and_while():
    def k(n: int, y: list[int]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        s: int = 0
        for j in range(n):
            s += j
        while s > 0:
            s -= 1
            break
        y[i] = s

    r = translate_function_for_gpu(k)
    assert "for (int j =" in r.source
    assert "while (" in r.source


def test_translate_thread_and_block_idx():
    def k(n: int, y: list[int]) -> None:
        i: int = GlobalIdx.x
        t: int = ThreadIdx.x
        b: int = BlockIdx.x
        if i >= n:
            return
        y[i] = t + b

    r = translate_function_for_gpu(k)
    assert "gl_LocalInvocationID.x" in r.source
    assert "gl_WorkGroupID.x" in r.source


def test_translate_bool_and_or():
    def k(n: int, flag: bool, y: list[int]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        if flag and i > 0 or i < 0:
            y[i] = 1
        else:
            y[i] = 0

    r = translate_function_for_gpu(k)
    assert "&&" in r.source and "||" in r.source


def test_translate_pow_rejected():
    def k(a: float, y: list[float]) -> None:
        i: int = GlobalIdx.x
        y[i] = a ** 2.0

    with pytest.raises(TypeError, match=r"\*\*|pow"):
        translate_function_for_gpu(k)


def test_translate_result_fields():
    def k(n: int) -> None:
        pass

    r = translate_function_for_gpu(k)
    assert r.func_name == "k"
    assert isinstance(r.source, str)
    assert r.local_size_x == 64
    assert r.spirv is None
