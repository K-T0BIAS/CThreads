"""Unit tests for GpuSignature preamble emission."""

from __future__ import annotations

import pytest

from cthreads.compiler.translation.Source import Source
from cthreads.gpu.compiler.translation.Signature import GpuSignature
from cthreads.gpu.compiler.translation.context import GpuTranslationContext
from cthreads.gpu.gpu_kernel_meta import build_gpu_kernel_meta
from cthreads.types import PyBool, PyFloat, PyInt, PyList


def _sig(fn, local_size_x: int = 64):
    ctx = GpuTranslationContext(fn=fn, local_size_x=local_size_x)
    result = GpuSignature.translate(Source.parse_function(fn), ctx)
    return ctx, result


def test_saxpy_preamble_matches_meta_and_smoke_shape():
    def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
        pass

    meta = build_gpu_kernel_meta(saxpy)
    ctx, sig = _sig(saxpy)
    assert sig.binding_count == meta.binding_count == 3
    assert sig.scalar_bytes == meta.scalar_bytes == 8
    assert sig.scalar_fields == [("n", "int"), ("a", "float")]
    assert sig.list_fields == [(1, "x", "float"), (2, "y", "float")]
    assert "layout(local_size_x = 64)" in sig.preamble
    assert "binding = 0" in sig.preamble
    assert "int n;" in sig.preamble and "float a;" in sig.preamble
    assert "binding = 1" in sig.preamble and "float data[];" in sig.preamble
    assert "} x;" in sig.preamble and "} y;" in sig.preamble
    assert "n" in ctx.scalar_params and "a" in ctx.scalar_params
    assert "x" in ctx.list_params and "y" in ctx.list_params
    assert isinstance(ctx.symbols["n"], PyInt)
    assert isinstance(ctx.symbols["x"], PyList)


def test_lists_only_no_binding_zero_scalars():
    def k(x: list[int], y: list[int]) -> None:
        pass

    _, sig = _sig(k)
    assert sig.binding_count == 2
    assert sig.scalar_bytes == 0
    assert sig.scalar_fields == []
    assert "binding = 0" in sig.preamble and "binding = 1" in sig.preamble
    assert "buffer Scalars" not in sig.preamble
    assert sig.list_fields == [(0, "x", "int"), (1, "y", "int")]


def test_scalars_only_binding_zero():
    def k(a: int, b: float, c: bool) -> None:
        pass

    _, sig = _sig(k)
    assert sig.binding_count == 1
    assert sig.scalar_bytes == 12
    assert "bool c;" in sig.preamble
    assert "data[]" not in sig.preamble


@pytest.mark.parametrize("local", [1, 8, 32, 64, 128, 256])
def test_custom_local_size(local):
    def k(n: int) -> None:
        pass

    _, sig = _sig(k, local_size_x=local)
    assert f"layout(local_size_x = {local})" in sig.preamble
    assert sig.local_size_x == local


def test_param_order_preserved_in_scalar_block():
    def k(b: float, a: int) -> None:
        pass

    _, sig = _sig(k)
    assert sig.scalar_fields == [("b", "float"), ("a", "int")]
    pos_b = sig.preamble.index("float b;")
    pos_a = sig.preamble.index("int a;")
    assert pos_b < pos_a


def test_list_binding_order_follows_params():
    def k(n: int, z: list[float], a: list[int], b: list[bool]) -> None:
        pass

    _, sig = _sig(k)
    assert sig.list_fields == [
        (1, "z", "float"),
        (2, "a", "int"),
        (3, "b", "bool"),
    ]
    assert sig.binding_count == 4


def test_missing_annotation_raises():
    def k(n: int, a) -> None:  # type: ignore[no-untyped-def]
        pass

    with pytest.raises(TypeError, match="type annotation"):
        _sig(k)


def test_vararg_rejected():
    def k(*args: int) -> None:
        pass

    with pytest.raises(TypeError, match=r"\*args"):
        _sig(k)


def test_kwargs_rejected():
    def k(**kwargs: int) -> None:
        pass

    with pytest.raises(TypeError, match=r"\*\*kwargs"):
        _sig(k)


def test_kwonly_rejected():
    def k(*, n: int) -> None:
        pass

    with pytest.raises(TypeError, match="kw-only"):
        _sig(k)


def test_non_none_return_rejected():
    def k(n: int) -> int:
        return n

    with pytest.raises(TypeError, match="return must be None"):
        _sig(k)


def test_none_return_ok():
    def k(n: int) -> None:
        pass

    _, sig = _sig(k)
    assert sig.func_name == "k"


def test_unsupported_dict_param():
    def k(d: dict[str, int]) -> None:
        pass

    with pytest.raises(TypeError):
        _sig(k)


def test_unsupported_str_param():
    def k(s: str) -> None:
        pass

    with pytest.raises(TypeError):
        _sig(k)


def test_unsupported_nested_list():
    def k(x: list[list[int]]) -> None:
        pass

    with pytest.raises(TypeError):
        _sig(k)


def test_list_block_name_capitalized():
    def k(values: list[float]) -> None:
        pass

    _, sig = _sig(k)
    assert "buffer Values" in sig.preamble
    assert "} values;" in sig.preamble


def test_single_list_binding_zero():
    def k(xs: list[int]) -> None:
        pass

    _, sig = _sig(k)
    assert sig.binding_count == 1
    assert "binding = 0" in sig.preamble
    assert "buffer Scalars" not in sig.preamble


def test_bool_list_elem():
    def k(flags: list[bool]) -> None:
        pass

    _, sig = _sig(k)
    assert "bool data[];" in sig.preamble


def test_ctx_symbols_typed():
    def k(n: int, flag: bool, a: float, xs: list[int]) -> None:
        pass

    ctx, _ = _sig(k)
    assert isinstance(ctx.symbols["n"], PyInt)
    assert isinstance(ctx.symbols["flag"], PyBool)
    assert isinstance(ctx.symbols["a"], PyFloat)
    assert isinstance(ctx.symbols["xs"], PyList)


def test_std430_mentioned_on_buffers():
    def k(n: int, x: list[float]) -> None:
        pass

    _, sig = _sig(k)
    assert sig.preamble.count("std430") >= 2
