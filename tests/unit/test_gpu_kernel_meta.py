"""Unit tests for build_gpu_kernel_meta and schemas."""

from __future__ import annotations

import pytest

from cthreads.gpu.gpu_kernel_meta import (
    GPU_KERNELS,
    GpuParamMeta,
    GpuTypeSchema,
    build_gpu_kernel_meta,
    pytype_to_gpu_schema,
)
from cthreads.types import PyBool, PyDict, PyFloat, PyInt, PyList, PyString


@pytest.fixture(autouse=True)
def _clear_kernels():
    GPU_KERNELS.clear()
    yield
    GPU_KERNELS.clear()


def test_saxpy_meta_shape():
    def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
        pass

    meta = build_gpu_kernel_meta(saxpy)
    assert meta.symbol == "saxpy"
    assert meta.binding_count == 3
    assert meta.scalar_bytes == 8
    assert meta.local_size_x == 64
    assert [p.name for p in meta.params] == ["n", "a", "x", "y"]
    assert meta.params[0].pass_as == "value"
    assert meta.params[2].pass_as == "ref"
    assert meta.params[2].elem_kind == "float"
    assert meta.params[2].elem_bytes == 4
    d = meta.to_dict()
    assert d["symbol"] == "saxpy"
    assert saxpy.__gpu_kernel_meta__["binding_count"] == 3  # type: ignore[attr-defined]
    assert GPU_KERNELS["saxpy"] is meta


def test_custom_symbol_and_local_size():
    def k(n: int) -> None:
        pass

    meta = build_gpu_kernel_meta(k, symbol="my_k", local_size_x=32)
    assert meta.symbol == "my_k"
    assert meta.local_size_x == 32
    assert "my_k" in GPU_KERNELS


@pytest.mark.parametrize(
    "fn_src, binding, scalar",
    [
        ("def k(a: int) -> None: pass", 1, 4),
        ("def k(a: float, b: bool) -> None: pass", 1, 8),
        ("def k(x: list[int]) -> None: pass", 1, 0),
        ("def k(x: list[float], y: list[bool]) -> None: pass", 2, 0),
        ("def k(n: int, x: list[int]) -> None: pass", 2, 4),
    ],
)
def test_binding_and_scalar_bytes(fn_src, binding, scalar):
    ns: dict = {}
    exec(fn_src, ns)
    meta = build_gpu_kernel_meta(ns["k"])
    assert meta.binding_count == binding
    assert meta.scalar_bytes == scalar


def test_rejects_return_int():
    def k(n: int) -> int:
        return n

    with pytest.raises(TypeError, match="return must be None"):
        build_gpu_kernel_meta(k)


def test_rejects_missing_annotation():
    def k(n: int, a) -> None:  # type: ignore[no-untyped-def]
        pass

    with pytest.raises(TypeError, match="type annotation"):
        build_gpu_kernel_meta(k)


def test_rejects_varargs():
    def k(*args: int) -> None:
        pass

    with pytest.raises(TypeError, match=r"\*args"):
        build_gpu_kernel_meta(k)


def test_rejects_kwargs():
    def k(**kwargs: int) -> None:
        pass

    with pytest.raises(TypeError, match=r"\*\*kwargs"):
        build_gpu_kernel_meta(k)


def test_rejects_kwonly():
    def k(*, n: int) -> None:
        pass

    with pytest.raises(TypeError, match="keyword-only"):
        build_gpu_kernel_meta(k)


def test_rejects_str_param():
    def k(x: str) -> None:
        pass

    with pytest.raises(TypeError):
        build_gpu_kernel_meta(k)


def test_rejects_dict_param():
    def k(x: dict[str, int]) -> None:
        pass

    with pytest.raises(TypeError):
        build_gpu_kernel_meta(k)


def test_rejects_list_str():
    def k(x: list[str]) -> None:
        pass

    with pytest.raises(TypeError):
        build_gpu_kernel_meta(k)


def test_rejects_nested_list():
    def k(x: list[list[int]]) -> None:
        pass

    with pytest.raises(TypeError):
        build_gpu_kernel_meta(k)


def test_pytype_to_gpu_schema_scalars():
    assert pytype_to_gpu_schema(PyInt()).kind == "int"
    assert pytype_to_gpu_schema(PyFloat()).spirv_type == "float"
    assert pytype_to_gpu_schema(PyBool()).elem_bytes == 4


def test_pytype_to_gpu_schema_list():
    s = pytype_to_gpu_schema(PyList(PyFloat()))
    assert s.kind == "list"
    assert s.spirv_type == "float[]"
    assert s.inner is not None
    assert s.inner.kind == "float"


def test_pytype_rejects_dict_string():
    with pytest.raises(TypeError):
        pytype_to_gpu_schema(PyDict(PyString(), PyInt()))
    with pytest.raises(TypeError):
        pytype_to_gpu_schema(PyString())


def test_param_meta_pass_as_invalid():
    with pytest.raises(TypeError, match="pass_as"):
        GpuParamMeta(
            name="x",
            pass_as="inout",  # type: ignore[arg-type]
            schema=GpuTypeSchema(kind="int", spirv_type="int", elem_bytes=4),
        )


def test_list_schema_to_dict_requires_inner():
    bad = GpuTypeSchema(kind="list", spirv_type="int[]", elem_bytes=4, inner=None)
    with pytest.raises(ValueError, match="inner"):
        bad.to_dict()


def test_param_to_dict_list_flattens_elem():
    p = GpuParamMeta(
        name="xs",
        pass_as="ref",
        schema=pytype_to_gpu_schema(PyList(PyInt())),
    )
    d = p.to_dict()
    assert d["kind"] == "list"
    assert d["elem_kind"] == "int"
    assert d["elem_bytes"] == 4
    assert "schema" in d
