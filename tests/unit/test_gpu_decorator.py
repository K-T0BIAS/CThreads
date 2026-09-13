"""@Gpu decorator registration (no launch / emit)."""

from __future__ import annotations

import pytest

from cthreads.frontend.Registry import REGISTRY
from cthreads.gpu.frontend import Gpu
from cthreads.gpu.frontend.errors import GPUNotAvailable
from cthreads.gpu import frontend as gpu_frontend


@pytest.fixture(autouse=True)
def _clear_registry():
    REGISTRY.clear()
    yield
    REGISTRY.clear()


def test_gpu_decorator_requires_available(monkeypatch):
    monkeypatch.setattr(gpu_frontend.wrapper, "available", lambda: False)

    with pytest.raises(GPUNotAvailable):

        @Gpu
        def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
            pass


def test_gpu_decorator_registers_and_attaches_meta(monkeypatch):
    monkeypatch.setattr(gpu_frontend.wrapper, "available", lambda: True)

    @Gpu
    def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
        pass

    assert saxpy.__gpu__ is True
    assert saxpy.__gpu_version__ == REGISTRY.VERSION
    assert saxpy.__qualname__ in REGISTRY.gpu_functions
    meta = saxpy.__gpu_kernel_meta__
    assert meta["symbol"] == "saxpy"
    assert meta["binding_count"] == 3
    assert meta["scalar_bytes"] == 8
    # Still a normal Python function.
    assert saxpy(4, 2.0, [1.0], [0.0]) is None


def test_gpu_decorator_log_factory(monkeypatch, capsys):
    monkeypatch.setattr(gpu_frontend.wrapper, "available", lambda: True)
    monkeypatch.setattr(gpu_frontend.wrapper, "device_name", lambda: "FakeGPU")

    @Gpu(log=True)
    def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
        pass

    assert saxpy.__gpu__ is True
    assert "FakeGPU" in capsys.readouterr().out


def test_gpu_decorator_rejects_non_none_return(monkeypatch):
    monkeypatch.setattr(gpu_frontend.wrapper, "available", lambda: True)

    with pytest.raises(TypeError, match="return must be None"):

        @Gpu
        def bad(x: int) -> int:
            return x


def test_gpu_decorator_rejects_dict(monkeypatch):
    monkeypatch.setattr(gpu_frontend.wrapper, "available", lambda: True)

    with pytest.raises(TypeError):

        @Gpu
        def bad(d: dict[str, int]) -> None:
            pass


def test_gpu_decorator_rejects_varargs(monkeypatch):
    monkeypatch.setattr(gpu_frontend.wrapper, "available", lambda: True)

    with pytest.raises(TypeError):

        @Gpu
        def bad(*args: int) -> None:
            pass


def test_gpu_decorator_bool_and_int_list(monkeypatch):
    monkeypatch.setattr(gpu_frontend.wrapper, "available", lambda: True)

    @Gpu
    def k(n: int, flag: bool, xs: list[int], ys: list[bool]) -> None:
        pass

    meta = k.__gpu_kernel_meta__
    assert meta["binding_count"] == 3
    assert meta["scalar_bytes"] == 8
    kinds = [p["kind"] for p in meta["params"]]
    assert kinds == ["int", "bool", "list", "list"]
