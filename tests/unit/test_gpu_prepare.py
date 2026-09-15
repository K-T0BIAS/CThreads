"""Unit / live tests for prepare() and gpu()."""

from __future__ import annotations

import pytest

from helpers_gpu import prepare_module

from cthreads.frontend.Registry import REGISTRY
from cthreads.gpu import GlobalIdx, Gpu, available, gpu, prepare, shutdown
from cthreads.gpu.frontend.errors import GPUNotAvailable
from cthreads.gpu.runtime import GpuJob


@pytest.fixture(autouse=True)
def _reset_gpu_prepare_state():
    prepare_mod = prepare_module()
    REGISTRY.clear()
    prepare_mod._gpu_prepared = False
    yield
    REGISTRY.clear()
    # Isolation: shutdown releases ShaderCache; lib clears `_gpu_prepared`.
    try:
        shutdown()
    except Exception:
        pass
    prepare_mod._gpu_prepared = False


def test_prepare_function_and_runtime_module_coexist():
    """Public API is the function; internals live on `cthreads.gpu.runtime`."""
    import cthreads.gpu.runtime as runtime_mod
    from cthreads.gpu import prepare as prepare_fn

    assert callable(prepare_fn)
    assert prepare_fn is runtime_mod.prepare
    assert hasattr(runtime_mod, "_gpu_prepared")
    assert prepare_module() is runtime_mod


def test_gpu_rejects_non_gpu_fn():
    def plain(n: int) -> None:
        pass

    with pytest.raises(TypeError, match="@Gpu"):
        gpu(plain, 1)


def test_gpu_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        gpu(None)  # type: ignore[arg-type]


def test_gpu_rejects_kwargs():
    if not available():
        pytest.skip("GPU not available")

    @Gpu
    def k(n: int) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return

    with pytest.raises(TypeError, match="keyword"):
        gpu(k, 1, force=False, extra=1)


def test_gpu_rejects_arity_mismatch():
    if not available():
        pytest.skip("GPU not available")

    @Gpu
    def k(n: int, a: float) -> None:
        pass

    with pytest.raises(TypeError, match="expected 2"):
        gpu(k, 1)


def test_prepare_raises_when_nothing_registered():
    if not available():
        pytest.skip("GPU not available")
    with pytest.raises(RuntimeError, match="Nothing registered"):
        prepare()


def test_prepare_unavailable(monkeypatch):
    prepare_mod = prepare_module()
    monkeypatch.setattr(prepare_mod._ext_gpu_api, "available", lambda: False)
    with pytest.raises(GPUNotAvailable):
        prepare()


def test_gpu_unavailable(monkeypatch):
    prepare_mod = prepare_module()
    monkeypatch.setattr(prepare_mod._ext_gpu_api, "available", lambda: False)

    def fake_gpu_fn(n: int) -> None:
        pass

    fake_gpu_fn.__gpu__ = True  # type: ignore[attr-defined]
    with pytest.raises(GPUNotAvailable):
        gpu(fake_gpu_fn, 1)


def test_gpu_job_result_is_none():
    job = GpuJob(
        type(
            "R",
            (),
            {
                "start": lambda self: None,
                "done": lambda self: True,
                "join": lambda self: None,
            },
        )()
    )
    assert job.result() is None


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_live_prepare_and_gpu_saxpy():
    prepare_mod = prepare_module()

    @Gpu
    def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = a * x[i] + y[i]

    x = [1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 30.0, 40.0]
    a = 2.0
    expect = [a * xi + yi for xi, yi in zip(x, y)]

    info = prepare()
    assert "rewritten" in info
    assert prepare_mod._gpu_prepared is True

    job = gpu(saxpy, len(x), a, x, y)
    assert isinstance(job, GpuJob)
    job.join()
    assert y == expect
    assert job.done()
    assert job.result() is None


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_live_gpu_auto_prepare():
    @Gpu
    def fill(n: int, ys: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        ys[i] = 1.0

    ys = [0.0, 0.0, 0.0, 0.0]
    prepare_module()._gpu_prepared = False
    gpu(fill, 4, ys).join()
    assert ys == [1.0, 1.0, 1.0, 1.0]


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_lib_reserved_param_name_out():
    """Live path: reserved `out` fails at GLSL compile with a clear hint."""

    @Gpu
    def fill(n: int, out: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        out[i] = 1.0

    ys = [0.0, 0.0]
    with pytest.raises(RuntimeError, match="reserved words|GLSL compile failed"):
        gpu(fill, 2, ys).join()


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_live_gpu_second_launch_same_kernel():
    @Gpu
    def add1(n: int, y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = y[i] + 1.0

    y = [0.0, 0.0, 0.0]
    gpu(add1, 3, y).join()
    gpu(add1, 3, y).join()
    assert y == [2.0, 2.0, 2.0]


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_live_larger_than_one_workgroup():
    @Gpu
    def scale(n: int, a: float, x: list[float], y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = a * x[i]

    n = 200
    x = [float(i) for i in range(n)]
    y = [0.0] * n
    gpu(scale, n, 3.0, x, y).join()
    assert y == [3.0 * float(i) for i in range(n)]


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_live_force_reprepare():
    @Gpu
    def k(n: int, y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = 7.0

    y = [0.0, 0.0]
    gpu(k, 2, y).join()
    y2 = [0.0, 0.0]
    gpu(k, 2, y2, force=True).join()
    assert y2 == [7.0, 7.0]


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_compile_sets_prepared_flag():
    from cthreads.gpu import compile

    prepare_mod = prepare_module()

    @Gpu
    def k(n: int, y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = 1.0

    prepare_mod._gpu_prepared = False
    info = compile()
    assert prepare_mod._gpu_prepared is True
    assert "rewritten" in info


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_shutdown_then_gpu_reregisters():
    """
    shutdown() releases ShaderCache; next gpu() must rewalk registry and work.
    """
    prepare_mod = prepare_module()

    @Gpu
    def fill(n: int, y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = 3.0

    y = [0.0, 0.0]
    gpu(fill, 2, y).join()
    assert y == [3.0, 3.0]

    shutdown()
    assert prepare_mod._gpu_prepared is False

    y2 = [0.0, 0.0]
    gpu(fill, 2, y2).join()
    assert y2 == [3.0, 3.0]
    assert prepare_mod._gpu_prepared is True
