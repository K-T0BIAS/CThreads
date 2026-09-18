"""
GPU shader / launch tests.

Substrate smokes stay on `_ext.gpu.testing`. Launch + join use the product
path: `_ext.gpu.launch_gpu_kernel` / `GpuJob` (via `_ext_gpu_api`).
"""

from __future__ import annotations

import pytest

from cthreads import gpu
from cthreads.gpu import _ext_gpu_api
from cthreads.gpu.frontend.errors import GpuInvalidArgument
from cthreads.gpu.gpu_kernel_meta import build_gpu_kernel_meta


def _ext_gpu():
    return gpu._ext_gpu_api._gpu


def _require_gpu_testing():
    ext = _ext_gpu()
    if ext is None:
        pytest.skip("cthreads built without CTHREADS_GPU (_ext.gpu missing)")
    if not hasattr(ext, "testing"):
        pytest.skip(
            "_ext.gpu.testing missing (local gpu/testing/ not in tree; "
            "optional gitignored helpers)"
        )
    if not gpu.available():
        pytest.skip("Vulkan loader/device not available in this environment")
    return ext.testing


def _map_probe(exc: BaseException):
    return gpu._map_error(exc)


def test_public_gpu_has_no_shader_smoke_exports():
    assert not hasattr(gpu, "smoke_create_entry")
    assert not hasattr(gpu, "smoke_update_descriptors")
    assert not hasattr(gpu, "smoke_launch_saxpy")
    assert not hasattr(gpu, "register_smoke_saxpy")
    assert not hasattr(gpu, "testing")


def test_ext_gpu_exports_launch_api():
    ext = _ext_gpu()
    if ext is None:
        pytest.skip("cthreads built without CTHREADS_GPU (_ext.gpu missing)")
    assert hasattr(ext, "launch_gpu_kernel")
    assert hasattr(ext, "GpuJob")


def test_live_smoke_create_entry():
    testing = _require_gpu_testing()
    try:
        testing.smoke_create_entry()
    finally:
        gpu.shutdown()


def test_live_smoke_update_descriptors():
    testing = _require_gpu_testing()
    try:
        testing.smoke_update_descriptors()
    finally:
        gpu.shutdown()


def test_live_smoke_cache_register_and_get():
    testing = _require_gpu_testing()
    try:
        testing.smoke_cache_register_and_get()
    finally:
        gpu.shutdown()


def test_live_launch_saxpy_product_path():
    """
    Register smoke SPIR-V (test-only), then launch/join via product bindings.
    """
    testing = _require_gpu_testing()
    if not hasattr(testing, "register_smoke_saxpy"):
        pytest.skip("rebuild with register_smoke_saxpy")
    if not hasattr(_ext_gpu(), "launch_gpu_kernel"):
        pytest.skip("rebuild with product launch_gpu_kernel")

    def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
        pass

    try:
        symbol = testing.register_smoke_saxpy()
        meta = build_gpu_kernel_meta(saxpy, symbol=symbol).to_dict()
        meta["group_count_x"] = 1
        meta["group_count_y"] = 1
        meta["group_count_z"] = 1

        n = 4
        a = 2.0
        x = [1.0, 2.0, 3.0, 4.0]
        y = [10.0, 20.0, 30.0, 40.0]
        expect = [a * xi + yi for xi, yi in zip(x, y)]

        job = _ext_gpu_api.launch_gpu_kernel(meta, [n, a, x, y])
        job.join()
        assert y == expect
        assert job.done()
    finally:
        if hasattr(testing, "clear_shader_cache"):
            try:
                testing.clear_shader_cache()
            except Exception:
                pass
        gpu.shutdown()


def test_live_probe_cache_duplicate_add():
    testing = _require_gpu_testing()
    try:
        with pytest.raises(Exception) as ei:
            testing.probe_cache_duplicate_add()
        mapped = _map_probe(ei.value)
        assert isinstance(mapped, GpuInvalidArgument)
        assert "already exists" in str(mapped) or "GpuInvalidArgument" in str(mapped)
    finally:
        gpu.shutdown()


def test_live_probe_update_empty_list_slot():
    testing = _require_gpu_testing()
    try:
        with pytest.raises(Exception) as ei:
            testing.probe_update_empty_list_slot()
        mapped = _map_probe(ei.value)
        assert isinstance(mapped, GpuInvalidArgument)
    finally:
        gpu.shutdown()


def test_live_probe_create_entry_zero_bindings():
    testing = _require_gpu_testing()
    try:
        with pytest.raises(Exception) as ei:
            testing.probe_create_entry_zero_bindings()
        mapped = _map_probe(ei.value)
        assert isinstance(mapped, GpuInvalidArgument)
    finally:
        gpu.shutdown()
