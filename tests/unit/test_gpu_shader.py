"""
GPU shader / launch smoke (test-only _ext.gpu.testing).

Covers create_entry, ShaderCache, update_descriptors, and launch_gpu_kernel
via smoke_launch_saxpy (join writeback). Live checks skip when CTHREADS_GPU
is off or no Vulkan device is available.
"""

from __future__ import annotations

import pytest

from cthreads import gpu
from cthreads.gpu.errors import GpuInvalidArgument


def _ext_gpu():
    return gpu._gpu


def _require_gpu_testing():
    ext = _ext_gpu()
    if ext is None:
        pytest.skip("cthreads built without CTHREADS_GPU (_ext.gpu missing)")
    if not hasattr(ext, "testing"):
        pytest.skip("_ext.gpu.testing missing (rebuild with CTHREADS_GPU=ON)")
    if not gpu.available():
        pytest.skip("Vulkan loader/device not available in this environment")
    return ext.testing


def _map_probe(exc: BaseException):
    return gpu._map_error(exc)


def test_public_gpu_has_no_shader_smoke_exports():
    assert not hasattr(gpu, "smoke_create_entry")
    assert not hasattr(gpu, "smoke_update_descriptors")
    assert not hasattr(gpu, "smoke_launch_saxpy")
    assert not hasattr(gpu, "testing")


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


def test_live_smoke_launch_saxpy():
    testing = _require_gpu_testing()
    if not hasattr(testing, "smoke_launch_saxpy"):
        pytest.skip("rebuild with latest gpu testing (smoke_launch_saxpy)")
    try:
        testing.smoke_launch_saxpy()
    finally:
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
