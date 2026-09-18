"""Unit tests for GPU error mapping and types."""

from __future__ import annotations

import pytest

from cthreads.gpu.frontend.errors import (
    CThreadsGPUError,
    GPUNotAvailable,
    GpuInvalidArgument,
    GpuUseAfterDestroy,
    VulkanInitFailed,
    VulkanLoaderNotFound,
    VulkanNoDevice,
    VulkanNotBuiltError,
    VulkanOutOfMemory,
    _map_error,
)


@pytest.mark.parametrize(
    "msg, cls",
    [
        ("VulkanLoaderNotFound: missing", VulkanLoaderNotFound),
        ("VulkanNoDevice: none", VulkanNoDevice),
        ("VulkanOutOfMemory: oom", VulkanOutOfMemory),
        ("GpuUseAfterDestroy: gone", GpuUseAfterDestroy),
        ("GpuInvalidArgument: bad", GpuInvalidArgument),
        ("VulkanNotBuilt: off", VulkanNotBuiltError),
        ("VulkanInitFailed: boom", VulkanInitFailed),
        ("something else entirely", VulkanInitFailed),
    ],
)
def test_map_error_prefixes(msg, cls):
    out = _map_error(RuntimeError(msg))
    assert isinstance(out, cls)
    assert isinstance(out, CThreadsGPUError)
    assert msg in str(out)


def test_error_str_contains_prefix():
    err = GPUNotAvailable("no gpu")
    assert "cthreads gpu error" in str(err)
    assert "no gpu" in str(err)
    assert err.detail == "no gpu"


def test_default_details():
    assert "Vulkan loader not found" in VulkanLoaderNotFound().detail
    assert "No Vulkan compute device" in VulkanNoDevice().detail
    assert "out of memory" in VulkanOutOfMemory().detail.lower()
