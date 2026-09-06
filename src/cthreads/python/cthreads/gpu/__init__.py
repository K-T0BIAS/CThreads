"""Vulkan GPU runtime probe API (Issue 1).

Soft-imports ``cthreads._ext.gpu`` so CPU-only builds still import cleanly.
"""

from __future__ import annotations

from .errors import (
    CThreadsGPUError,
    GPUNotAvailable,
    GpuInvalidArgument,
    GpuUseAfterDestroy,
    VulkanInitFailed,
    VulkanLoaderNotFound,
    VulkanNoDevice,
    VulkanNotBuiltError,
    VulkanOutOfMemory,
)

try:
    from cthreads._ext import gpu as _gpu
except ImportError:
    _gpu = None


def _map_error(exc: BaseException) -> CThreadsGPUError:
    msg = str(exc)
    if "VulkanLoaderNotFound" in msg:
        return VulkanLoaderNotFound(msg)
    if "VulkanNoDevice" in msg:
        return VulkanNoDevice(msg)
    if "VulkanOutOfMemory" in msg:
        return VulkanOutOfMemory(msg)
    if "GpuUseAfterDestroy" in msg:
        return GpuUseAfterDestroy(msg)
    if "GpuInvalidArgument" in msg:
        return GpuInvalidArgument(msg)
    if "VulkanNotBuilt" in msg:
        return VulkanNotBuiltError(msg)
    if "VulkanInitFailed" in msg:
        return VulkanInitFailed(msg)
    return VulkanInitFailed(msg)


def available() -> bool:
    """Return True if Vulkan loader + compute device can be initialized."""
    if _gpu is None:
        return False
    return bool(_gpu.available())


def device_name() -> str:
    """Return the active GPU name (calls init). Raises on failure / not built."""
    if _gpu is None:
        raise VulkanNotBuiltError(
            "cthreads built without CTHREADS_GPU; rebuild with -DCTHREADS_GPU=ON"
        )
    try:
        return str(_gpu.device_name())
    except Exception as exc:
        raise _map_error(exc) from exc


def init() -> None:
    """Explicitly initialize the Vulkan context."""
    if _gpu is None:
        raise VulkanNotBuiltError(
            "cthreads built without CTHREADS_GPU; rebuild with -DCTHREADS_GPU=ON"
        )
    try:
        _gpu.init()
    except Exception as exc:
        raise _map_error(exc) from exc


def shutdown() -> None:
    """Destroy device/instance and unload the Vulkan loader (no-op if not built)."""
    if _gpu is None:
        return
    _gpu.shutdown()


__all__ = [
    "CThreadsGPUError",
    "GPUNotAvailable",
    "GpuInvalidArgument",
    "GpuUseAfterDestroy",
    "VulkanInitFailed",
    "VulkanLoaderNotFound",
    "VulkanNoDevice",
    "VulkanNotBuiltError",
    "VulkanOutOfMemory",
    "available",
    "device_name",
    "init",
    "shutdown",
]
