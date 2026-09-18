"""
Vulkan GPU runtime probe API.

Native access goes through `_ext_gpu_api` (`cthreads._ext.gpu`).
Public helpers and error types are re-exported from `frontend`.

Launch helpers (`prepare` / `gpu` / `compile`) live in `runtime` so the
callable name `prepare` does not shadow a submodule.
"""

from . import _ext_gpu_api
from .frontend import (
    BlockDim,
    BlockIdx,
    CThreadsGPUError,
    GPUNotAvailable,
    GlobalIdx,
    GridDim,
    Gpu,
    GpuInvalidArgument,
    GpuUseAfterDestroy,
    ThreadIdx,
    VulkanInitFailed,
    VulkanLoaderNotFound,
    VulkanNoDevice,
    VulkanNotBuiltError,
    VulkanOutOfMemory,
    _map_error,
    available,
    device_name,
    init,
    shutdown,
)
from .arena import GpuArena
from .runtime import GpuJob, compile, gpu, prepare


def __getattr__(name: str):
    if name == "_gpu":
        return _ext_gpu_api._gpu
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Gpu",
    "GpuArena",
    "GpuJob",
    "BlockDim",
    "BlockIdx",
    "GlobalIdx",
    "GridDim",
    "ThreadIdx",
    "CThreadsGPUError",
    "GPUNotAvailable",
    "GpuInvalidArgument",
    "GpuUseAfterDestroy",
    "VulkanInitFailed",
    "VulkanLoaderNotFound",
    "VulkanNoDevice",
    "VulkanNotBuiltError",
    "VulkanOutOfMemory",
    "_map_error",
    "_gpu",
    "_ext_gpu_api",
    "available",
    "compile",
    "device_name",
    "gpu",
    "init",
    "prepare",
    "shutdown",
]
