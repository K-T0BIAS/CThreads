"""
Public GPU probe wrappers.

Maps native C++ error prefixes to `frontend.errors` types and calls through
`_ext_gpu_api`.
"""

from .. import _ext_gpu_api
from .errors import (
    VulkanNotBuiltError,
    _map_error,
)


def available() -> bool:
    """
    Return True if the Vulkan loader and a compute device can be initialized.

    #### Returns
    - bool = True when GPU init would succeed

    #### Example:
    ``py
    from cthreads import gpu
    if gpu.available():
        print(gpu.device_name())
    ``
    """
    return _ext_gpu_api.available()


def device_name() -> str:
    """
    Return the active GPU name (calls init). Raises on failure or not built.

    #### Returns
    - str = Vulkan device name

    #### Raises
    - VulkanNotBuiltError = extension compiled without CTHREADS_GPU
    - CThreadsGPUError = mapped native init / device errors

    #### Example:
    ``py
    from cthreads import gpu
    name = gpu.device_name()
    ``
    """
    if _ext_gpu_api._gpu is None:
        raise VulkanNotBuiltError(
            "cthreads built without CTHREADS_GPU; rebuild with -DCTHREADS_GPU=ON"
        )
    try:
        return _ext_gpu_api.device_name()
    except Exception as exc:
        raise _map_error(exc) from exc


def init() -> None:
    """
    Explicitly initialize the Vulkan context.

    #### Returns
    - None

    #### Raises
    - VulkanNotBuiltError = extension compiled without CTHREADS_GPU
    - CThreadsGPUError = mapped native init errors

    #### Example:
    ``py
    from cthreads import gpu
    gpu.init()
    ``
    """
    if _ext_gpu_api._gpu is None:
        raise VulkanNotBuiltError(
            "cthreads built without CTHREADS_GPU; rebuild with -DCTHREADS_GPU=ON"
        )
    try:
        _ext_gpu_api.init()
    except Exception as exc:
        raise _map_error(exc) from exc


def shutdown() -> None:
    """
    Destroy the device/instance and unload the Vulkan loader.

    Native ShaderCache is released with the device (explicit memory management).
    Marks `prepare` so the next `prepare()` / `gpu()` rewalks the registry and
    re-registers SPIR-V. No-op when the GPU extension is not built.

    #### Returns
    - None

    #### Example:
    ``py
    from cthreads import gpu
    gpu.shutdown()
    ``
    """
    _ext_gpu_api.shutdown()
    from .. import runtime as runtime_mod

    runtime_mod._gpu_prepared = False
