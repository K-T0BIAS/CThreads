"""ctypes-style error types for cthreads.gpu (mapped from C++ message prefixes)."""


class CThreadsGPUError(Exception):
    def __init__(self, detail: str = "Unknown Error") -> None:
        self.detail = detail
        self.message = f"\033[91mcthreads gpu error\033[0m: {detail}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class VulkanNotBuiltError(CThreadsGPUError):
    """_ext was compiled without CTHREADS_GPU."""

    def __init__(self, detail: str = "cthreads built without CTHREADS_GPU") -> None:
        super().__init__(detail)


class VulkanLoaderNotFound(CThreadsGPUError):
    """vulkan-1.dll / libvulkan.so.1 could not be loaded."""

    def __init__(self, detail: str = "Vulkan loader not found") -> None:
        super().__init__(detail)


class VulkanNoDevice(CThreadsGPUError):
    """Loader ok but no compute-capable device."""

    def __init__(self, detail: str = "No Vulkan compute device") -> None:
        super().__init__(detail)


class VulkanInitFailed(CThreadsGPUError):
    """Instance/device creation or missing Vulkan entry point."""

    def __init__(self, detail: str = "Vulkan init failed") -> None:
        super().__init__(detail)


class GPUNotAvailable(CThreadsGPUError):
    """Generic: GPU path not usable (not built, no loader, or no device)."""

    def __init__(self, detail: str = "GPU not available") -> None:
        super().__init__(detail)
