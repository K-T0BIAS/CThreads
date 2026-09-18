"""
`@Gpu` decorator: mark, validate, and register a GPU kernel function.

Does not emit SPIR-V or launch. That happens later via GpuCompileSession / `gpu()`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from ...frontend.Registry import REGISTRY
from ..gpu_kernel_meta import build_gpu_kernel_meta
from .errors import GPUNotAvailable
from .lib import available, device_name

F = TypeVar("F", bound=Callable[..., Any])


def Gpu(fn: F | None = None, *, log: bool = False):
    """
    Mark a function as a `@Gpu` kernel and register it for later compile/launch.

    Supports `@Gpu` and `@Gpu(log=True)`. Validates annotations against the GPU
    allowlist (`-> None`, scalars and `list` of scalars), attaches
    `fn.__gpu_kernel_meta__`, and returns the same function (still runs as
    normal Python when called directly).

    #### Args:
    - fn: Callable | None = function when used as `@Gpu`; omit for `@Gpu(log=...)`
    - log: bool = if True, print the assigned device name (default False)

    #### Returns
    - Callable = the marked function, or a decorator when `fn` is omitted

    #### Raises
    - GPUNotAvailable = Vulkan GPU path is not usable in this process
    - TypeError = missing or unsupported annotations

    #### Example:
    ``py
    from cthreads.gpu import Gpu

    @Gpu
    def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
        pass

    @Gpu(log=True)
    def saxpy_logged(n: int, a: float, x: list[float], y: list[float]) -> None:
        pass
    ``
    """

    def apply(f: F) -> F:
        if not available():
            raise GPUNotAvailable(
                "GPU is not available (build with CTHREADS_GPU=ON and a Vulkan "
                "device)"
            )

        f.__gpu__ = True # type: ignore[attr-defined]
        f.__gpu_version__ = REGISTRY.VERSION # type: ignore[attr-defined]

        # Validate annotations and attach launch meta (no SPIR-V yet).
        build_gpu_kernel_meta(f)

        REGISTRY.register_gpu_function(f)

        if log:
            print(
                f"\033[92mGPU LOG:\033[0m function {f.__name__} is assigned to "
                f"GPU {device_name()}"
            )
        return f

    if fn is not None:
        return apply(fn)
    return apply
