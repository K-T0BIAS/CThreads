"""
High-level GPU prepare + `gpu()` launch entry.

Mirrors CPU `cthreads.prepare` / `thread`: compile registered `@Gpu` kernels,
then launch via `_ext.gpu.launch_gpu_kernel`.
"""

from typing import Any, Callable

from ..job import Job
from . import _ext_gpu_api
from .compiler.orchestrator import GpuCompileSession
from .frontend.errors import GPUNotAvailable, _map_error
from .gpu_kernel_meta import build_gpu_kernel_meta
from .gpu_marshal import infer_group_count_x, ordered_values_for_meta

# True after a successful GpuCompileSession.compile in this process.
# Cleared by frontend shutdown() when native ShaderCache is released.
_gpu_prepared: bool = False


class GpuJob(Job):
    """
    Job wrapper for native GpuJob handles (void kernels; `result()` is None).
    """

    def result(self) -> None:
        """
        GPU kernels are writeback-only; there is no scalar return value.

        #### Returns
        - None
        """
        return None


def compile(force: bool = False) -> dict[str, Any]:
    """
    Drain registered `@Gpu` functions through GpuCompileSession (SPIR-V emit).

    #### Args:
    - force: bool = rewrite `__Gpu__` artifacts even when fingerprints match

    #### Returns
    - dict[str, Any] = session result (`root`, `cache`, `rewritten`)
    """
    global _gpu_prepared
    info = GpuCompileSession.compile(force=force)
    _gpu_prepared = True
    return info


def prepare(force: bool = False) -> dict[str, Any]:
    """
    Compile all registered `@Gpu` kernels and register SPIR-V in ShaderCache.

    #### Args:
    - force: bool = if True, re-init Vulkan and force-rebuild GPU units

    #### Returns
    - dict[str, Any] = compile session info

    #### Raises
    - GPUNotAvailable = no usable Vulkan device / GPU extension
    - RuntimeError = nothing registered or compile failed
    """
    global _gpu_prepared
    if not _ext_gpu_api.available():
        raise GPUNotAvailable(
            "GPU is not available (build with CTHREADS_GPU=ON and a Vulkan device)"
        )
    if force:
        _ext_gpu_api.shutdown()
        _ext_gpu_api.init()
        _gpu_prepared = False
    return compile(force=force)


def gpu(
    fn: Callable[..., Any],
    *args: Any,
    force: bool = False,
    **kwargs: Any,
) -> GpuJob:
    """
    Launch a `@Gpu` kernel and return a joinable job handle.

    Ensures GPU compile/emit has run, then submits via `launch_gpu_kernel`.
    List arguments are written back in place on `join()`.

    #### Args:
    - fn: Callable = `@Gpu`-decorated kernel
    - *args: Any = positional kernel arguments (param order)
    - force: bool = force recompile + Vulkan re-init before launch
    - **kwargs: Any = not supported yet

    #### Returns
    - GpuJob = awaitable / joinable handle (result is always None)

    #### Raises
    - TypeError = missing `@Gpu`, bad arity, or unexpected kwargs
    - GPUNotAvailable = Vulkan path not usable
    - Exception = mapped native launch failures

    #### Example:
    ``py
    from cthreads.gpu import Gpu, GlobalIdx, gpu

    @Gpu
    def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = a * x[i] + y[i]

    x = [1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 30.0, 40.0]
    gpu(saxpy, len(x), 2.0, x, y).join()
    ``
    """
    global _gpu_prepared

    if kwargs:
        raise TypeError("gpu(): keyword arguments are not supported yet")
    if not callable(fn):
        raise TypeError("gpu(): fn must be callable")
    if not getattr(fn, "__gpu__", False):
        raise TypeError(
            f"gpu(): {getattr(fn, '__qualname__', fn)!r} is not a @Gpu function"
        )

    if not _ext_gpu_api.available():
        raise GPUNotAvailable(
            "GPU is not available (build with CTHREADS_GPU=ON and a Vulkan device)"
        )

    if force or not _gpu_prepared:
        prepare(force=force)

    meta_obj = getattr(fn, "__gpu_kernel_meta__", None)
    if not isinstance(meta_obj, dict):
        meta_obj = build_gpu_kernel_meta(fn).to_dict()
    meta: dict[str, Any] = dict(meta_obj)

    ordered = ordered_values_for_meta(meta, args)
    if meta.get("group_count_x") is None:
        meta["group_count_x"] = infer_group_count_x(meta, ordered)
    if meta.get("group_count_y") is None:
        meta["group_count_y"] = 1
    if meta.get("group_count_z") is None:
        meta["group_count_z"] = 1

    try:
        raw = _ext_gpu_api.launch_gpu_kernel(meta, ordered)
    except Exception as exc:
        raise _map_error(exc) from exc

    job = GpuJob(raw)
    job.start()
    return job
