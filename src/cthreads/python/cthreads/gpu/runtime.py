"""
High-level GPU prepare + `gpu()` launch entry.

Mirrors CPU `cthreads.prepare` / `thread`: compile registered `@Gpu` kernels,
then launch via `_ext.gpu.launch_gpu_kernel`.
"""

from typing import Any, Callable

from ..job import Job
from . import _ext_gpu_api
from .arena import lookup_resident
from .compiler.orchestrator import GpuCompileSession
from .frontend.errors import GPUNotAvailable, GpuInvalidArgument, _map_error
from .gpu_kernel_meta import build_gpu_kernel_meta
from .gpu_marshal import infer_group_count_x, ordered_values_for_meta

# True after a successful GpuCompileSession.compile in this process.
# Cleared by frontend shutdown() when native ShaderCache is released.
_gpu_prepared: bool = False


class GpuJob(Job):
    """
    Job wrapper for native GpuJob handles (void kernels; `result()` is None).
    """

    def join(self, download: bool = True) -> None:
        """
        Wait for the GPU fence; optionally write ref lists back into Python.

        #### Args:
        - download: bool = if True (default), download ref lists on join.
          If False, skip writeback (use GpuArena.sync for resident lists).

        #### Returns
        - None
        """
        if not self._started:
            self.start()
        raw_join = self._raw.join
        try:
            raw_join(download)
        except TypeError:
            # Older native builds without the download argument.
            if download is False:
                raise GpuInvalidArgument(
                    "GpuJob.join(download=False) requires a rebuild with "
                    "CTHREADS_GPU residency support"
                ) from None
            raw_join()

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


def _resident_meta_for_args(
    meta: dict[str, Any], ordered: list[Any]
) -> dict[int, str]:
    """
    Map value_index -> GpuState name for arena-bound list args.

    Raises if a bound list length no longer matches the registered numel.
    """
    params = meta.get("params")
    if not isinstance(params, list):
        return {}
    resident: dict[int, str] = {}
    for i, param in enumerate(params):
        if not isinstance(param, dict) or param.get("kind") != "list":
            continue
        slot = lookup_resident(ordered[i])
        if slot is None:
            continue
        host = ordered[i]
        if not isinstance(host, list):
            continue
        if len(host) != slot.numel:
            raise GpuInvalidArgument(
                f"gpu(): bound list {slot.name!r} length changed "
                f"(was {slot.numel}, now {len(host)}); call arena.bind again"
            )
        meta_kind = param.get("elem_kind")
        if meta_kind is not None and meta_kind != slot.elem_kind:
            raise GpuInvalidArgument(
                f"gpu(): bound list {slot.name!r} elem_kind {slot.elem_kind!r} "
                f"does not match kernel {meta_kind!r}"
            )
        resident[i] = slot.state_name
    return resident


def gpu(
    fn: Callable[..., Any],
    *args: Any,
    force: bool = False,
    **kwargs: Any,
) -> GpuJob:
    """
    Launch a `@Gpu` kernel and return a joinable job handle.

    Ensures GPU compile/emit has run, then submits via `launch_gpu_kernel`.
    List arguments are written back in place on `join()` unless
    `join(download=False)` is used with GpuArena-resident lists.

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

    resident = _resident_meta_for_args(meta, ordered)
    if resident:
        meta["resident"] = resident

    try:
        raw = _ext_gpu_api.launch_gpu_kernel(meta, ordered)
    except Exception as exc:
        raise _map_error(exc) from exc

    job = GpuJob(raw)
    job.start()
    return job
