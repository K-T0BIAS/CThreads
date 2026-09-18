"""
Lazy binding to `cthreads._ext.gpu` (native Vulkan submodule).

Central entry for all Python code that talks to the C++ GPU package.
Soft-imports so CPU-only builds still import `cthreads.gpu` cleanly.
"""

from __future__ import annotations

from typing import Any

try:
    from cthreads._ext import gpu as _gpu
except ImportError:
    _gpu = None  # type: ignore[assignment]


def _require_ext_gpu():
    """
    Return the native `_ext.gpu` module.

    #### Returns
    - module = pybind `cthreads._ext.gpu` submodule

    #### Raises
    - RuntimeError = extension was built without CTHREADS_GPU
    """
    if _gpu is None:
        raise RuntimeError(
            "cthreads._ext.gpu is not available - rebuild with -DCTHREADS_GPU=ON"
        )
    return _gpu


def available() -> bool:
    """
    Report whether the Vulkan loader and a compute device can initialize.

    Never raises. Returns False when the GPU extension is missing or init
    would fail.

    #### Returns
    - bool = True when a compute-capable GPU context can be created
    """
    if _gpu is None:
        return False
    return bool(_gpu.available())


def device_name() -> str:
    """
    Return the active GPU device name (may call init first).

    #### Returns
    - str = Vulkan device name string

    #### Raises
    - RuntimeError = `_ext.gpu` is not built
    - Exception = native init / device query failures (unmapped)
    """
    return str(_require_ext_gpu().device_name())


def init() -> None:
    """
    Explicitly initialize the process-wide Vulkan context.

    #### Returns
    - None

    #### Raises
    - RuntimeError = `_ext.gpu` is not built
    - Exception = native init failures (unmapped)
    """
    _require_ext_gpu().init()


def shutdown() -> None:
    """
    Destroy the Vulkan device/instance and unload the loader.

    No-op when the GPU extension is not built.

    #### Returns
    - None
    """
    if _gpu is None:
        return
    _gpu.shutdown()


def testing() -> Any | None:
    """
    Return the test-only `_ext.gpu.testing` submodule when present.

    #### Returns
    - module | None = testing helpers, or None if missing / not built
    """
    if _gpu is None:
        return None
    return getattr(_gpu, "testing", None)


def launch_gpu_kernel(meta: dict[str, Any], ordered_values: list[Any]) -> Any:
    """
    Submit one GPU kernel from metadata and ordered Python arguments.

    Returns a native GpuJob handle. Does not wait; the caller joins that
    handle for fence wait and list writeback. The kernel `symbol` must already
    be registered in the ShaderCache.

    #### Args:
    - meta: dict[str, Any] = kernel metadata (`symbol`, `params`, layout fields)
    - ordered_values: list[Any] = arguments in parameter order matching `meta`

    #### Returns
    - Any = native `_ext.gpu.GpuJob` (SpawnedGpuKernel)

    #### Raises
    - RuntimeError = `_ext.gpu` is not built
    - Exception = native launch failures (unmapped)

    #### Technical terms:
    - GpuJob: per-launch GPU job handle (fence, pack, writeback plan)
    - writeback: download of ref list buffers into the same Python list objects
    """
    return _require_ext_gpu().launch_gpu_kernel(meta, ordered_values)


def compile_glsl(source: str) -> bytes:
    """
    Compile GLSL compute source to SPIR-V via vendored glslang in `_ext`.

    #### Args:
    - source: str = full compute shader text

    #### Returns
    - bytes = SPIR-V binary

    #### Raises
    - RuntimeError = `_ext.gpu` is not built or lacks compile_glsl
    - Exception = native compile failures (unmapped)
    """
    gpu = _require_ext_gpu()
    compile_native = getattr(gpu, "compile_glsl", None)
    if not callable(compile_native):
        raise RuntimeError(
            "cthreads._ext.gpu.compile_glsl is missing - rebuild with "
            "CTHREADS_GPU=ON (glslang vendored into _ext)"
        )
    return bytes(compile_native(source))


def register_shader(symbol: str, spirv: bytes, binding_count: int) -> None:
    """
    Insert a compute pipeline into the process ShaderCache from SPIR-V bytes.

    Calls native `ShaderRegistry::register_spirv` (sole writer). Duplicate
    symbols raise.

    #### Args:
    - symbol: str = ShaderCache key / kernel name
    - spirv: bytes = SPIR-V binary (multiple of 4 bytes)
    - binding_count: int = number of STORAGE_BUFFER bindings (>= 1)

    #### Returns
    - None

    #### Raises
    - RuntimeError = `_ext.gpu` is not built
    - Exception = native create/insert failures (unmapped)
    """
    _require_ext_gpu().register_shader(symbol, spirv, binding_count)


def gpu_state() -> Any:
    """
    Return the process-wide native GpuState singleton.

    Named device-local buffers live here outside of a single launch. Does not
    expose Vulkan handles; use `add(name, nbytes)` / `remove(name)` / etc.

    #### Returns
    - Any = native `_ext.gpu.GpuState`

    #### Raises
    - RuntimeError = `_ext.gpu` is not built
    """
    return _require_ext_gpu().GpuState.instance()
