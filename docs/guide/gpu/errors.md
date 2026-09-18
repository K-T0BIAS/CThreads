# GPU errors and troubleshooting

Probe helpers, exception types, and common failure modes for `cthreads.gpu`.

# Contents

- [Probe API](#probe-api)
- [Exception types](#exception-types)
- [When `@Gpu` raises at decorate time](#when-gpu-raises-at-decorate-time)
- [Common symptoms](#common-symptoms)
- [Build notes](#build-notes)

# Probe API

```python
from cthreads import gpu

gpu.available()    # bool: loader + compute device usable
gpu.device_name()  # str: raises if not built / init fails
gpu.init()         # explicit Vulkan init
gpu.shutdown()     # tear down device; next prepare/gpu recompiles
```

`available()` is the soft probe: it returns `False` instead of raising when the
path cannot start. Prefer it for feature detection.

`device_name()` and `init()` raise mapped errors on failure.

`shutdown()` releases the native shader cache with the device and marks the
runtime so the next `prepare()` / `gpu()` walks the registry again.

# Exception types

All of the following subclass `CThreadsGPUError` (except where noted).

| Type | Typical cause |
|------|----------------|
| `VulkanNotBuiltError` | `_ext` compiled without `CTHREADS_GPU` |
| `VulkanLoaderNotFound` | `vulkan-1.dll` / `libvulkan.so.1` missing |
| `VulkanNoDevice` | Loader present, no compute-capable device |
| `VulkanInitFailed` | Instance/device creation or missing entry points |
| `VulkanOutOfMemory` | GPU memory allocation failed |
| `GpuInvalidArgument` | Bad sizes, arena misuse, join flags, dtype mismatch |
| `GpuUseAfterDestroy` | Use after native resource destroy |
| `GPUNotAvailable` | Generic "GPU path not usable" from Python helpers |

Import:

```python
from cthreads.gpu import (
    CThreadsGPUError,
    GPUNotAvailable,
    GpuInvalidArgument,
    VulkanLoaderNotFound,
    VulkanNotBuiltError,
    VulkanNoDevice,
)
```

Native errors are mapped through `_map_error` based on message prefixes.

# When `@Gpu` raises at decorate time

`@Gpu` requires `available()` to be true. On a machine without Vulkan compute,
importing a module that eagerly decorates kernels can fail at import time.

Patterns:

1. Probe in `__main__` and import GPU modules only then.
2. Or document that the application requires a GPU.
3. For libraries, delay decoration / registration until the caller opts in.

# Common symptoms

| Symptom | Likely cause | What to try |
|---------|--------------|-------------|
| `VulkanNotBuiltError` | Extension built without GPU | `pip install cthreads-gpu` or rebuild with `-DCTHREADS_GPU=ON` |
| `available()` is False | No loader, no device, or CPU-only build | Update GPU drivers; confirm Vulkan ICD; install `cthreads-gpu` / rebuild with GPU ON |
| `VulkanLoaderNotFound` | Runtime library missing | Install/repair GPU drivers; on Linux install `vulkan-icd-loader` + vendor ICD |
| `TypeError` on decorate | Unsupported annotation | Scalars and `list` of scalars only |
| `TypeError` from `gpu()` | Missing `@Gpu` or bad arity | Check decorator and positional args |
| Wrong / partial results | Missing `i >= n` guard | Add bounds check |
| Stale Python lists in arena loop | No `sync`, `download=False` | Call `arena.sync()` |
| `GpuInvalidArgument` length changed | Mutated list length after bind | Rebind after resize |
| Barrier `TypeError` for `Barrier(n)` | Constructor in `@Gpu` | Use `Barrier.arrive_and_wait()` or `__sync_threads()` |
| `__sync_threads` RuntimeError | Called from host Python | Only inside compiled `@Gpu` bodies |
| GLSL / SPIR-V compile error | Unsupported statement or math | Simplify body; stick to documented subset |

# Build notes

End users of GPU-enabled wheels need **GPU drivers** with a Vulkan ICD. They do
not need the LunarG SDK to *run*.

Contributors building from source with GPU enabled:

```powershell
# PowerShell
$env:CMAKE_ARGS="-DCTHREADS_GPU=ON"
pip install -e ".[test]"
```

```bash
# bash
export CMAKE_ARGS="-DCTHREADS_GPU=ON"
pip install -e ".[test]"
```

Also valid:

```bash
pip install -e . --config-settings=cmake.define.CTHREADS_GPU=ON
```

`find_package(Vulkan)` requires Vulkan headers (SDK) on the build machine. Runtime
still loads the loader dynamically.

Full install context: [install.md](../../install.md#gpu-vulkan-compute).

## See also

- [quickstart.md](./quickstart.md)
- [api.md](./api.md)
- Contributor Vulkan notes: [vk_guide/03-sdk-runtime-drivers.md](../../vk_guide/03-sdk-runtime-drivers.md)
