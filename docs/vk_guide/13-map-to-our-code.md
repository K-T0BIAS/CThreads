# 13 — Map this guide to our repository

## Files (GPU)

| Path | Role |
|------|------|
| `src/cthreads/cpp/gpu/headers/context.hpp` | `Context` handles + all `PFN_*` |
| `src/cthreads/cpp/gpu/impl/context.cpp` | Load loader, init/shutdown, resolve entry points |
| `src/cthreads/cpp/gpu/headers/memory.hpp` | `BufferKind`, `GpuBuffer`, memory API |
| `src/cthreads/cpp/gpu/impl/memory.cpp` | find/create/destroy/upload/download + TransferEngine |
| `src/cthreads/cpp/gpu/headers/pack.hpp` | `GpuPack` create/upload/download |
| `src/cthreads/cpp/gpu/impl/pack.cpp` | Pack helpers |
| `src/cthreads/cpp/gpu/headers/descriptors.hpp` | Descriptor pool/set/update (namespace `pack`) |
| `src/cthreads/cpp/gpu/impl/descriptors.cpp` | Descriptor helpers |
| `src/cthreads/cpp/gpu/headers/shader.hpp` / `shader_cache.hpp` | `create_entry`, `ShaderCache` |
| `src/cthreads/cpp/gpu/impl/shader.cpp` / `shader_cache.cpp` | Pipeline build + cache |
| `src/cthreads/cpp/gpu/headers/module.hpp` | `SpawnedGpuKernel`, `launch_gpu_kernel` |
| `src/cthreads/cpp/gpu/impl/module.cpp` | Launch + join writeback |
| `src/cthreads/cpp/gpu/testing/` | Pack roundtrip + saxpy smoke SPIR-V |
| `src/cthreads/cpp/bindings/gpu_module.cpp` | pybind `cthreads._ext.gpu` |
| `src/cthreads/cpp/bindings/gpu_testing_module.cpp` | Test-only `_ext.gpu.testing` |
| `src/cthreads/cpp/bindings/module.cpp` | `#ifdef CTHREADS_WITH_GPU` calls `bind_gpu` |
| `src/cthreads/cpp/CMakeLists.txt` | `CTHREADS_GPU` option + sources |
| `src/cthreads/python/cthreads/gpu/` | Python façade + errors |
| `tests/unit/test_gpu_*.py` | Context / pack / shader+launch tests |

Newcomer-oriented C++ module docs: [docs/internals/gpu/README.md](../internals/gpu/README.md).

## How to study with the code open

### Pass 1 — Context

1. Read guide 03-05.
2. Open `context.hpp` — read every PFN comment.
3. Open `context.cpp` — follow `open_loader` -> `create_instance_and_device` -> `shutdown_unlocked`.
4. Run `gpu.available()` / `device_name()` with `CTHREADS_GPU=ON`.

### Pass 2 — Memory

1. Read guide 06-08.
2. Open `memory.hpp` — `BufferKind` + `GpuBuffer`.
3. Walk `create_buffer` and `upload_buffer` in `memory.cpp` line by line.
4. Mentally simulate uploading 4 floats.

### Pass 3 — Pack, descriptors, shaders, launch

1. Read guide 09-12.
2. Open `pack.hpp` / `descriptors.hpp` / `shader_cache.hpp` / `module.hpp`.
3. Trace product `launch_gpu_kernel` after `testing.register_smoke_saxpy`.
4. Run `tests/unit/test_gpu_shader.py::test_live_launch_saxpy_product_path`.

## Build flag reminder

```bat
set CMAKE_ARGS=-DCTHREADS_GPU=ON
pip install -e .
```

Wipe `build/` if CMake cached GPU off.

## Python error mapping

C++ throws `std::runtime_error` with prefixes like:

```text
cthreads.gpu.VulkanLoaderNotFound: ...
cthreads.gpu.VulkanInitFailed: ...
cthreads.gpu.VulkanNoDevice: ...
cthreads.gpu.GpuInvalidArgument: ...
```

`cthreads.gpu` catches and raises typed exceptions. Keep prefixes stable when adding new errors.
