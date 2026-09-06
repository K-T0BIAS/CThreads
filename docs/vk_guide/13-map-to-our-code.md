# 13 — Map this guide to our repository

## Files (GPU)

| Path | Role |
|------|------|
| `src/cthreads/cpp/gpu/headers/context.hpp` | `Context` handles + all `PFN_*` |
| `src/cthreads/cpp/gpu/impl/context.cpp` | Load loader, init/shutdown, resolve entry points |
| `src/cthreads/cpp/gpu/headers/memory.hpp` | `BufferKind`, `GpuBuffer`, memory API |
| `src/cthreads/cpp/gpu/impl/memory.cpp` | find/create/destroy/upload/download |
| `src/cthreads/cpp/bindings/gpu_module.cpp` | pybind `cthreads._ext.gpu` |
| `src/cthreads/cpp/bindings/module.cpp` | `#ifdef CTHREADS_WITH_GPU` calls `bind_gpu` |
| `src/cthreads/cpp/CMakeLists.txt` | `CTHREADS_GPU` option + sources |
| `src/cthreads/python/cthreads/gpu/` | Python façade + errors |
| `tests/unit/test_gpu_context.py` | Context / availability tests (skip if no GPU) |

Expected as the GPU stack grows:

| Path (expected) | Role |
|-----------------|------|
| `gpu/headers/pack.hpp` (name may vary) | GpuPack |
| `gpu/.../transfer.*` | Reused command pool / staging |
| shader `.spv` / GLSL | Reference saxpy |
| more bindings | roundtrip / launch |

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

### Pass 3 — Descriptors and shaders

1. Read guide 09-12.
2. Sketch on paper the saxpy descriptor layout.
3. Sketch the command buffer for copy + dispatch + fence.

## Build flag reminder

```bat
set CMAKE_ARGS=-DCTHREADS_GPU=ON
pip install -e . -v
```

Wipe `build/` if CMake cached GPU off.

## Python error mapping

C++ throws `std::runtime_error` with prefixes like:

```text
cthreads.gpu.VulkanLoaderNotFound: ...
cthreads.gpu.VulkanInitFailed: ...
cthreads.gpu.VulkanNoDevice: ...
```

`cthreads.gpu` catches and raises typed exceptions. Keep prefixes stable when adding new errors.
