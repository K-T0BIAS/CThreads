# GPU internals (C++ Vulkan path)

This folder documents the C++ GPU modules under `src/cthreads/cpp/gpu/`. It is written for newcomers who know C++ and maybe Python, but have never used Vulkan. Product-facing Python APIs such as `gpu()` and `@Gpu` are not finished yet. What you see here is the permanent substrate those APIs will call.

Related reading:

- Contributor Vulkan tutorial style notes: [docs/vk_guide/README.md](../../vk_guide/README.md)
- Future work (CPU `@Thread` calling GPU): [docs/gpu_future_cpu_to_gpu.md](../../gpu_future_cpu_to_gpu.md)
- Build flag: CMake `CTHREADS_GPU=ON` compiles these sources into `cthreads._ext`

## What problem this stack solves

CPU `@Thread` kernels run as normal operating system threads calling generated C++. GPU work is different. The GPU is a separate processor with its own memory. You cannot hand it a Python list pointer and expect a shader to read it. You must:

1. Open a connection to a Vulkan-capable GPU (the [Context](./context.md)).
2. Allocate GPU memory and copy host bytes in and out ([memory](./memory.md)).
3. Group one launch's buffers into a GpuPack ([pack](./pack.md)).
4. Tell the shader which binding number maps to which buffer ([descriptors](./descriptors.md)).
5. Build a reusable compute pipeline from SPIR-V bytes ([shader](./shader.md)).
6. Launch: record bind+dispatch, submit with a fence, then `join` writeback ([module](./module.md)).

## Mental model in one picture

```text
Python / tests
    |
    v
Context (loader, device, queue, function pointers, TransferEngine)
    |
    +-- memory::  create/destroy GpuBuffer, upload/download via staging
    |
    +-- pack::    GpuPack (scalar SSBO + list SSBOs)
    |               + descriptor pool/set/update (same namespace, descriptors.hpp)
    |
    +-- shader::  ShaderCacheEntry (module, layouts, pipeline)
    |             ShaderCache (symbol -> entry)
    |
    +-- launch::  launch_gpu_kernel -> SpawnedGpuKernel::join (writeback)
```

## Binding convention

cthreads packs shader arguments like this:

| Binding | Contents |
|-|-|
| 0 | One storage buffer for all scalars (`int`, `float`, `bool`, POD fields) |
| 1 | First list argument |
| 2 | Second list argument |
| ... | More lists |

There are no buffer device addresses (raw GPU pointers) stuffed inside the scalar struct. The descriptor set is the wiring table.

## Module index

| Doc | Namespace / types | Source |
|-|-|-|
| [Context](./context.md) | `cthreads::gpu::Context`, `TransferEngine` | `headers/context.hpp`, `impl/context.cpp` |
| [Memory](./memory.md) | `cthreads::gpu::memory` | `headers/memory.hpp`, `impl/memory.cpp` |
| [Pack](./pack.md) | `cthreads::gpu::pack` (GpuPack create/upload/download) | `headers/pack.hpp`, `impl/pack.cpp` |
| [Descriptors](./descriptors.md) | `cthreads::gpu::pack` (pool/set/update) | `headers/descriptors.hpp`, `impl/descriptors.cpp` |
| [Shader](./shader.md) | `cthreads::gpu::shader` | `headers/shader.hpp`, `shader_cache.hpp`, `impl/shader.cpp`, `shader_cache.cpp` |
| [Module / launch](./module.md) | `SpawnedGpuKernel`, `launch_gpu_kernel` | `headers/module.hpp`, `impl/module.cpp` |

## Build and availability

GPU code is compiled only when `CTHREADS_GPU` is on. The extension still loads `vulkan-1` (or `libvulkan.so.1`) dynamically at runtime. A machine without a Vulkan loader or GPU fails cleanly through Python `cthreads.gpu.available()` rather than crashing the import of CPU-only wheels.

## What is intentionally not here yet

- Public `gpu()` entry and `@Gpu` SPIR-V emit
- Threadable / schema marshal writeback (list[float]/list[int] writeback is in `SpawnedGpuKernel::join`)
- Inflight job store (header stub only)
- Dummy SSBOs for empty list slots in `update_descriptors`

Those build on the modules documented here.
