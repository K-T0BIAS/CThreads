# 00 — Read me first

## What cthreads is building

cthreads already turns typed Python (`@Thread`) into **CPU** native kernels:

```text
Python args -> pack (C++ struct) -> run on OS thread -> writeback -> same Python objects
```

The GPU path follows the **same product idea** with a different middle:

```text
Python args -> GpuPack (Vulkan buffers) -> run SPIR-V compute -> writeback -> same Python objects
```

Users still pass `list` / `int` / `float`. They do **not** learn a second buffer type.
Vulkan stays inside `_ext`.

## What Vulkan is (intuition)

Vulkan is a **low-level remote control for the GPU**.

- The API does not hide a high-level "draw a triangle" call.
- Callers create memory, bind it, record commands, submit them to a queue, and wait until done.
- The driver does less magic. The application owns lifetimes, synchronization, and layouts.

That is why Vulkan feels verbose. Every step that OpenGL/CUDA hid becomes a function call.
The upside: portable compute on NVIDIA / AMD / Intel with one API, and predictable behavior.

## What this project deliberately ignores

These topics are **out of scope** for the cthreads compute backend:

| Topic | Why skip |
|-------|----------|
| Swapchain / windows / present | Not drawing to the screen |
| Render passes / framebuffers | Graphics only |
| Images / textures / samplers | Buffer-first compute path |
| Graphics pipelines / vertex input | Compute only |
| Multi-GPU / sparse memory | Out of scope for now |
| Buffer device address / bindless | Rejected for the current design |
| Ray tracing | Out of scope |

If a random tutorial spends chapters on "hello triangle," skim or skip those sections.
The cthreads "hello world" is: **upload floats, run compute (when landed), download floats**.

## How Vulkan compares to things contributors might know

### vs writing normal C++

| C++ | Vulkan |
|-----|--------|
| `new[]` / `malloc` | `vkAllocateMemory` + choose memory type |
| `memcpy` to a buffer the CPU owns | `memcpy` only works if memory is **host-visible** |
| Call a function | Record commands, then **submit** to a queue (async GPU) |
| Function returns when done | GPU may still be running; **fence** or similar to wait |

### vs CUDA (if familiar)

| CUDA | cthreads Vulkan path |
|------|----------------------|
| `cudaMalloc` | device-local `VkBuffer` + memory |
| `cudaMemcpy` | staging buffer + `vkCmdCopyBuffer` |
| `<<<grid,block>>>` kernel launch | `vkCmdDispatch` in a command buffer |
| CUDA toolkit for end users | **Drivers only** for end users; SDK headers for *building* GPU-enabled `_ext` |

### vs OpenGL compute

OpenGL hides a lot of binding and sync. Vulkan makes bind points and barriers explicit.
Same idea (SSBO = storage buffer), more paperwork.

## The one sentence to remember

**Vulkan objects are handles to driver resources; almost nothing happens until a command buffer is submitted to a queue; CPU and GPU run out of sync unless the CPU waits.**

## Locked decisions in this project (so generic tutorials do not confuse contributors)

1. **Binding convention pack:** one scalar SSBO + one SSBO per `list`.
2. **Device-local** data for shaders; **staging** for CPU copies.
3. **Descriptors** bind buffers by binding index (not pointers in the scalar struct).
4. **Launch then join** — no mid-run Python `__sync_state` on GPU.
5. **Dynamic load** `vulkan-1.dll` / `libvulkan.so.1` — do not hard-link for default CPU wheels.

If a blog says "just map the SSBO and memcpy," that is a shortcut cthreads does **not** use as the production list path.

## Suggested study rhythm

1. Read 01-03 for intuition (no code required).
2. Read 04-05 while looking at `gpu/headers/context.hpp` and `gpu/impl/context.cpp`.
3. Read 06-08 while looking at `gpu/headers/memory.hpp` and `gpu/impl/memory.cpp`.
4. Read 09-12 before working on pipelines, descriptors, or `@Gpu` emit.
5. Keep 14 glossary open while coding.
