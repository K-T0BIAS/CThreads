# Context and TransferEngine

Source: `src/cthreads/cpp/gpu/headers/context.hpp`, `src/cthreads/cpp/gpu/impl/context.cpp`.

Namespace: `cthreads::gpu`.

This document explains the process-wide Vulkan connection that every other GPU helper needs. If you have never used Vulkan, start here.

## Why a Context exists

Vulkan is not a single library call like "run this kernel." It is a layered API:

1. An operating system library (the Vulkan loader) finds GPU drivers.
2. You create an instance (your application's connection to the loader).
3. You pick a physical device (a GPU the driver can see).
4. You create a logical device (your opened session on that GPU).
5. You get a queue (a submission port where you send work).
6. You resolve dozens of function pointers by name, because cthreads loads the loader dynamically instead of linking `vulkan-1` into every wheel.

The `Context` struct holds all of that state for the whole process. There is one Context, accessed through `context()`, guarded by a mutex for `init` / `shutdown` / `available`.

## Public functions

### `void init()`

Opens the loader, creates instance and device, resolves entry points, marks `ready = true`, and creates the TransferEngine and LaunchEngine. Throws typed-style error strings (for example `cthreads.gpu.VulkanLoaderNotFound`) that Python maps into exceptions in `cthreads.gpu`.

Call this when you need the GPU. Python `cthreads.gpu.init()` ends up here.

### `void shutdown()`

Destroys children first, then parents:

1. LaunchEngine (free fences, then command pool)
2. TransferEngine (staging buffer, fence, command pool)
2. Shader cache entries (pipelines and layouts)
3. Logical device
4. Instance
5. Unload the loader library
6. Clear all function pointers and handles

Safe to call if the Context was never initialized.

### `bool available()`

Tries to ensure the Context is ready without throwing to the caller for "no GPU" style probes. Used by Python `available()` and by tests that skip when there is no device.

### `const std::string& device_name()`

Returns the human-readable GPU name from Vulkan device properties. Requires a ready Context.

### `Context& context()`

Returns the singleton. Most C++ helpers take `Context&` explicitly so ownership and testing stay clear.

## Struct `Context` fields (grouped)

### Loader and bootstrap

- `loader_module`: operating system handle to `vulkan-1.dll` / `libvulkan.so.1`.
- `vkGetInstanceProcAddr`: the bootstrap function used to look up almost every other Vulkan function by name string.

### Instance and device entry points

Examples: `vkCreateInstance`, `vkEnumeratePhysicalDevices`, `vkCreateDevice`, `vkGetDeviceQueue`, `vkDestroyDevice`.

These are stored as typed function pointers (`PFN_vk...`). A pointer-to-function (PFN) is simply a C function pointer with the Vulkan signature. cthreads assigns them during init after the loader is open.

### Buffer and memory entry points

Used by [memory](./memory.md): create/destroy buffers, allocate/free device memory, map host-visible memory, query memory properties.

### Command, copy, and sync entry points

Used by transfers and launch: command pools, command buffers, `vkCmdCopyBuffer`, fences, `vkQueueSubmit`, wait/reset fences.

### Shader and pipeline entry points

Used by [shader](./shader.md): create/destroy shader modules, descriptor set layouts, pipeline layouts, and compute pipelines.

### Descriptor pool and update entry points

Used by [descriptors](./descriptors.md): create/destroy descriptor pools, allocate/free sets, `vkUpdateDescriptorSets`.

### Compute dispatch entry points

Used by [module / launch](./module.md): `vkCmdBindPipeline`, `vkCmdBindDescriptorSets`, `vkCmdDispatch`, `vkCmdPipelineBarrier`.

### Opaque handles

- `instance`: connection to the loader for this application.
- `physical_device`: the chosen GPU.
- `device`: the logical device (opened GPU session).
- `queue`: compute queue used to submit copies and dispatches.
- `queue_family`: index of the queue family that supports compute (needed when creating command pools).
- `device_name`: string name for logging and Python.
- `ready`: true only after a fully successful init.

### TransferEngine and mutex

See the next section. `transfer_engine_mutex` serializes use of the single shared engine.

### LaunchEngine and mutex

`LaunchEngine` owns the process-lifetime command pool used by `launch_gpu_kernel`. Jobs checkout a command buffer + fence, submit under `launch_engine_mutex`, wait their own fence on join, then return the CB/fence to free lists. Overlapping jobs are supported; one shared fence is not.

## Technical terms

- Vulkan loader: system library that discovers Installable Client Drivers (ICDs), which are the vendor GPU drivers.
- Instance: application-level Vulkan object. Not the GPU itself.
- Physical device: one GPU as seen by the driver.
- Logical device: your opened handle to use that GPU.
- Queue: port where the CPU submits recorded command buffers.
- Queue family: group of queues with the same capabilities (graphics, compute, transfer).
- Dynamic loading: open the DLL/SO at runtime and resolve symbols by name, instead of linking at build time.
- `VK_NULL_HANDLE`: sentinel meaning "no object."

## TransferEngine

### Purpose

Uploading and downloading buffers needs a short GPU copy: host-visible staging memory to or from a device-local buffer. Creating a brand new command pool and fence for every copy would be slow and wasteful. The TransferEngine keeps reusable machinery on the Context for the process lifetime.

### Fields

- `command_pool`: Vulkan command pool for the compute queue family. Command buffers for copies are allocated from here and freed after each wait.
- `fence`: CPU-waitable fence. Created signaled so the first reset/wait path treats it as idle. After each copy, the CPU waits on this fence.
- `staging`: optional host-visible `GpuBuffer`. Empty until the first upload or download grows it. It only grows; it never shrinks until Context shutdown.

### Lifecycle

- Created in `init_transfer_engine` after the device is ready.
- Destroyed in `shutdown_transfer_engine` before the logical device is destroyed.
- Staging is grown by `memory::ensure_staging` under the transfer engine mutex.

### Concurrency

There is one engine today. All upload/download paths hold `transfer_engine_mutex` for the whole transfer (staging grow, memcpy, GPU copy, wait). That avoids races on the shared pool, fence, staging buffer, and queue submit.

A future pool of engines is discussed for CPU threads launching GPU work. See [gpu_future_cpu_to_gpu.md](../../gpu_future_cpu_to_gpu.md).

## Key Vulkan calls during init (story order)

1. Load the loader library (`LoadLibrary` / `dlopen`).
2. Resolve `vkGetInstanceProcAddr`, then `vkCreateInstance`.
3. `vkCreateInstance` builds the instance.
4. Resolve instance-level functions.
5. `vkEnumeratePhysicalDevices` lists GPUs; pick one with a compute queue family (prefer discrete when possible).
6. `vkCreateDevice` opens the logical device; `vkGetDeviceQueue` gets the queue.
7. Resolve device-level functions (buffers, commands, shaders, descriptors).
8. Create TransferEngine pool and fence.
9. Create LaunchEngine command pool (CB/fence free lists grow on demand).
9. Set `ready = true`.

## Key Vulkan calls during shutdown (story order)

1. Wait on the transfer fence if needed; destroy staging, fence, command pool.
2. Clear the shader cache (destroy pipelines and layouts).
3. `vkDestroyDevice`.
4. `vkDestroyInstance`.
5. Unload the loader; null all pointers.

## Relationship to other modules

Every helper in memory, pack, descriptors, and shader takes a `Context&` and expects `ready == true` plus the entry points it needs. If init failed or shutdown already ran, those helpers throw `VulkanInitFailed`-style errors.

## Python surface today

`cthreads.gpu` exposes `init`, `shutdown`, `available`, and `device_name`, plus typed errors. It does not yet expose packs, shaders, or jobs. Those will wrap the C++ types documented in the sibling pages.
