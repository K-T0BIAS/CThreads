# Memory helpers

Source: `src/cthreads/cpp/gpu/headers/memory.hpp`, `src/cthreads/cpp/gpu/impl/memory.cpp`.

Namespace: `cthreads::gpu::memory`.

Depends on: [Context and TransferEngine](./context.md).

This module owns one idea: a contiguous byte region on the GPU (`GpuBuffer`), plus helpers to create it, destroy it, and copy bytes between host memory and device-local storage.

## Host memory versus device memory

- Host memory is ordinary process RAM. C++ `memcpy` and Python buffers live here.
- Device-local memory is GPU memory that shaders prefer. The CPU usually cannot keep a permanent pointer into it.
- Host-visible memory is a special GPU allocation the CPU can map. It is often slower for heavy shader traffic, so cthreads uses it only as a temporary staging mirror.

cthreads's rule: shader-facing data lives in device-local buffers. Host traffic always goes through staging plus a GPU copy.

## Technical terms

- Storage buffer (SSBO): a buffer a compute shader can read and write through a descriptor binding.
- Staging buffer: host-visible buffer used only as the CPU side of an upload or download.
- `VkBuffer`: opaque Vulkan handle describing a buffer resource (size and usage). It is not a raw C pointer to bytes.
- `VkDeviceMemory`: opaque handle for an allocated memory slab. You bind a buffer to memory before using it.
- Memory type: one of the driver-exposed categories (device-local, host-visible, host-coherent, and so on).
- Fence: a GPU timeline object the CPU can wait on until submitted work finishes.
- Command buffer: a recorded list of GPU commands (for transfers, usually a single copy).
- Command pool: allocator that owns command buffers for one queue family.

## Enum `BufferKind`

One create API, two property sets.

### `BufferKind::Staging`

- Usage: transfer source and transfer destination.
- Memory: host-visible and host-coherent.
- After create, the buffer is mapped and `GpuBuffer::mapped` points at CPU-writable bytes.

### `BufferKind::DeviceLocal`

- Usage: storage buffer plus transfer source and destination (so shaders and copies both work).
- Memory: device-local.
- Never persistently mapped (`mapped` stays null).

## Struct `GpuBuffer`

Fields:

- `buffer`: `VkBuffer` handle, or `VK_NULL_HANDLE` if empty.
- `memory`: `VkDeviceMemory` bound to that buffer, or null handle if empty.
- `size`: caller-facing byte count requested at create time.
- `mapped`: CPU pointer for staging only.
- `kind`: staging or device-local.

Ownership: the struct owns the Vulkan objects until `destroy_buffer` runs. Moving the struct moves the handles; it does not clone GPU bytes.

## Function `find_memory_type`

Vulkan returns a bitmask of legal memory types for a new buffer (`type_bits`). You also request property flags (for example host-visible). This helper walks the physical device's memory types and returns the first index that is allowed by the bitmask and has every requested property.

Used internally by `create_buffer`. You rarely call it from higher layers.

## Function `create_buffer`

Creates a `GpuBuffer` of the given kind and size.

Steps in plain language:

1. Reject size 0 and missing entry points.
2. Choose usage flags and memory properties from `BufferKind`.
3. Call `vkCreateBuffer` to create the buffer object.
4. Query memory requirements with `vkGetBufferMemoryRequirements`.
5. Pick a memory type with `find_memory_type`.
6. Call `vkAllocateMemory`, then `vkBindBufferMemory` at offset 0.
7. For staging, call `vkMapMemory` and store the pointer in `mapped`.

Throws on failure. On partial failure it cleans up objects it already created.

## Function `destroy_buffer`

Unmaps staging memory if needed, destroys the `VkBuffer`, frees `VkDeviceMemory`, and clears the struct. Safe on an already-empty buffer.

## Function `upload_buffer`

Copies host bytes into a device-local `GpuBuffer`.

Path:

```text
host pointer
  -> memcpy into TransferEngine staging (grown if needed)
  -> GPU vkCmdCopyBuffer staging -> device-local
  -> wait on TransferEngine fence
```

The whole path holds `context.transfer_engine_mutex` so two threads cannot share the engine's staging, pool, or fence unsafely.

Requirements:

- Target buffer must be device-local and non-null.
- `data` non-null; `size` in `(0, buffer.size]`.

## Function `download_buffer`

Opposite direction:

```text
device-local
  -> GPU copy into TransferEngine staging
  -> wait
  -> memcpy staging -> host pointer
```

Same mutex and validation rules as upload.

## Internal helper `copy_buffer_and_wait`

Not part of the public header. Records a one-time command buffer that copies `size` bytes from one `VkBuffer` to another, submits it on the Context queue with the TransferEngine fence, waits, then frees the command buffer.

Important Vulkan calls:

- `vkAllocateCommandBuffers` / `vkFreeCommandBuffers`
- `vkBeginCommandBuffer` / `vkEndCommandBuffer`
- `vkCmdCopyBuffer`
- `vkResetFences` / `vkQueueSubmit` / `vkWaitForFences`

Caller must already hold the transfer engine mutex.

## Internal helper `ensure_staging`

Grows the TransferEngine staging buffer so it is at least `size` bytes. Never shrinks until Context shutdown. Caller must hold the transfer engine mutex.

## Why not map device-local buffers?

Many discrete GPUs cannot give the CPU a fast permanent pointer to device-local memory. Staging plus an explicit GPU copy is the portable model and matches how real engines move data.

## How pack uses this

[Pack](./pack.md) creates one device-local `GpuBuffer` for scalars and one per non-empty list. Upload and download helpers on the pack call `upload_buffer` / `download_buffer` for each region.
