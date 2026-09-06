# 02 — Mental model: objects, handles, and "nothing runs until submit"

## Vulkan is an object graph

Almost everything you create is an **object**:

- Instance
- Physical device (the GPU hardware the loader knows about)
- Logical device (your open connection to that GPU)
- Queue (a port you submit work to)
- Buffer (a typed "array of bytes" object)
- Device memory (the actual allocated slab)
- Command pool / command buffer (recorded GPU instructions)
- Fence (CPU-side "GPU finished" signal)
- Shader module, pipeline, descriptor set (launch path)

Objects form a tree of ownership. Typical rule:

**Destroy children before parents.** Example: destroy buffers and command pools before destroying the logical device; destroy the device before the instance.

## Handles are not pointers you dereference

When Vulkan returns a `VkBuffer`, you get an opaque **handle** (often an integer-sized id).

You cannot do:

```cpp
buffer->data[i] = 1.0f;  // NOT how Vulkan works
```

You talk to objects only through API functions:

```cpp
vkDestroyBuffer(device, buffer, nullptr);
```

`VK_NULL_HANDLE` means "no object" (like a null pointer, but for handles).

In our code, `Context` stores many handles and many **function pointers** (`PFN_vkCreateBuffer`, …) because we load Vulkan dynamically.

## Explicit means: you fill structs

Vulkan APIs almost always take a `*CreateInfo` struct:

```cpp
VkBufferCreateInfo info{};
info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
info.size = 1024;
info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
...
vkCreateBuffer(device, &info, nullptr, &buffer);
```

Rules of thumb:

1. Zero the struct (`{}` in C++).
2. Set `sType` to the matching enum (tells the driver which struct this is).
3. Set only the fields you need; leave the rest zero/null.
4. Check the `VkResult` return (`VK_SUCCESS` or throw).

This pattern repeats everywhere. Once you recognize it, Vulkan stops feeling like "a million APIs" and starts feeling like "the same paperwork for every object."

## CPU time vs GPU time

This is the biggest intuition jump.

```text
CPU thread                         GPU
---------                         ---
record commands into CB
submit CB to queue  ------------>  (maybe starts later)
do other CPU work                  execute copies / dispatches
vkWaitForFences  <---------------  signals fence when done
```

If you forget to wait, you might download a buffer the GPU has not finished writing.
That is a classic bug class (CPU/GPU race).

The upload/download helpers wait on a fence after the copy so the CPU memcpy of staging memory is safe.

## Queues are submission ports

A GPU exposes **queue families** (groups of queues with capabilities):

- graphics
- compute
- transfer
- present (for windows — we ignore)

cthreads picks a family that supports **compute** (and also uses it for buffer copies).
We create one logical device requesting that family, then call `vkGetDeviceQueue` to get `VkQueue queue`.

All work we care about goes: **record -> vkQueueSubmit(queue, ...)**.

## Two kinds of "memory" people confuse

1. **Host memory** — normal RAM your C++ `memcpy` can touch.
2. **Device memory** — memory the GPU allocates through Vulkan (`VkDeviceMemory`).

Some device memory is **host-visible**: the driver can give you a CPU pointer via `vkMapMemory` (staging).
Some is **device-local**: fast for the GPU, **not** for persistent CPU poking (our SSBOs).

A `VkBuffer` alone is not enough. You must:

1. Create the buffer object (describes size/usage).
2. Ask for memory requirements.
3. Allocate matching `VkDeviceMemory`.
4. **Bind** memory to the buffer.

Until bind succeeds, the buffer cannot store data.

## Layers of our stack (keep this map)

```text
Python cthreads.gpu / future gpu()
        |
        v
pybind _ext.gpu
        |
        v
GpuPack (option 5)
        |
        v
memory:: create/upload/download
        |
        v
Context (instance/device/queue/PFNs)
        |
        v
vulkan-1.dll / libvulkan.so.1  (loader)
        |
        v
GPU driver (ICD)
```

## Validation layers (optional while learning)

The Vulkan SDK can enable **validation layers**: extra checks that print readable errors when you misuse the API.

For shipping cthreads to users we do **not** require layers.
While developing, turning them on is highly recommended (see SDK docs for `VK_LAYER_KHRONOS_validation`).

We may gate this behind something like `CTHREADS_VK_VALIDATE=1` later. Not required to understand Context and memory.
