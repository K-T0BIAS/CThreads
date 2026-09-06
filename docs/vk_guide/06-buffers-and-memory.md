# 06 — Buffers and device memory

This is the core of the memory layer (before copies).

## Two objects, one job

To store N bytes the GPU can use, you need **both**:

1. **`VkBuffer`** — describes a buffer resource: size, usage flags, sharing mode.
2. **`VkDeviceMemory`** — the actual allocated memory slab.

Then you **bind** them:

```text
vkBindBufferMemory(device, buffer, memory, offset=0)
```

Until bind succeeds, the buffer is an empty shell.

In our API this pair is `GpuBuffer`:

```text
GpuBuffer {
  buffer, memory, size, mapped, kind
}
```

## Usage flags (what the buffer is allowed to do)

When creating a buffer you set `VkBufferUsageFlags`. Think of them as permissions.

| Flag | Meaning for us |
|------|----------------|
| `TRANSFER_SRC` | May be source of a GPU copy |
| `TRANSFER_DST` | May be destination of a GPU copy |
| `STORAGE_BUFFER` | May be bound as an SSBO for compute shaders |

cthreads buffer kinds:

**Staging**

```text
TRANSFER_SRC | TRANSFER_DST
```

**DeviceLocal** (shader data)

```text
STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST
```

Why transfer bits on device-local? Because upload/download copy through them.

## Memory properties (where the bytes live)

After `vkCreateBuffer`, call `vkGetBufferMemoryRequirements`:

- `size` — how many bytes to allocate (may be larger than you asked; alignment)
- `alignment`
- `memoryTypeBits` — bitmask of legal memory type indices

Then look at the physical device's memory types (`vkGetPhysicalDeviceMemoryProperties`).

Each type has flags like:

| Flag | Meaning |
|------|---------|
| `HOST_VISIBLE` | CPU can map it and read/write |
| `HOST_COHERENT` | No manual flush/invalidate needed for visibility |
| `DEVICE_LOCAL` | Lives in GPU-friendly memory (often VRAM) |

### `find_memory_type` in cthreads

```text
for i in 0 .. memoryTypeCount-1:
  if bit i not set in type_bits: skip
  if (type.flags & requested) == requested: return i
throw if none
```

Requested properties:

- Staging: `HOST_VISIBLE | HOST_COHERENT`
- DeviceLocal: `DEVICE_LOCAL`

## Mapping (CPU pointer into device memory)

`vkMapMemory` returns a `void*` the CPU can `memcpy` into.

**Only do this for host-visible memory.**

cthreads rules:

- Staging: map once at create, keep `GpuBuffer.mapped` until destroy.
- DeviceLocal: **never** persistently map. Upload goes through staging.

Unmap with `vkUnmapMemory` before freeing memory.

## Create algorithm (what `create_buffer` does)

```text
1. Validate context.ready, size > 0
2. Choose usage + memory property flags from BufferKind
3. vkCreateBuffer
4. vkGetBufferMemoryRequirements
5. find_memory_type(...)
6. vkAllocateMemory (allocationSize = mem_reqs.size)
7. vkBindBufferMemory(..., offset 0)
8. If Staging: vkMapMemory -> mapped
9. Store caller size in GpuBuffer.size
```

On failure: destroy whatever was created (buffer/memory) before throwing.

## Destroy algorithm

```text
if mapped: unmap
destroy buffer
free memory
zero the GpuBuffer fields
```

Empty buffer (already null handles): no-op.

## Why size 0 is rejected

Vulkan implementations often dislike zero-sized buffers.
Empty Python lists are handled by **GpuPack** (no buffer / skip), not by creating a 0-byte `VkBuffer`.

## Sharing mode

cthreads uses `VK_SHARING_MODE_EXCLUSIVE`: one queue family owns the buffer.
We only use one compute/transfer family, so exclusive is correct and simpler.

## Common mistakes

1. Creating a buffer and forgetting to allocate/bind memory.
2. Mapping device-local memory that is not host-visible.
3. Using `GpuBuffer.size` wrong vs `mem_reqs.size` (we store the **caller** size; allocation may be larger — copies use caller size).
4. Destroying the device while buffers still exist.
5. Assuming `list` memory in Python is the GPU buffer — it is not; marshal must copy.
