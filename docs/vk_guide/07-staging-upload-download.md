# 07 — Staging, upload, and download (final memory model)

## The problem

Shaders want **device-local** buffers (fast).
The CPU wants to `memcpy` from Python lists (host RAM).

Those are often **different** memory heaps. You cannot always map the SSBO and pretend it is a C array.

## The solution in cthreads

```text
UPLOAD:
  host pointer
    --memcpy-->  staging (host-visible, mapped)
    --GPU copy-->  device-local SSBO

DOWNLOAD:
  device-local SSBO
    --GPU copy-->  staging (mapped)
    --memcpy-->  host pointer
```

This is the permanent design for list/scalar marshal traffic.

## What `upload_buffer` does (in `memory.cpp` today)

Given a **device-local** `GpuBuffer`:

1. Check kind, null data, size bounds.
2. `create_buffer(..., Staging)` of `size` bytes.
3. `memcpy(staging.mapped, data, size)`.
4. Record/submit `vkCmdCopyBuffer(staging -> device)` and wait on a fence.
5. `destroy_buffer(staging)`.

`download_buffer` reverses the copy direction, then `memcpy` out of staging.

## One-shot command pool vs TransferEngine

In the current memory implementation, each upload/download creates a temporary:

- command pool
- command buffer
- fence

then destroys them after wait.

That is **correct** and matches the final *API*.

A later **TransferEngine** on `Context` can reuse one pool/fence/scratch staging for speed.
That is an optimization of the same path, not a new memory model.

## Barriers (intuition for later)

GPUs reorder work. Sometimes you need a **pipeline barrier** so "copy finished" happens before "shader reads."

For CPU-side round-trips (copy then CPU wait then CPU read), a fence wait is enough because the CPU does not read device memory until the GPU signaled.

When a shader runs in the same command buffer after a copy, insert a barrier between copy and dispatch. See the pipelines chapter.

## Why not host-visible SSBOs only?

It works on some integrated GPUs and for tiny demos.
It is the wrong long-term model for:

- discrete GPUs (VRAM vs sysmem)
- large lists
- matching how production Vulkan compute is written

cthreads does not ship a "map the SSBO forever" production path.

## End-to-end round-trip (what tests will prove)

```text
floats = [1, 2, 3, 4]
create device-local buffer (16 bytes)
upload_buffer(buf, floats)
download_buffer(buf, out)
assert out == floats
destroy
```

Same idea for a small scalar struct blob.

## How this becomes GpuPack

```text
GpuPack:
  scalars: GpuBuffer DeviceLocal   # bytes of std430 struct
  lists[]: GpuBuffer DeviceLocal   # one per list arg

upload pack: upload each piece
download pack: download each piece into Python objects
```

Staging is an implementation detail inside `upload_buffer` / `download_buffer` (or the transfer engine). Users never see it.
