# 08 — Command buffers, queues, and fences

## Why command buffers exist

The GPU does not execute your C++ line by line.
You **record** a list of GPU commands, then **submit** that list to a queue.

Think of a command buffer as a recipe card:

```text
Begin
  CopyBuffer A -> B
  (later) Bind pipeline
  (later) Dispatch
End
Submit to queue
```

Recording is cheap CPU work. Executing happens on the GPU, possibly later.

## Objects involved

| Object | Role |
|--------|------|
| `VkCommandPool` | Allocator/owner of command buffers for one queue family |
| `VkCommandBuffer` | The recorded recipe |
| `VkQueue` | Where recipes are submitted |
| `VkFence` | CPU-waitable "this submit finished" flag |

## Record / submit / wait pattern (copy)

This is exactly what our `copy_buffer_and_wait` helper does:

```text
1. vkCreateCommandPool (family = context.queue_family)
2. vkAllocateCommandBuffers (one primary buffer)
3. vkBeginCommandBuffer (ONE_TIME_SUBMIT)
4. vkCmdCopyBuffer(src, dst, region)
5. vkEndCommandBuffer
6. vkCreateFence
7. vkQueueSubmit(queue, cmd, fence)
8. vkWaitForFences(..., UINT64_MAX)
9. destroy fence, free command buffer, destroy pool
```

### Primary vs secondary

cthreads uses **primary** command buffers (can be submitted directly).
Secondary buffers are for advanced recording reuse; ignore for now.

### One-time submit

`VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT` means: we will record, submit once, then throw away / reset.
Perfect for copies.

## Fences vs semaphores (only fences matter for us now)

| Primitive | Who waits | Typical use |
|-----------|-----------|-------------|
| **Fence** | CPU | `join`, wait for copy before memcpy |
| **Semaphore** | GPU | Sync between queue submits / present (graphics) |

cthreads launch/wait uses **fences**.
Semaphores can be ignored until multi-queue or swapchains (not part of this compute path).

## Reset and reuse (launch hot-path)

Instead of destroy/recreate every time:

- Keep a pool with `RESET` flag or reset individual buffers.
- `vkResetFences` before reuse.
- Re-record and submit again.

Reusing pools and cached pipelines is how launches stay cheap.

## Queue submit is asynchronous

After `vkQueueSubmit` returns `VK_SUCCESS`, the GPU might still be working.
Only after the fence signals is it safe to:

- destroy the buffers you copied from/to (if you are done with them)
- `memcpy` from staging after a download copy
- tell Python the job finished

`GpuJob.join()` waits on the fence for the dispatch submit.

## Ordering: destroy vs in-flight work

Never destroy a buffer or pool while the GPU still has commands referencing it.
Wait for fences first, then destroy.

The upload helper waits, then destroys staging — correct order.

## Transient pools

`VK_COMMAND_POOL_CREATE_TRANSIENT_BIT` hints that command buffers are short-lived.
cthreads uses that for one-shot copies.

## How this connects to compute dispatch (preview)

A full kernel launch command buffer looks like:

```text
Begin
  (optional) barriers after uploads
  Bind pipeline
  Bind descriptor set (buffers)
  Push constants (optional)
  vkCmdDispatch(groups_x, groups_y, groups_z)
End
Submit + fence
```

Same machinery as copy; different commands inside.
