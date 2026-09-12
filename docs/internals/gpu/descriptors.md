# Descriptors (pack namespace)

Source: `src/cthreads/cpp/gpu/headers/descriptors.hpp`, `src/cthreads/cpp/gpu/impl/descriptors.cpp`.

Namespace: `cthreads::gpu::pack` (same as GpuPack; separate files for clarity).

Depends on: [Context](./context.md), [Pack](./pack.md), [Shader](./shader.md) (for set layouts and `ShaderCacheEntry`).

## Purpose in plain language

A compute shader does not receive C++ pointers. It declares numbered bindings, for example "binding 0 is my scalar block" and "binding 1 is list x." On the CPU you own `VkBuffer` handles inside a `GpuPack`. Descriptors are the table that connects those two worlds:

```text
shader binding 0  ->  pack.scalar_buffer
shader binding 1  ->  pack.container_slots[0]
shader binding 2  ->  pack.container_slots[1]
...
```

There are two layers:

1. Descriptor set layout: the schema (which binding numbers exist and that each is a storage buffer). Created once per kernel in `shader::create_entry` and stored on `ShaderCacheEntry`.
2. Descriptor set: one filled-in instance of that schema for one launch. Created from a pool, then updated to point at this launch's pack buffers.

## Technical terms

- Descriptor: one slot in the table (for us, always a storage buffer reference).
- Descriptor set layout: immutable schema of bindings for a pipeline.
- Descriptor set: concrete table instance you bind before dispatch.
- Descriptor pool: allocator that vends descriptor sets (similar in spirit to a memory pool).
- Storage buffer descriptor: a descriptor type that points at a `VkBuffer` used as an SSBO.
- `vkUpdateDescriptorSets`: Vulkan call that writes buffer handles into a set.
- Binding number: integer the shader and the CPU agree on (0 for scalars, then lists).

## Struct `DescriptorPool`

Fields:

- `pool`: Vulkan `VkDescriptorPool` handle.
- `max_sets`: how many sets the pool was created to hold.
- `binding_count`: how many storage buffer descriptors each set needs (must match the shader cache entry).

The pool is created with `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT` so individual sets can be returned with `free_set` when a job finishes.

## Function `create_pool`

Creates a pool for compute sets that follow the binding convention.

Parameters:

- `binding_count`: storage buffers per set (1 + number of list slots).
- `max_sets`: capacity.

Internally it declares one pool size entry of type `STORAGE_BUFFER` with `descriptorCount = binding_count * max_sets`, then calls `vkCreateDescriptorPool`.

## Function `destroy_pool`

Destroys the Vulkan pool and clears the struct. All sets allocated from the pool become invalid. Prefer freeing live sets first when jobs still hold them.

## Function `allocate_set`

Allocates one `VkDescriptorSet` from the pool using a `VkDescriptorSetLayout` (normally `ShaderCacheEntry::set_layout`). The set is empty until `update_descriptors` runs.

Key call: `vkAllocateDescriptorSets`.

## Function `free_set`

Returns a set to the pool via `vkFreeDescriptorSets`. Safe no-op if the set handle is already null. After success, the caller's set handle is set to `VK_NULL_HANDLE`.

## Function `update_descriptors` (binding count overload)

Writes the pack into the set.

Rules:

- `binding_count` must equal `1 + pack.container_slots.size()`.
- Binding 0 uses `pack.scalar_buffer`.
- Binding `i` for `i >= 1` uses `pack.container_slots[i - 1]`.
- Every written binding must have a non-null buffer and non-zero size. Empty pack slots are not supported yet.

For each binding it builds a `VkDescriptorBufferInfo` (buffer, offset 0, range = size) and a `VkWriteDescriptorSet` of type `STORAGE_BUFFER`, then calls `vkUpdateDescriptorSets` once for all writes.

## Function `update_descriptors` (entry overload)

Convenience wrapper that uses `entry.binding_count` from a `ShaderCacheEntry`.

## Who calls these

The launch path (`launch_gpu_kernel` today via tests; later `gpu()`):

```text
ShaderCache.get(symbol) -> entry
create_pool / allocate_set(entry.set_layout)
update_descriptors(set, entry, pack)
record CB: barrier -> bind pipeline -> bind set -> dispatch
vkQueueSubmit(..., fence)
// join: wait fence -> download ref lists -> release
```

The shader cache never updates descriptors. It only stores the layout schema.

## Key Vulkan calls summary

| Call | Role |
|-|-|
| `vkCreateDescriptorPool` | Create the allocator |
| `vkDestroyDescriptorPool` | Destroy the allocator |
| `vkAllocateDescriptorSets` | Get one set matching a layout |
| `vkFreeDescriptorSets` | Return a set to the pool |
| `vkUpdateDescriptorSets` | Point bindings at `VkBuffer`s |

## Current limitation: empty lists

`GpuPack` allows empty container slots without a buffer. `update_descriptors` throws if a required binding has a null buffer, because standard Vulkan does not accept a null buffer descriptor without special null-descriptor features. Until a process-wide dummy SSBO exists, test kernels should use non-empty buffers for every binding they declare.

## What comes after update

Descriptors alone do not run the shader. [Module / launch](./module.md) records the command buffer, submits with a fence, and `join` downloads ref lists into Python.
