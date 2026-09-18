# GpuPack

Source: `src/cthreads/cpp/gpu/headers/pack.hpp`, `src/cthreads/cpp/gpu/impl/pack.cpp`.

Namespace: `cthreads::gpu::pack`.

Depends on: [Context](./context.md), [Memory](./memory.md).

Descriptor pool and update helpers live in the same namespace but in separate files. See [Descriptors](./descriptors.md).

## Purpose

A `GpuPack` is the per-launch bag of GPU buffers for one compute dispatch under the binding convention:

- One device-local scalar storage buffer (binding 0), or none if there are no scalar bytes.
- One device-local storage buffer per list/container argument (bindings 1..N).

The pack does not know Python argument names or std430 field offsets. Marshal and codegen decide how to flatten scalars into the scalar blob and which list is slot `i`. The pack only owns buffers and moves bytes.

## Technical terms

- Binding convention: one scalar SSBO plus one SSBO per list, with no buffer device addresses inside the scalar block.
- SSBO: storage buffer object; a shader-readable and writable buffer bound through descriptors.
- Container slot: one list-like argument inside the pack.
- `numel`: number of elements in a slot. Zero means "no Vulkan buffer for this slot."
- std430: a GLSL/SPIR-V memory layout rule for how fields pack in a storage buffer. Marshal must match it; this module only stores opaque bytes.

## Struct `ContainerSpec`

Create-time size description for one list slot.

- `elem_bytes`: bytes per element (4 for `float` or 32-bit `int`).
- `numel`: element count. If zero, create keeps an empty slot and does not call `vkCreateBuffer` with size 0.

## Struct `ContainerSlot`

- `buffer`: device-local `GpuBuffer` when `numel > 0`; empty handles otherwise.
- `spec`: the `ContainerSpec` used at create time (also used for upload size checks).

## Struct `GpuPack`

- `scalar_buffer`: device-local blob for all packed scalars. Empty if `scalar_bytes` was 0 at create.
- `container_slots`: vector in binding order for bindings 1..N.

Returning a `GpuPack` by value moves handles only. It does not clone GPU memory.

## Function `create_gpu_pack`

Allocates the pack on a ready Context.

Behavior:

- If `scalar_bytes > 0`, create a device-local scalar buffer of that size.
- For each container spec, append a slot. If `numel > 0`, require `elem_bytes > 0` and create a device-local buffer of `elem_bytes * numel`. If `numel == 0`, keep a slot with no buffer.

Zero-size Vulkan buffers are never created.

## Upload helpers

### `upload_scalars`

Copies host bytes into `pack.scalar_buffer` via [memory upload](./memory.md).

### `upload_container`

Copies host bytes into one non-empty slot. Size must match the slot's byte size.

### `upload_containers`

Uploads every non-empty slot from parallel host pointers. Empty slots are skipped.

## Download helpers

### `download_scalars` / `download_container` / `download_containers`

Mirror the upload helpers in the device-to-host direction. Results land in caller-owned host memory. `SpawnedGpuKernel::join` uses these for ref-list writeback (see [Module / launch](./module.md)).

## Function `destroy_gpu_pack`

Destroys every non-null buffer through `memory::destroy_buffer` and clears the pack. Safe on an already-empty pack. Does not shut down the Context.

## Empty lists

Empty lists are first-class in the pack: the slot exists so binding indices stay stable, but there is no `VkBuffer`. Descriptor update currently rejects null buffers (see [Descriptors](./descriptors.md)). A future dummy SSBO may fill empty bindings; until then, smoke launches should use non-empty lists for every binding the shader declares.

## How this fits the launch path

```text
create_gpu_pack
upload_scalars / upload_containers
allocate descriptor set + update_descriptors(pack)   // descriptors.hpp
record CB: barrier + bind pipeline/set + dispatch    // module.cpp
vkQueueSubmit(..., fence)
join: wait fence -> barrier -> download_* into kept Python lists -> release
destroy_gpu_pack
```

## Testing today

`_ext.gpu.testing` exposes pack round-trip helpers (float and int packs) and
`register_smoke_saxpy` (SPIR-V cache only). Launch/join use product
`_ext.gpu.launch_gpu_kernel`. Pytest: `tests/unit/test_gpu_pack.py`,
`tests/unit/test_gpu_shader.py`.
