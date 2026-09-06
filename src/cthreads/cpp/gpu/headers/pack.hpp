#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "memory.hpp"

namespace cthreads::gpu {
struct Context;
}

/**
 * Option 5 GpuPack: one device-local scalar SSBO plus one device-local SSBO per
 * list/container. Host traffic goes through memory:: upload/download helpers.
 *
 * This is a generic runtime bag of buffers. Per-kernel std430 layout and which
 * Python arg maps to which slot are marshal/codegen concerns, not this type.
 *
 * #### Technical terms:
 * - scalar SSBO: single buffer holding all packed scalar bytes for one launch.
 * - container slot: one list (or similar) argument; empty numel means no VkBuffer.
 * - upload / download: host <-> device-local copy via staging (see memory::).
 */
namespace cthreads::gpu::pack {

/**
 * How large one container slot should be at create time.
 *
 * #### Fields:
 * - elem_bytes: size_t = bytes per element (e.g. 4 for float or int32).
 * - numel: size_t = element count. 0 means no buffer is allocated for that slot.
 */
struct ContainerSpec {
    size_t elem_bytes = 0;
    size_t numel = 0;
};

/**
 * One list/container argument inside a GpuPack.
 *
 * #### Fields:
 * - buffer: GpuBuffer = device-local SSBO when numel > 0; empty handles when numel == 0.
 * - spec: ContainerSpec = elem size and count used at create (and for bounds checks).
 */
struct ContainerSlot {
    cthreads::gpu::memory::GpuBuffer buffer;
    ContainerSpec spec;
};

/**
 * Per-launch GPU argument pack (option 5).
 *
 * Owns Vulkan allocations until destroy_gpu_pack. Returning GpuPack by value
 * moves handles only; device bytes are not copied.
 *
 * #### Fields:
 * - scalar_buffer: GpuBuffer = device-local blob for all scalars (empty if scalar_bytes was 0).
 * - container_slots: vector of ContainerSlot = binding order 1..N matching create specs.
 */
struct GpuPack {
    cthreads::gpu::memory::GpuBuffer scalar_buffer;
    std::vector<ContainerSlot> container_slots;
};

/**
 * Allocates a GpuPack: device-local scalar buffer (if scalar_bytes > 0) and one
 * device-local buffer per container with numel > 0.
 *
 * Empty containers (numel == 0) keep a slot with no VkBuffer. Zero-size Vulkan
 * buffers are never created.
 *
 * #### Parameters:
 * - context: Context& = initialized GPU context with buffer/memory entry points.
 * - scalar_bytes: size_t = byte size of the scalar blob (0 = no scalar buffer).
 * - container_specs: vector<ContainerSpec> = per-list elem_bytes and numel, in binding order.
 *
 * #### Returns:
 * - GpuPack = owned buffers; caller must destroy_gpu_pack when done.
 *
 * #### Throws:
 * - runtime_error on bad specs (e.g. numel > 0 but elem_bytes == 0) or Vulkan alloc failure.
 */
GpuPack create_gpu_pack(
    cthreads::gpu::Context& context,
    size_t scalar_bytes,
    std::vector<ContainerSpec> container_specs
);

/**
 * Copies host scalar bytes into pack.scalar_buffer (host to device).
 *
 * #### Parameters:
 * - context: Context& = same device that created the pack.
 * - pack: GpuPack& = destination pack (must have a scalar buffer large enough).
 * - data: const void* = host source bytes.
 * - size: size_t = bytes to copy; must be <= scalar_buffer.size.
 *
 * #### Throws:
 * - runtime_error if there is no scalar buffer, data is null, size is invalid, or transfer fails.
 */
void upload_scalars(
    cthreads::gpu::Context& context,
    GpuPack& pack,
    const void* data,
    size_t size
);

/**
 * Copies host bytes into one container slot (host to device).
 *
 * #### Parameters:
 * - context: Context& = same device that created the pack.
 * - pack: GpuPack& = destination pack.
 * - index: size_t = container_slots index.
 * - data: const void* = host source bytes.
 * - size: size_t = bytes to copy; must equal elem_bytes * numel for that slot.
 *
 * #### Throws:
 * - runtime_error if index is out of range, the slot is empty (numel == 0), data is null,
 *   size mismatches, or transfer fails.
 */
void upload_container(
    cthreads::gpu::Context& context,
    GpuPack& pack,
    size_t index,
    const void* data,
    size_t size
);

/**
 * Uploads every non-empty container slot from parallel host pointers.
 *
 * data[i] / sizes[i] correspond to container_slots[i]. Slots with numel == 0
 * are skipped (pointer/size for those entries are ignored).
 *
 * #### Parameters:
 * - context: Context& = same device that created the pack.
 * - pack: GpuPack& = destination pack.
 * - data: vector of host pointers, one per slot.
 * - sizes: vector of byte counts, one per slot (same length as data and slots).
 *
 * #### Throws:
 * - runtime_error if lengths mismatch pack.container_slots or any per-slot upload fails.
 */
void upload_containers(
    cthreads::gpu::Context& context,
    GpuPack& pack,
    const std::vector<const void*>& data,
    const std::vector<size_t>& sizes
);

/**
 * Copies pack.scalar_buffer into host memory (device to host).
 *
 * #### Parameters:
 * - context: Context& = same device that created the pack.
 * - pack: GpuPack& = source pack.
 * - data: void* = host destination bytes.
 * - size: size_t = bytes to copy; must be non-zero and <= scalar_buffer.size.
 *
 * #### Throws:
 * - runtime_error if there is no scalar buffer, data is null, size is invalid, or transfer fails.
 */
void download_scalars(
    cthreads::gpu::Context& context,
    GpuPack& pack,
    void* data,
    size_t size
);

/**
 * Copies one container slot into host memory (device to host).
 *
 * #### Parameters:
 * - context: Context& = same device that created the pack.
 * - pack: GpuPack& = source pack.
 * - index: size_t = container_slots index.
 * - data: void* = host destination bytes.
 * - size: size_t = bytes to copy; must be <= container_slots[index].buffer.size.
 *
 * #### Throws:
 * - runtime_error if index is out of range, the slot is empty (numel == 0), data is null,
 *   size mismatches, or transfer fails.
 */
void download_container(
    cthreads::gpu::Context& context,
    GpuPack& pack,
    size_t index,
    void* data,
    size_t size
);

/**
 * Downloads every non-empty container slot into parallel host pointers.
 *
 * Slots with numel == 0 are skipped. data/sizes length must match container_slots.
 *
 * #### Parameters:
 * - context: Context& = same device that created the pack.
 * - pack: GpuPack& = source pack.
 * - data: vector of host destinations, one per slot.
 * - sizes: vector of byte counts, one per slot (same length as data and slots).
 *
 * #### Throws:
 * - runtime_error if lengths mismatch pack.container_slots or any per-slot download fails.
 */
void download_containers(
    cthreads::gpu::Context& context,
    GpuPack& pack,
    std::vector<void*>& data,
    const std::vector<size_t>& sizes
);

/**
 * Destroys all buffers owned by the pack and clears its fields.
 *
 * Safe to call on an already-empty pack. Does not destroy the Context.
 *
 * #### Parameters:
 * - context: Context& = same device that created the pack.
 * - pack: GpuPack& = pack to free; left empty after return.
 */
void destroy_gpu_pack(cthreads::gpu::Context& context, GpuPack& pack);

} // namespace cthreads::gpu::pack
