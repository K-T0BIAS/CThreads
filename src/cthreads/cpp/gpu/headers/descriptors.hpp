#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>

#include "pack.hpp"
#include "shader_cache.hpp"

namespace cthreads::gpu {
struct Context;
}

/**
 * Per-launch descriptor helpers for GpuPack wiring (namespace pack).
 *
 * The set layout and pipeline live on ShaderCacheEntry (created once per
 * symbol). Each job allocates a set, calls update_descriptors with that job's
 * GpuPack, then binds the set at dispatch time.
 *
 * #### Technical terms:
 * - descriptor set layout: schema of bindings (from create_entry / cache).
 * - descriptor pool: allocator from which concrete descriptor sets are taken.
 * - descriptor set: one instance of the layout; holds VkBuffer pointers for a launch.
 * - update_descriptors: writes binding i -> GpuPack buffer i (scalars then lists).
 */
namespace cthreads::gpu::pack {

/**
 * Pool that can allocate descriptor sets for a fixed STORAGE_BUFFER binding count.
 *
 * Created with FREE_DESCRIPTOR_SET_BIT so sets can be returned with free_set
 * when a job finishes. Destroy the pool only after all sets from it are freed
 * (or the device is shutting down and no jobs remain).
 *
 * #### Fields:
 * - pool: VkDescriptorPool = Vulkan pool handle; VK_NULL_HANDLE if empty.
 * - max_sets: uint32_t = how many sets this pool can still hold at create time.
 * - binding_count: uint32_t = STORAGE_BUFFER descriptors per set (binding convention).
 */
struct DescriptorPool {
    VkDescriptorPool pool = VK_NULL_HANDLE;
    uint32_t max_sets = 0;
    uint32_t binding_count = 0;
};

/**
 * Creates a descriptor pool for compute sets that follow the binding convention.
 *
 * Each set needs binding_count STORAGE_BUFFER descriptors. The pool can
 * allocate up to max_sets such sets. Sets may be freed individually.
 *
 * #### Parameters:
 * - context: Context& = initialized GPU context with descriptor pool entry points.
 * - binding_count: uint32_t = bindings per set (must match ShaderCacheEntry).
 * - max_sets: uint32_t = maximum sets this pool may allocate (>= 1).
 *
 * #### Returns:
 * - DescriptorPool = owned pool; caller must destroy_pool when done.
 *
 * #### Throws:
 * - runtime_error if context is not ready, counts are 0, PFNs are missing, or
 *   vkCreateDescriptorPool fails.
 */
DescriptorPool create_pool(
    cthreads::gpu::Context& context,
    uint32_t binding_count,
    uint32_t max_sets
);

/**
 * Destroys a descriptor pool and resets the DescriptorPool fields.
 *
 * All sets allocated from this pool become invalid. Prefer free_set on each
 * live set first when jobs may still hold sets.
 *
 * #### Parameters:
 * - context: Context& = same device that created the pool.
 * - pool: DescriptorPool& = pool to destroy; left empty on return.
 */
void destroy_pool(
    cthreads::gpu::Context& context,
    DescriptorPool& pool
);

/**
 * Allocates one descriptor set from the pool using the given set layout.
 *
 * The layout must match the pool's binding_count (same layout used in
 * ShaderCacheEntry::set_layout). The set is empty until update_descriptors.
 *
 * #### Parameters:
 * - context: Context& = initialized GPU context.
 * - pool: DescriptorPool& = pool with remaining capacity.
 * - set_layout: VkDescriptorSetLayout = layout from ShaderCacheEntry.
 *
 * #### Returns:
 * - VkDescriptorSet = allocated set (not a owned C++ type; free with free_set).
 *
 * #### Throws:
 * - runtime_error if the pool is empty/invalid, layout is null, or allocate fails.
 */
VkDescriptorSet allocate_set(
    cthreads::gpu::Context& context,
    DescriptorPool& pool,
    VkDescriptorSetLayout set_layout
);

/**
 * Returns a set to its pool. Safe no-op if set is VK_NULL_HANDLE.
 *
 * #### Parameters:
 * - context: Context& = same device as the pool.
 * - pool: DescriptorPool& = pool that allocated the set.
 * - set: VkDescriptorSet& = set to free; set to VK_NULL_HANDLE on return.
 *
 * #### Throws:
 * - runtime_error if free fails (pool must have been created with free bit).
 */
void free_set(
    cthreads::gpu::Context& context,
    DescriptorPool& pool,
    VkDescriptorSet& set
);

/**
 * Writes buffer bindings from a GpuPack into a descriptor set.
 *
 * If pack has a scalar buffer: binding 0 = scalars, 1..N = lists.
 * If not: binding 0..N-1 = lists (no scalar descriptor).
 * binding_count must equal (has_scalars ? 1 : 0) + container_slots.size().
 * Every written binding must have a non-null VkBuffer.
 *
 * Called by the launch path after allocate_set and before recording bind/dispatch.
 *
 * #### Parameters:
 * - context: Context& = initialized GPU context with vkUpdateDescriptorSets.
 * - set: VkDescriptorSet = destination set from allocate_set.
 * - binding_count: uint32_t = number of STORAGE_BUFFER bindings to write.
 * - pack: const GpuPack& = source buffers (optional scalars, then lists).
 *
 * #### Throws:
 * - runtime_error if set is null, binding_count mismatches the pack, any
 *   required buffer handle is null, or update entry points are missing.
 */
void update_descriptors(
    cthreads::gpu::Context& context,
    VkDescriptorSet set,
    uint32_t binding_count,
    const GpuPack& pack
);

/**
 * Convenience: update_descriptors using entry.binding_count.
 *
 * #### Parameters:
 * - context: Context& = initialized GPU context.
 * - set: VkDescriptorSet = destination set.
 * - entry: const ShaderCacheEntry& = cached layout metadata (binding_count).
 * - pack: const GpuPack& = source buffers.
 */
void update_descriptors(
    cthreads::gpu::Context& context,
    VkDescriptorSet set,
    const cthreads::gpu::shader::ShaderCacheEntry& entry,
    const GpuPack& pack
);

} // namespace cthreads::gpu::pack
