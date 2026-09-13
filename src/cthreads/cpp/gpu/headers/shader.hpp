#pragma once

#include <cstddef>
#include <cstdint>

#include "shader_cache.hpp"

namespace cthreads::gpu {
struct Context;
}

/**
 * Stateless helpers that build and tear down ShaderCacheEntry Vulkan objects.
 *
 * create_entry turns SPIR-V + a binding count into the reusable pipeline row
 * stored in ShaderCache. It does not allocate per-launch descriptor sets or
 * bind a GpuPack; that is the launch / inflight path.
 *
 * Pass a ready Context with shader/pipeline create (and destroy) entry points.
 *
 * #### Technical terms:
 * - SPIR-V: binary compute shader (uint32 words). Built for tests as committed
 *   bytes or via shaderc; @Gpu emit feeds the same helper later.
 * - binding_count: STORAGE_BUFFER bindings for the binding convention
 *   ((scalars ? 1 : 0) + number of list SSBOs; at least 1 total).
 * - ShaderCacheEntry: module + set layout + pipeline layout + compute pipeline.
 */
namespace cthreads::gpu::shader {

/**
 * Creates a ShaderCacheEntry from SPIR-V and a binding-convention binding count.
 *
 * Builds, in order: shader module, descriptor set layout (bindings 0 ..
 * binding_count-1 as STORAGE_BUFFER), pipeline layout, compute pipeline.
 * On failure, destroys any objects already created and throws.
 *
 * Does not insert into ShaderCache; ShaderRegistry::register_spirv calls
 * create_entry then ShaderCache::add.
 *
 * #### Parameters:
 * - context: Context& = initialized GPU context with device and create entry points.
 * - spirv: const uint32_t* = SPIR-V code words (must be valid compute SPIR-V).
 * - spirv_word_count: size_t = number of uint32 words in spirv (byte size / 4).
 * - binding_count: uint32_t = number of SSBO bindings (must be >= 1).
 *
 * #### Returns:
 * - ShaderCacheEntry = owned Vulkan handles; move into ShaderCache::add or
 *   destroy_entry when abandoning the entry.
 *
 * #### Throws:
 * - runtime_error if context is not ready, spirv is null/empty, binding_count
 *   is 0, create entry points are missing, or any Vulkan create fails.
 *
 * #### Technical terms:
 * - VkShaderModule: Vulkan wrapper around SPIR-V bytes.
 * - VkDescriptorSetLayout: schema only; concrete descriptor sets are per launch.
 * - VkPipeline: compiled compute program ready for vkCmdBindPipeline.
 */
ShaderCacheEntry create_entry(
    cthreads::gpu::Context& context,
    const uint32_t* spirv,
    size_t spirv_word_count,
    uint32_t binding_count
);

/**
 * Destroys Vulkan objects owned by a ShaderCacheEntry and resets handles to null.
 *
 * Safe to call on an already-empty entry. Used by ShaderCache::clear and by
 * callers that abandon an entry before add.
 *
 * #### Parameters:
 * - context: Context& = same device that created the entry (needs destroy PFNs).
 * - entry: ShaderCacheEntry& = entry to destroy; left empty on return.
 *
 * #### Throws:
 * - Does not throw on destroy failure paths that only null handles when the
 *   device is already gone (shutdown edge cases).
 */
void destroy_entry(
    cthreads::gpu::Context& context,
    ShaderCacheEntry& entry
);

} // namespace cthreads::gpu::shader
