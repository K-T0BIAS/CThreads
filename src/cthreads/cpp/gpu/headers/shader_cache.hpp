#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vulkan/vulkan.h>

namespace cthreads::gpu {
struct Context;
}

/**
 * Shader cache: reusable per-symbol Vulkan pipeline objects.
 *
 * Binding schema and compute pipelines are fixed for a kernel symbol. Per-launch
 * buffers and descriptor sets live on the job / inflight path, not here.
 *
 * #### Technical terms:
 * - Context: process-wide Vulkan connection (device, queue, loaded entry points).
 * - SPIR-V: binary shader IR the driver consumes (see create_entry in shader.hpp).
 * - descriptor set layout: schema of SSBO bindings (0 scalars, 1..N lists).
 * - compute pipeline: compiled shader + layout ready to bind and dispatch.
 */
namespace cthreads::gpu::shader {

/**
 * Per-kernel Vulkan objects reused across launches of the same symbol.
 *
 * Move-only: Vulkan handles must not be copied (would double-destroy).
 *
 * #### Fields:
 * - shader_module: VkShaderModule = SPIR-V module (may be null after pipeline create).
 * - set_layout: VkDescriptorSetLayout = bindings 0..N-1 (scalars at 0 if present, then lists).
 * - pipeline_layout: VkPipelineLayout = layout used to create the compute pipeline.
 * - pipeline: VkPipeline = compute pipeline ready to bind.
 * - binding_count: uint32_t = STORAGE_BUFFER bindings ((scalars?1:0) + list count).
 */
struct ShaderCacheEntry {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t binding_count = 0;

    ShaderCacheEntry() = default;
    ShaderCacheEntry(const ShaderCacheEntry&) = delete;
    ShaderCacheEntry& operator=(const ShaderCacheEntry&) = delete;
    ShaderCacheEntry(ShaderCacheEntry&& other) noexcept;
    ShaderCacheEntry& operator=(ShaderCacheEntry&& other) noexcept;
};

/**
 * Sole writer into ShaderCache (create_entry + add).
 *
 * Python reaches this via `_ext.gpu.register_shader`. Tests and future native
 * callers use the same type - do not friend other writers or call add() directly.
 */
struct ShaderRegistry {
    /**
     * Build a pipeline entry from SPIR-V and insert it under `symbol`.
     *
     * #### Parameters:
     * - context: Context& = ready GPU context
     * - symbol: string = ShaderCache key (kernel name)
     * - spirv: const uint32_t* = SPIR-V words
     * - spirv_word_count: size_t = word count
     * - binding_count: uint32_t = SSBO binding count (>= 1)
     *
     * #### Returns:
     * - const ShaderCacheEntry& = entry stored in the cache
     *
     * #### Throws:
     * - runtime_error = create_entry failure or duplicate symbol
     */
    static const ShaderCacheEntry& register_spirv(
        cthreads::gpu::Context& context,
        const std::string& symbol,
        const uint32_t* spirv,
        size_t spirv_word_count,
        uint32_t binding_count
    );
};

/**
 * Process-wide map of kernel symbol -> reusable pipeline objects.
 *
 * Writers: ShaderRegistry only. Everyone else: get / clear.
 * Entries are immutable after insert; clear runs on Context shutdown.
 */
class ShaderCache {
private:
    std::unordered_map<std::string, ShaderCacheEntry> _cache;
    std::mutex _cache_mutex;

    ShaderCache() = default;
    ~ShaderCache();

    // ShaderRegistry-only. Duplicate key throws.
    const ShaderCacheEntry& add(const std::string& key, ShaderCacheEntry&& entry);

public:
    static ShaderCache& getInstance();

    ShaderCache(const ShaderCache&) = delete;
    ShaderCache& operator=(const ShaderCache&) = delete;
    ShaderCache(ShaderCache&&) = delete;
    ShaderCache& operator=(ShaderCache&&) = delete;

    // Throws if the symbol is not registered.
    const ShaderCacheEntry& get(const std::string& key);

    // Destroy all Vulkan objects on the entry, then empty the map.
    // Call from Context shutdown before destroying the logical device.
    void clear(cthreads::gpu::Context& context);

    friend struct cthreads::gpu::Context; // shutdown / future access
    friend struct ShaderRegistry;
};

} // namespace cthreads::gpu::shader
