#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vulkan/vulkan.h>

namespace cthreads::gpu {
struct Context;
}

namespace cthreads::gpu::testing {
struct ShaderCacheTestAccess;
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
 * - set_layout: VkDescriptorSetLayout = bindings 0 scalars, 1..N lists.
 * - pipeline_layout: VkPipelineLayout = layout used to create the compute pipeline.
 * - pipeline: VkPipeline = compute pipeline ready to bind.
 * - binding_count: uint32_t = number of STORAGE_BUFFER bindings (1 + list count).
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
 * Process-wide map of kernel symbol -> reusable pipeline objects.
 *
 * Writers (registry, later) call add. Everyone else only get / clear.
 * Entries are immutable after insert; clear runs on Context shutdown.
 */
class ShaderCache {
private:
    std::unordered_map<std::string, ShaderCacheEntry> _cache;
    std::mutex _cache_mutex;

    ShaderCache() = default;
    ~ShaderCache();

    // Registry-only once a friend exists. Duplicate key throws.
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
    // Test-only access to private add (see gpu/testing/shader_smoke.cpp).
    friend struct cthreads::gpu::testing::ShaderCacheTestAccess;
};

} // namespace cthreads::gpu::shader
