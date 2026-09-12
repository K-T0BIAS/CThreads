#include "../headers/shader_cache.hpp"
#include "../headers/context.hpp"

#include <stdexcept>
#include <utility>

namespace cthreads::gpu::shader {

void destroy_entry(Context& context, ShaderCacheEntry& entry) {
    if (context.device == VK_NULL_HANDLE) {
        // Cannot destroy without a device; drop handle values only.
        entry.shader_module = VK_NULL_HANDLE;
        entry.set_layout = VK_NULL_HANDLE;
        entry.pipeline_layout = VK_NULL_HANDLE;
        entry.pipeline = VK_NULL_HANDLE;
        entry.binding_count = 0;
        return;
    }
    // Pipeline before layouts/module (children before parents).
    if (entry.pipeline != VK_NULL_HANDLE && context.vkDestroyPipeline) {
        context.vkDestroyPipeline(context.device, entry.pipeline, nullptr);
        entry.pipeline = VK_NULL_HANDLE;
    }
    if (entry.pipeline_layout != VK_NULL_HANDLE &&
        context.vkDestroyPipelineLayout) {
        context.vkDestroyPipelineLayout(
            context.device, entry.pipeline_layout, nullptr);
        entry.pipeline_layout = VK_NULL_HANDLE;
    }
    if (entry.set_layout != VK_NULL_HANDLE &&
        context.vkDestroyDescriptorSetLayout) {
        context.vkDestroyDescriptorSetLayout(
            context.device, entry.set_layout, nullptr);
        entry.set_layout = VK_NULL_HANDLE;
    }
    if (entry.shader_module != VK_NULL_HANDLE &&
        context.vkDestroyShaderModule) {
        context.vkDestroyShaderModule(
            context.device, entry.shader_module, nullptr);
        entry.shader_module = VK_NULL_HANDLE;
    }
    entry.binding_count = 0;
}

ShaderCacheEntry::ShaderCacheEntry(ShaderCacheEntry&& other) noexcept
    : shader_module(other.shader_module),
      set_layout(other.set_layout),
      pipeline_layout(other.pipeline_layout),
      pipeline(other.pipeline),
      binding_count(other.binding_count) {
    other.shader_module = VK_NULL_HANDLE;
    other.set_layout = VK_NULL_HANDLE;
    other.pipeline_layout = VK_NULL_HANDLE;
    other.pipeline = VK_NULL_HANDLE;
    other.binding_count = 0;
}

ShaderCacheEntry& ShaderCacheEntry::operator=(ShaderCacheEntry&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    // Assumes this entry's handles are already null or ownership was transferred.
    // clear() destroys before erase; move-assign is only for empty or stolen rows.
    shader_module = other.shader_module;
    set_layout = other.set_layout;
    pipeline_layout = other.pipeline_layout;
    pipeline = other.pipeline;
    binding_count = other.binding_count;
    other.shader_module = VK_NULL_HANDLE;
    other.set_layout = VK_NULL_HANDLE;
    other.pipeline_layout = VK_NULL_HANDLE;
    other.pipeline = VK_NULL_HANDLE;
    other.binding_count = 0;
    return *this;
}

ShaderCache& ShaderCache::getInstance() {
    static ShaderCache instance;
    return instance;
}

ShaderCache::~ShaderCache() {
    // Static teardown order vs Context is undefined. Shutdown must clear first
    // so handles are already null here; only drop the map.
    std::lock_guard<std::mutex> lock(_cache_mutex);
    _cache.clear();
}

const ShaderCacheEntry& ShaderCache::add(
    const std::string& key, ShaderCacheEntry&& entry) {
    std::lock_guard<std::mutex> lock(_cache_mutex);
    auto [it, inserted] =
        _cache.emplace(key, std::move(entry));
    if (!inserted) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: shader cache entry already "
            "exists: " +
            key);
    }
    return it->second;
}

const ShaderCacheEntry& ShaderCache::get(const std::string& key) {
    std::lock_guard<std::mutex> lock(_cache_mutex);
    const auto it = _cache.find(key);
    if (it == _cache.end()) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: shader not found in cache: " +
            key);
    }
    return it->second;
}

void ShaderCache::clear(Context& context) {
    std::lock_guard<std::mutex> lock(_cache_mutex);
    for (auto& [key, entry] : _cache) {
        (void)key;
        destroy_entry(context, entry);
    }
    _cache.clear();
}

} // namespace cthreads::gpu::shader
