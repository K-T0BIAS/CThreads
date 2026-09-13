#include "../headers/descriptors.hpp"
#include "../headers/context.hpp"
#include "../headers/shader_cache.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace cthreads::gpu::pack {

DescriptorPool create_pool(
    Context& context,
    uint32_t binding_count,
    uint32_t max_sets
) {
    if (!context.ready || context.device == VK_NULL_HANDLE) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: create_pool needs an initialized "
            "device");
    }
    if (binding_count == 0 || max_sets == 0) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: create_pool binding_count and "
            "max_sets must be >= 1");
    }
    if (!context.vkCreateDescriptorPool) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: create_pool missing "
            "vkCreateDescriptorPool");
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = binding_count * max_sets;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    // Allow free_set per job when the launch completes.
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = max_sets;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    DescriptorPool out{};
    out.binding_count = binding_count;
    out.max_sets = max_sets;
    if (context.vkCreateDescriptorPool(
            context.device, &pool_info, nullptr, &out.pool) != VK_SUCCESS) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: vkCreateDescriptorPool failed");
    }
    return out;
}

void destroy_pool(Context& context, DescriptorPool& pool) {
    if (pool.pool == VK_NULL_HANDLE) {
        pool = DescriptorPool{};
        return;
    }
    if (context.device != VK_NULL_HANDLE && context.vkDestroyDescriptorPool) {
        context.vkDestroyDescriptorPool(context.device, pool.pool, nullptr);
    }
    pool = DescriptorPool{};
}

VkDescriptorSet allocate_set(
    Context& context,
    DescriptorPool& pool,
    VkDescriptorSetLayout set_layout
) {
    if (!context.ready || context.device == VK_NULL_HANDLE) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: allocate_set needs an initialized "
            "device");
    }
    if (pool.pool == VK_NULL_HANDLE) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: allocate_set pool is empty");
    }
    if (set_layout == VK_NULL_HANDLE) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: allocate_set set_layout is null");
    }
    if (!context.vkAllocateDescriptorSets) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: allocate_set missing "
            "vkAllocateDescriptorSets");
    }

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = pool.pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &set_layout;

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (context.vkAllocateDescriptorSets(context.device, &alloc_info, &set) !=
        VK_SUCCESS) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: vkAllocateDescriptorSets failed");
    }
    return set;
}

void free_set(Context& context, DescriptorPool& pool, VkDescriptorSet& set) {
    if (set == VK_NULL_HANDLE) {
        return;
    }
    if (pool.pool == VK_NULL_HANDLE) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: free_set pool is empty");
    }
    if (!context.vkFreeDescriptorSets || context.device == VK_NULL_HANDLE) {
        set = VK_NULL_HANDLE;
        return;
    }
    if (context.vkFreeDescriptorSets(context.device, pool.pool, 1, &set) !=
        VK_SUCCESS) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: vkFreeDescriptorSets failed");
    }
    set = VK_NULL_HANDLE;
}

void update_descriptors(
    Context& context,
    VkDescriptorSet set,
    uint32_t binding_count,
    const GpuPack& pack
) {
    if (!context.ready || context.device == VK_NULL_HANDLE) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: update_descriptors needs an "
            "initialized device");
    }
    if (set == VK_NULL_HANDLE) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: update_descriptors set is null");
    }
    if (binding_count == 0) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: update_descriptors binding_count "
            "must be >= 1");
    }
    const bool has_scalars = (pack.scalar_buffer.buffer != VK_NULL_HANDLE);
    const uint32_t expected =
        (has_scalars ? 1u : 0u) +
        static_cast<uint32_t>(pack.container_slots.size());
    if (binding_count != expected) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: update_descriptors binding_count "
            "must equal (scalars?1:0) + container_slots.size()");
    }
    if (!context.vkUpdateDescriptorSets) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: update_descriptors missing "
            "vkUpdateDescriptorSets");
    }

    // One buffer info + write per binding; infos must stay alive for the call.
    std::vector<VkDescriptorBufferInfo> buffer_infos(binding_count);
    std::vector<VkWriteDescriptorSet> writes(binding_count);

    for (uint32_t i = 0; i < binding_count; ++i) {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        if (has_scalars && i == 0) {
            buffer = pack.scalar_buffer.buffer;
            size = pack.scalar_buffer.size;
        } else {
            const uint32_t list_i = has_scalars ? (i - 1u) : i;
            const ContainerSlot& slot = pack.container_slots[list_i];
            buffer = slot.buffer.buffer;
            size = slot.buffer.size;
        }
        if (buffer == VK_NULL_HANDLE || size == 0) {
            throw std::runtime_error(
                "cthreads.gpu.GpuInvalidArgument: update_descriptors binding " +
                std::to_string(i) +
                " needs a non-empty GpuBuffer (empty pack slots not supported "
                "yet)");
        }

        buffer_infos[i] = {};
        buffer_infos[i].buffer = buffer;
        buffer_infos[i].offset = 0;
        buffer_infos[i].range = size;

        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }

    context.vkUpdateDescriptorSets(
        context.device,
        binding_count,
        writes.data(),
        0,
        nullptr);
}

void update_descriptors(
    Context& context,
    VkDescriptorSet set,
    const shader::ShaderCacheEntry& entry,
    const GpuPack& pack
) {
    update_descriptors(context, set, entry.binding_count, pack);
}

} // namespace cthreads::gpu::pack
