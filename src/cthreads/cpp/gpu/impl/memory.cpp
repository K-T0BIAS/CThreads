#include "../headers/memory.hpp"
#include "../headers/context.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <mutex>

namespace cthreads::gpu::memory {

namespace {

/**
* Ensure that the global context singleton has been initialized and is ready to use.
*
* #### Parameters:
* - context: The global context singleton (holds all dynamically linked vulkan function ptrs aswell as the device information)
* - where: The name of the function that is calling this helper. This is used to construct the error message.
*
* #### Throws:
* - std::runtime_error: If the context is not ready.
*/
void require_ready(const Context& context, const char* where) {
    if (!context.ready || context.device == VK_NULL_HANDLE) {
        throw std::runtime_error(
            std::string("cthreads.gpu.VulkanInitFailed: ") + where +
            " needs an initialized device");
    }
}

void require_transfer_engine(const Context& context, const char* where) {
    // assumes the engine mutex is locked
    if (context.transfer_engine.command_pool == VK_NULL_HANDLE ||
        context.transfer_engine.fence == VK_NULL_HANDLE) {
        throw std::runtime_error(
            std::string("cthreads.gpu.VulkanInitFailed: ") + where +
            " needs an initialized TransferEngine (pool + fence)");
    }
}

// GPU copy via Context TransferEngine pool + fence, then CPU wait.
// Caller must hold context.transfer_engine_mutex for the whole call.
void copy_buffer_and_wait(
    Context& context,
    VkBuffer src,
    VkBuffer dst,
    VkDeviceSize size
) {
    require_transfer_engine(context, "copy_buffer_and_wait");
    if (!context.vkAllocateCommandBuffers || !context.vkFreeCommandBuffers ||
        !context.vkBeginCommandBuffer || !context.vkEndCommandBuffer ||
        !context.vkCmdCopyBuffer || !context.vkQueueSubmit ||
        !context.vkWaitForFences || !context.vkResetFences || !context.queue) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: copy_buffer_and_wait missing "
            "command/fence entry points or queue");
    }

    TransferEngine& te = context.transfer_engine;
    VkCommandPool pool = te.command_pool;
    VkFence fence = te.fence;

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (context.vkAllocateCommandBuffers(context.device, &alloc_info, &cmd) !=
        VK_SUCCESS) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: vkAllocateCommandBuffers failed");
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (context.vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
        context.vkFreeCommandBuffers(context.device, pool, 1, &cmd);
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: vkBeginCommandBuffer failed");
    }

    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = size;
    context.vkCmdCopyBuffer(cmd, src, dst, 1, &region);

    if (context.vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        context.vkFreeCommandBuffers(context.device, pool, 1, &cmd);
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: vkEndCommandBuffer failed");
    }

    // Fence is created signaled and left signaled after each wait; reset for reuse.
    if (context.vkResetFences(context.device, 1, &fence) != VK_SUCCESS) {
        context.vkFreeCommandBuffers(context.device, pool, 1, &cmd);
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: vkResetFences failed");
    }

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (context.vkQueueSubmit(context.queue, 1, &submit, fence) != VK_SUCCESS) {
        context.vkFreeCommandBuffers(context.device, pool, 1, &cmd);
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: vkQueueSubmit failed");
    }

    if (context.vkWaitForFences(
            context.device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        context.vkFreeCommandBuffers(context.device, pool, 1, &cmd);
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: vkWaitForFences failed");
    }

    context.vkFreeCommandBuffers(context.device, pool, 1, &cmd);
}

} // namespace

uint32_t find_memory_type(
    Context& context,
    uint32_t type_bits,
    VkMemoryPropertyFlags properties
) {
    // ensure the context is setup correctly
    if (context.physical_device == VK_NULL_HANDLE ||
        !context.vkGetPhysicalDeviceMemoryProperties) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: find_memory_type needs a ready "
            "physical device and vkGetPhysicalDeviceMemoryProperties"
        );
    }

    VkPhysicalDeviceMemoryProperties memory_props{}; // get mem properties from the ctx device
    context.vkGetPhysicalDeviceMemoryProperties(
        context.physical_device, &memory_props);

    for (uint32_t i = 0; i < memory_props.memoryTypeCount; ++i) {
        const bool allowed_by_buffer = (type_bits & (1u << i)) != 0; // check if the ith bit is 1
        if (!allowed_by_buffer) {
            continue;
        }
        const VkMemoryPropertyFlags flags =
            memory_props.memoryTypes[i].propertyFlags;
        if ((flags & properties) == properties) { // checck if all properties bits are set in flags
            return i;
        }
    }

    throw std::runtime_error(
        "cthreads.gpu.VulkanInitFailed: no memory type matches type_bits and "
        "requested properties");
}

GpuBuffer create_buffer(
    Context& context,
    VkDeviceSize size,
    BufferKind kind
) {
    require_ready(context, "create_buffer"); // ensure the ctx is ready (the device must be initialized)
    if (size == 0) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: create_buffer size must be greater than 0");
    }
    if (!context.vkCreateBuffer || !context.vkGetBufferMemoryRequirements ||
        !context.vkAllocateMemory || !context.vkBindBufferMemory ||
        !context.vkDestroyBuffer || !context.vkFreeMemory) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: create_buffer missing Vulkan entry points");
    }

    // setup flags based of the buffers kind
    VkBufferUsageFlags usage = 0;
    VkMemoryPropertyFlags mem_props = 0;
    if (kind == BufferKind::Staging) {
        // Host can memcpy here; GPU copies to/from device-local buffers.
        usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        mem_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    } else {
        // Shader SSBO + copies for marshal upload/download.
        usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        mem_props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }

    GpuBuffer out{}; // create gpu buffer struct (internally owns vulkan buffer and memory handles)
    out.kind = kind;

    // create the vk buffer
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (context.vkCreateBuffer(context.device, &bci, nullptr, &out.buffer) !=
        VK_SUCCESS) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: vkCreateBuffer failed");
    }

    VkMemoryRequirements mem_reqs{}; // get the memory requirements for the buffer
    context.vkGetBufferMemoryRequirements(context.device, out.buffer, &mem_reqs);

    const uint32_t memory_type = // find the memory type that matches the requirements
        find_memory_type(context, mem_reqs.memoryTypeBits, mem_props);

    // allocate the memory (allocated to the gpuBuffers memory handle)
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mem_reqs.size;
    mai.memoryTypeIndex = memory_type;
    if (context.vkAllocateMemory(context.device, &mai, nullptr, &out.memory) !=
        VK_SUCCESS) {
        context.vkDestroyBuffer(context.device, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        throw std::runtime_error(
            "cthreads.gpu.VulkanOutOfMemory: vkAllocateMemory failed");
    }

    if (context.vkBindBufferMemory(context.device, out.buffer, out.memory, 0) !=
        VK_SUCCESS) {
        context.vkFreeMemory(context.device, out.memory, nullptr);
        context.vkDestroyBuffer(context.device, out.buffer, nullptr);
        out.memory = VK_NULL_HANDLE;
        out.buffer = VK_NULL_HANDLE;
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: vkBindBufferMemory failed");
    }

    // if the buffer is for staging then also set the cpu pointer to the memory in buffer.mapped
    if (kind == BufferKind::Staging) {
        if (!context.vkMapMemory) { // if the mem isnt mapped sth went wrong -> free and throw
            context.vkFreeMemory(context.device, out.memory, nullptr);
            context.vkDestroyBuffer(context.device, out.buffer, nullptr);
            out.memory = VK_NULL_HANDLE;
            out.buffer = VK_NULL_HANDLE;
            throw std::runtime_error(
                "cthreads.gpu.VulkanInitFailed: vkMapMemory missing");
        }
        if (context.vkMapMemory( // map the memory to the cpu pointer
                context.device, out.memory, 0, mem_reqs.size, 0, &out.mapped) !=
            VK_SUCCESS) {
            context.vkFreeMemory(context.device, out.memory, nullptr);
            context.vkDestroyBuffer(context.device, out.buffer, nullptr);
            out.memory = VK_NULL_HANDLE;
            out.buffer = VK_NULL_HANDLE;
            out.mapped = nullptr;
            throw std::runtime_error(
                "cthreads.gpu.VulkanInitFailed: vkMapMemory failed");
        }
    }

    out.size = size;
    return out;
}

void destroy_buffer(Context& context, GpuBuffer& buffer) {
    // check if the buffer is already destroyed
    if (buffer.buffer == VK_NULL_HANDLE && buffer.memory == VK_NULL_HANDLE) {
        buffer = GpuBuffer{};
        return;
    }
    require_ready(context, "destroy_buffer"); // ensure the ctx is ready (the device must be initialized)

    // if the buffer is mapped then unmap it
    if (buffer.mapped != nullptr && context.vkUnmapMemory &&
        buffer.memory != VK_NULL_HANDLE) {
        context.vkUnmapMemory(context.device, buffer.memory);
        buffer.mapped = nullptr;
    }
    // destroy the vk buffer
    if (buffer.buffer != VK_NULL_HANDLE && context.vkDestroyBuffer) {
        context.vkDestroyBuffer(context.device, buffer.buffer, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
    }
    // free the vk memory
    if (buffer.memory != VK_NULL_HANDLE && context.vkFreeMemory) {
        context.vkFreeMemory(context.device, buffer.memory, nullptr);
        buffer.memory = VK_NULL_HANDLE;
    }
    buffer.size = 0;
    buffer.kind = BufferKind::DeviceLocal;
}

namespace {

// Grow-only host-visible scratch on the TransferEngine. Never shrinks until
// Context shutdown. Caller must hold context.transfer_engine_mutex.
void ensure_staging(Context& context, VkDeviceSize size) {
    TransferEngine& te = context.transfer_engine;
    if (te.staging.buffer != VK_NULL_HANDLE && te.staging.size >= size &&
        te.staging.mapped != nullptr) {
        return;
    }
    if (te.staging.buffer != VK_NULL_HANDLE) {
        destroy_buffer(context, te.staging);
    }
    te.staging = create_buffer(context, size, BufferKind::Staging);
}

} // namespace

void upload_buffer(
    Context& context,
    GpuBuffer& buffer,
    const void* data,
    VkDeviceSize size
) {
    require_ready(context, "upload_buffer");
    if (buffer.kind != BufferKind::DeviceLocal ||
        buffer.buffer == VK_NULL_HANDLE) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: upload_buffer requires a device-local "
            "GpuBuffer");
    }
    if (data == nullptr) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: upload_buffer data is null");
    }
    if (size == 0 || size > buffer.size) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: upload_buffer size invalid");
    }

    // One lock for engine check, staging grow, host memcpy, and GPU copy.
    std::lock_guard<std::mutex> lock(context.transfer_engine_mutex);
    require_transfer_engine(context, "upload_buffer");
    ensure_staging(context, size);
    GpuBuffer& staging = context.transfer_engine.staging;
    std::memcpy(staging.mapped, data, static_cast<size_t>(size));
    copy_buffer_and_wait(context, staging.buffer, buffer.buffer, size);
}

void download_buffer(
    Context& context,
    GpuBuffer& buffer,
    void* data,
    VkDeviceSize size
) {
    require_ready(context, "download_buffer");
    if (buffer.kind != BufferKind::DeviceLocal ||
        buffer.buffer == VK_NULL_HANDLE) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: download_buffer requires a "
            "device-local GpuBuffer");
    }
    if (data == nullptr) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: download_buffer data is null");
    }
    if (size == 0 || size > buffer.size) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: download_buffer size invalid");
    }

    // One lock for engine check, staging grow, GPU copy, and host memcpy.
    std::lock_guard<std::mutex> lock(context.transfer_engine_mutex);
    require_transfer_engine(context, "download_buffer");
    ensure_staging(context, size);
    GpuBuffer& staging = context.transfer_engine.staging;
    copy_buffer_and_wait(context, buffer.buffer, staging.buffer, size);
    std::memcpy(data, staging.mapped, static_cast<size_t>(size));
}

} // namespace cthreads::gpu::memory
