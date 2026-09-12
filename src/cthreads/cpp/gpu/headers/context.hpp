#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <cstdint>
#include <mutex>

#include "memory.hpp"

namespace cthreads::gpu {

/**
 * Reused copy machinery owned by Context for the process lifetime.
 *
 * Pool + fence are created in init. Staging scratch is optional and may stay
 * empty until the first upload/download grows it (see memory::).
 *
 * Destroy via shutdown_transfer_engine before destroying the logical device.
 */
struct TransferEngine {
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    // Host-visible scratch for H2D/D2H; empty until allocated. Grow-on-demand.
    memory::GpuBuffer staging;
};

struct Context {
    // OS handle to the loader shared library (HMODULE on Windows, void* on Linux).
    // Purpose: keep the DLL mapped and FreeLibrary/dlclose on shutdown.
    void* loader_module = nullptr;

    // Bootstrap entry from the loader. Type: pointer-to-function matching
    // VkResult-less GetInstanceProcAddr signature from the headers.
    // Usage: resolve almost every other Vulkan function by name string.
    // Naming Note: GetPRocAddress => GetFunctionAddress (Proc -> procedure -> function)
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    // Global (pre-instance) / instance-level entry points.
    // Each is a typed function pointer; assigned in init() via GetInstanceProcAddr.
    PFN_vkCreateInstance vkCreateInstance = nullptr; // creates the Vulkan instance (app <-> loader connection)
    PFN_vkDestroyInstance vkDestroyInstance = nullptr; // destroys the instance and its child resources owned at instance level
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr; // lists GPUs the loader can see
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr; // reads name, type, limits for one GPU
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties = nullptr; // lists queue families (graphics/compute/transfer) on one GPU
    PFN_vkCreateDevice vkCreateDevice = nullptr; // opens a logical device on a chosen physical GPU
    PFN_vkDestroyDevice vkDestroyDevice = nullptr; // destroys the logical device
    PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr; // gets a queue handle used to submit work

    // Buffer and memory functions
    PFN_vkCreateBuffer vkCreateBuffer = nullptr; // creates a buffer object (byte region descriptor; needs memory bound)
    PFN_vkDestroyBuffer vkDestroyBuffer = nullptr; // destroys a buffer object
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr; // reports size, alignment, and allowed memory types for a buffer
    PFN_vkAllocateMemory vkAllocateMemory = nullptr; // allocates a block of device (or host-visible) memory
    PFN_vkFreeMemory vkFreeMemory = nullptr; // frees a memory block from AllocateMemory
    PFN_vkBindBufferMemory vkBindBufferMemory = nullptr; // attaches a memory block to a buffer at an offset
    PFN_vkMapMemory vkMapMemory = nullptr; // exposes host-visible memory as a CPU pointer for read/write
    PFN_vkUnmapMemory vkUnmapMemory = nullptr; // releases a CPU mapping from MapMemory
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr; // lists memory heaps/types and their flags (e.g. host-visible, device-local)

    // Command pool, command buffer, copy, and sync functions
    PFN_vkCreateCommandPool vkCreateCommandPool = nullptr; // creates a pool that owns command buffers for one queue family
    PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr; // destroys a command pool and its buffers
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr; // allocates one or more command buffers from a pool
    PFN_vkFreeCommandBuffers vkFreeCommandBuffers = nullptr; // returns command buffers to the pool / frees them
    PFN_vkResetCommandBuffer vkResetCommandBuffer = nullptr; // clears a command buffer so it can be recorded again
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr; // starts recording commands into a command buffer
    PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr; // finishes recording; buffer is ready to submit
    PFN_vkCmdCopyBuffer vkCmdCopyBuffer = nullptr; // records a GPU copy from one buffer to another
    PFN_vkCreateFence vkCreateFence = nullptr; // creates a fence (CPU waits until GPU work finishes)
    PFN_vkDestroyFence vkDestroyFence = nullptr; // destroys a fence
    PFN_vkQueueSubmit vkQueueSubmit = nullptr; // submits recorded command buffers to a queue
    PFN_vkWaitForFences vkWaitForFences = nullptr; // blocks the CPU until the given fences signal
    PFN_vkResetFences vkResetFences = nullptr; // resets fences back to unsignaled for reuse

    // Shader / pipeline create + destroy (entry build and ShaderCache::clear).
    PFN_vkCreateShaderModule vkCreateShaderModule = nullptr;
    PFN_vkDestroyShaderModule vkDestroyShaderModule = nullptr;
    PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = nullptr;
    PFN_vkCreatePipelineLayout vkCreatePipelineLayout = nullptr;
    PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout = nullptr;
    PFN_vkCreateComputePipelines vkCreateComputePipelines = nullptr;
    PFN_vkDestroyPipeline vkDestroyPipeline = nullptr;

    // Descriptor pool / set / update (per-launch wiring of GpuPack buffers).
    PFN_vkCreateDescriptorPool vkCreateDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets = nullptr;
    PFN_vkFreeDescriptorSets vkFreeDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets = nullptr;

    // Compute dispatch recording (launch_gpu_kernel command buffers).
    PFN_vkCmdBindPipeline vkCmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets = nullptr;
    PFN_vkCmdDispatch vkCmdDispatch = nullptr;
    PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = nullptr;

    // Opaque Vulkan handles.
    VkInstance instance = VK_NULL_HANDLE;           // connection to the loader/app
    VkPhysicalDevice physical_device = VK_NULL_HANDLE; // chosen GPU
    VkDevice device = VK_NULL_HANDLE;               // logical device (opened GPU)
    VkQueue queue = VK_NULL_HANDLE;                 // compute submission port
    // Which queue family index we passed to vkCreateDevice (needed for pools).
    uint32_t queue_family = 0;
    // Human-readable GPU name from VkPhysicalDeviceProperties::deviceName.
    std::string device_name;
    // True only after init() fully succeeded.
    bool ready = false;

    // These are both temporary until the cpu side @gpu calls are implemented (this however is a future poject and not on the current timeline)
    TransferEngine transfer_engine;
    std::mutex transfer_engine_mutex;
};
// Process-wide singleton accessor.
Context& context();
// Create loader + instance + device + queue. Throws on failure.
void init();
// Destroy device/instance; unload loader; clear pointers. Safe to call if not ready.
void shutdown();
// If not ready, try init once; return ready without throwing (for available()).
bool available();
// Requires ready context; returns device_name.
const std::string& device_name();

} // namespace cthreads::gpu