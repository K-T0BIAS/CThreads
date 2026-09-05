#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <cstdint>

namespace cthreads::gpu {

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
    PFN_vkCreateInstance vkCreateInstance = nullptr;
    PFN_vkDestroyInstance vkDestroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkCreateDevice vkCreateDevice = nullptr;
    PFN_vkDestroyDevice vkDestroyDevice = nullptr;
    PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;
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