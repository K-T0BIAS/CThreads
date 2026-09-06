#include "../headers/context.hpp"
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <string>
#include <mutex>
#include <vector>

#include "../headers/memory.hpp"

#if defined(_WIN32)   
    // Windows (32-bit or 64-bit)
    #include <windows.h>
#elif defined(__linux__)  
    // Linux
    #include <dlfcn.h>
#else  
    // No MacOs support yet !!!!
    // Unknown -> throw error
    #error "cthreads: Unsupported OS"
#endif  

namespace cthreads::gpu {

namespace {

    // ------ Hidden TransferEngineHelpers ------

    void shutdown_transfer_engine(Context& c) {
        // Safe no-op if the engine was never created or already cleared.
        TransferEngine& te = c.transfer_engine;
        if (te.command_pool == VK_NULL_HANDLE &&
            te.fence == VK_NULL_HANDLE &&
            te.staging.buffer == VK_NULL_HANDLE) {
            return;
        }

        // Finish any in-flight copy before freeing GPU objects.
        if (te.fence != VK_NULL_HANDLE && c.device != VK_NULL_HANDLE &&
            c.vkWaitForFences) {
            c.vkWaitForFences(c.device, 1, &te.fence, VK_TRUE, UINT64_MAX);
        }

        // Staging first (uses device + buffer PFNs), then fence, then pool.
        if (te.staging.buffer != VK_NULL_HANDLE && c.device != VK_NULL_HANDLE) {
            memory::destroy_buffer(c, te.staging);
        }
        if (te.fence != VK_NULL_HANDLE && c.device != VK_NULL_HANDLE &&
            c.vkDestroyFence) {
            c.vkDestroyFence(c.device, te.fence, nullptr);
            te.fence = VK_NULL_HANDLE;
        }
        if (te.command_pool != VK_NULL_HANDLE && c.device != VK_NULL_HANDLE &&
            c.vkDestroyCommandPool) {
            c.vkDestroyCommandPool(c.device, te.command_pool, nullptr);
            te.command_pool = VK_NULL_HANDLE;
        }
        te = TransferEngine{};
    }

    void init_transfer_engine(Context& c) {
        // Pool + fence only. Staging is allocated later on demand so idle
        // contexts do not hold a fixed 1 MiB host-visible buffer.
        if (c.transfer_engine.command_pool != VK_NULL_HANDLE &&
            c.transfer_engine.fence != VK_NULL_HANDLE) {
            return;
        }

        if (c.device == VK_NULL_HANDLE || !c.vkCreateCommandPool ||
            !c.vkDestroyCommandPool || !c.vkCreateFence || !c.vkDestroyFence) {
            throw std::runtime_error(
                "cthreads.gpu.VulkanInitFailed: init_transfer_engine missing "
                "device or command/fence entry points");
        }

        // If a previous attempt left a half-built engine, clear it first.
        if (c.transfer_engine.command_pool != VK_NULL_HANDLE ||
            c.transfer_engine.fence != VK_NULL_HANDLE ||
            c.transfer_engine.staging.buffer != VK_NULL_HANDLE) {
            shutdown_transfer_engine(c);
        }

        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = c.queue_family;
        // TRANSIENT: short-lived recordings. RESET: allow vkResetCommandBuffer reuse.
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                          VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (c.vkCreateCommandPool(
                c.device, &pool_info, nullptr, &c.transfer_engine.command_pool) !=
            VK_SUCCESS) {
            throw std::runtime_error(
                "cthreads.gpu.VulkanInitFailed: vkCreateCommandPool failed");
        }

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // Signaled so the first wait/reset path can treat it as "idle".
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (c.vkCreateFence(
                c.device, &fence_info, nullptr, &c.transfer_engine.fence) !=
            VK_SUCCESS) {
            c.vkDestroyCommandPool(
                c.device, c.transfer_engine.command_pool, nullptr);
            c.transfer_engine.command_pool = VK_NULL_HANDLE;
            throw std::runtime_error(
                "cthreads.gpu.VulkanInitFailed: vkCreateFence failed");
        }
    }

    // ------ Hidden Context Helpers ------

    // Look up one export inside the already-loaded loader module.
    // module: void* (HMODULE on Windows). name: C string export name.
    // returns: raw code address, or nullptr if missing.
    static void* load_fn(void* module, const char* name) {
#if defined(_WIN32)
        return reinterpret_cast<void*>(
            // gets the function ptr address by name from the module lookup table
            GetProcAddress(static_cast<HMODULE>(module), name));
#else
        return dlsym(module, name);
#endif
    }

    // Resolve a Vulkan entry point by name and cast to the typed PFN_*.
    // c: Context with vkGetInstanceProcAddr already set.
    // instance: VK_NULL_HANDLE for global functions; real instance after create.
    // name: e.g. "vkCreateInstance".
    template <typename PFN>
    static PFN get_fn(Context& c, VkInstance instance, const char* name) {
        // PFN_vkVoidFunction = generic "pointer to some Vulkan fn".
        PFN_vkVoidFunction raw = c.vkGetInstanceProcAddr(instance, name);
        if (!raw) {
            throw std::runtime_error(
                std::string("cthreads.gpu.VulkanInitFailed: missing ") + name);
        }
        return reinterpret_cast<PFN>(raw);  // PFN = e.g. PFN_vkCreateInstance
    }

    // Open the Vulkan DLL / shared library and put the file pointer on the Context.loader_module
    // Throws on failure.
    // also assign the vkGetInstanceProcAddr.
    // Naming Note: GetPRocAddress 0> GetFunctionAddress
    static void open_loader(Context& c) {
#if defined(_WIN32)
        // Map the Vulkan loader DLL into this process (driver-installed).
        HMODULE mod = LoadLibraryA("vulkan-1.dll");
        if (!mod) {
            throw std::runtime_error(
                "cthreads.gpu.VulkanLoaderNotFound: vulkan-1.dll not found");
        }
        c.loader_module = static_cast<void*>(mod);
#else
        void* mod = dlopen("libvulkan.so.1", RTLD_NOW);
        if (!mod) {
            throw std::runtime_error(
                "cthreads.gpu.VulkanLoaderNotFound: libvulkan.so.1 not found");
        }
        c.loader_module = mod;
#endif
        // Only this first symbol comes from GetProcAddress/dlsym.
        // Everything else goes through vkGetInstanceProcAddr.
        c.vkGetInstanceProcAddr =
            reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                load_fn(c.loader_module, "vkGetInstanceProcAddr"));
        if (!c.vkGetInstanceProcAddr) {
            throw std::runtime_error(
                "cthreads.gpu.VulkanInitFailed: vkGetInstanceProcAddr missing");
        }
    }

    static void create_instance_and_device(Context& c) {
        // Only globals may be resolved with VK_NULL_HANDLE (Vulkan loader rules).
        // Instance-level procs (DestroyInstance, EnumeratePhysicalDevices, …)
        // must be resolved after vkCreateInstance with the real instance.
        c.vkCreateInstance = get_fn<PFN_vkCreateInstance>(
            c, VK_NULL_HANDLE, "vkCreateInstance");
        // VkApplicationInfo: tells the loader who we are (required sType pattern).
        VkApplicationInfo app{}; // vulkan app metadata
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; // set type for vulkan to interprete this struct
        app.pApplicationName = "cthreads";
        app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        app.pEngineName = "cthreads";
        app.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        app.apiVersion = VK_API_VERSION_1_1;  // request 1.1
        // VkInstanceCreateInfo: parameters for vkCreateInstance.
        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; // set type for vulkan to interprete this struct
        ici.pApplicationInfo = &app;
        // no layers/extensions for Issue 1
        // Out-param: writes the new VkInstance into c.instance.
        if (c.vkCreateInstance(&ici, nullptr, &c.instance) != VK_SUCCESS) {
            throw std::runtime_error(
                "cthreads.gpu.VulkanInitFailed: vkCreateInstance failed");
        }
        // Instance-level procs (pass c.instance).
        c.vkDestroyInstance = get_fn<PFN_vkDestroyInstance>(
            c, c.instance, "vkDestroyInstance");
        c.vkEnumeratePhysicalDevices = get_fn<PFN_vkEnumeratePhysicalDevices>(
            c, c.instance, "vkEnumeratePhysicalDevices");
        c.vkGetPhysicalDeviceProperties =
            get_fn<PFN_vkGetPhysicalDeviceProperties>( // get fn from vulkan (gets the properties of the physical device)
                c, c.instance, "vkGetPhysicalDeviceProperties");
        c.vkGetPhysicalDeviceQueueFamilyProperties =
            get_fn<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
                c, c.instance, "vkGetPhysicalDeviceQueueFamilyProperties");
        c.vkGetPhysicalDeviceMemoryProperties =
            get_fn<PFN_vkGetPhysicalDeviceMemoryProperties>(
                c, c.instance, "vkGetPhysicalDeviceMemoryProperties");
        c.vkCreateDevice = get_fn<PFN_vkCreateDevice>(
            c, c.instance, "vkCreateDevice");
        c.vkDestroyDevice = get_fn<PFN_vkDestroyDevice>(
            c, c.instance, "vkDestroyDevice");
        c.vkGetDeviceQueue = get_fn<PFN_vkGetDeviceQueue>(
            c, c.instance, "vkGetDeviceQueue");
        // --- list GPUs (two-call idiom: count, then data) ---
        uint32_t dev_count = 0;
        c.vkEnumeratePhysicalDevices(c.instance, &dev_count, nullptr);
        if (dev_count == 0) {
            throw std::runtime_error(
                "cthreads.gpu.VulkanNoDevice: no physical devices");
        }
        std::vector<VkPhysicalDevice> devices(dev_count);
        c.vkEnumeratePhysicalDevices(c.instance, &dev_count, devices.data());
        // Pick: need COMPUTE queue; prefer discrete GPU.
        int best_score = -1;
        for (VkPhysicalDevice pd : devices) {
            VkPhysicalDeviceProperties props{};
            c.vkGetPhysicalDeviceProperties(pd, &props);
            uint32_t qcount = 0;
            c.vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, nullptr);
            std::vector<VkQueueFamilyProperties> qprops(qcount);
            c.vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, qprops.data());
            for (uint32_t fi = 0; fi < qcount; ++fi) {
                if (!(qprops[fi].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                    continue;  // graphics-only family: skip
                }
                int score = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                                ? 1000
                                : 100;
                if (score > best_score) {
                    best_score = score;
                    c.physical_device = pd;
                    c.queue_family = fi;
                    c.device_name = props.deviceName;  // C string -> std::string
                }
            }
        }
        if (c.physical_device == VK_NULL_HANDLE) {
            throw std::runtime_error(
                "cthreads.gpu.VulkanNoDevice: no compute queue family");
        }
        // --- logical device = "open" that GPU for our process ---
        float priority = 1.0f;  // single queue, max priority in [0,1]
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = c.queue_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        if (c.vkCreateDevice(c.physical_device, &dci, nullptr, &c.device) != VK_SUCCESS) {
            throw std::runtime_error(
                "cthreads.gpu.VulkanInitFailed: vkCreateDevice failed");
        }
        // Queue handle is owned by the device; index 0 of that family.
        c.vkGetDeviceQueue(c.device, c.queue_family, 0, &c.queue);

        // Device-level buffer / memory / transfer entry points (Issue 2).
        // Resolved after the logical device exists; GIPA still returns loader trampolines.
        c.vkCreateBuffer = get_fn<PFN_vkCreateBuffer>(
            c, c.instance, "vkCreateBuffer");
        c.vkDestroyBuffer = get_fn<PFN_vkDestroyBuffer>(
            c, c.instance, "vkDestroyBuffer");
        c.vkGetBufferMemoryRequirements = get_fn<PFN_vkGetBufferMemoryRequirements>(
            c, c.instance, "vkGetBufferMemoryRequirements");
        c.vkAllocateMemory = get_fn<PFN_vkAllocateMemory>(
            c, c.instance, "vkAllocateMemory");
        c.vkFreeMemory = get_fn<PFN_vkFreeMemory>(
            c, c.instance, "vkFreeMemory");
        c.vkBindBufferMemory = get_fn<PFN_vkBindBufferMemory>(
            c, c.instance, "vkBindBufferMemory");
        c.vkMapMemory = get_fn<PFN_vkMapMemory>(
            c, c.instance, "vkMapMemory");
        c.vkUnmapMemory = get_fn<PFN_vkUnmapMemory>(
            c, c.instance, "vkUnmapMemory");
        c.vkCreateCommandPool = get_fn<PFN_vkCreateCommandPool>(
            c, c.instance, "vkCreateCommandPool");
        c.vkDestroyCommandPool = get_fn<PFN_vkDestroyCommandPool>(
            c, c.instance, "vkDestroyCommandPool");
        c.vkAllocateCommandBuffers = get_fn<PFN_vkAllocateCommandBuffers>(
            c, c.instance, "vkAllocateCommandBuffers");
        c.vkFreeCommandBuffers = get_fn<PFN_vkFreeCommandBuffers>(
            c, c.instance, "vkFreeCommandBuffers");
        c.vkResetCommandBuffer = get_fn<PFN_vkResetCommandBuffer>(
            c, c.instance, "vkResetCommandBuffer");
        c.vkBeginCommandBuffer = get_fn<PFN_vkBeginCommandBuffer>(
            c, c.instance, "vkBeginCommandBuffer");
        c.vkEndCommandBuffer = get_fn<PFN_vkEndCommandBuffer>(
            c, c.instance, "vkEndCommandBuffer");
        c.vkCmdCopyBuffer = get_fn<PFN_vkCmdCopyBuffer>(
            c, c.instance, "vkCmdCopyBuffer");
        c.vkCreateFence = get_fn<PFN_vkCreateFence>(
            c, c.instance, "vkCreateFence");
        c.vkDestroyFence = get_fn<PFN_vkDestroyFence>(
            c, c.instance, "vkDestroyFence");
        c.vkQueueSubmit = get_fn<PFN_vkQueueSubmit>(
            c, c.instance, "vkQueueSubmit");
        c.vkWaitForFences = get_fn<PFN_vkWaitForFences>(
            c, c.instance, "vkWaitForFences");
        c.vkResetFences = get_fn<PFN_vkResetFences>(
            c, c.instance, "vkResetFences");

        c.ready = true;
        // After device + PFNs + ready: reusable copy pool/fence (staging grows later).
        init_transfer_engine(c);
    }

    void shutdown_unlocked(Context& c) {
        // Children before parents: transfer engine (pool/fence/staging) then device.
        shutdown_transfer_engine(c);

        // 1) release logical device
        if (c.device != VK_NULL_HANDLE && c.vkDestroyDevice) { // check if device is set and if theres a destroy fn for it
            c.vkDestroyDevice(c.device, nullptr);  // set nullptr
            c.device = VK_NULL_HANDLE; // must be nulled out 
            c.queue = VK_NULL_HANDLE;  // must be nulled out 
        }
    
        // 2) release instance
        if (c.instance != VK_NULL_HANDLE && c.vkDestroyInstance) { // ensure instance is set and if theres a destroy fn for it
            c.vkDestroyInstance(c.instance, nullptr); // set to nullptr
            c.instance = VK_NULL_HANDLE; // null the handle to avoid dangling pointers
        }
        c.physical_device = VK_NULL_HANDLE; // can be nulled now that the instance and logical device are destroyed
    
        // 3) Unmap loader DLL so OS can unload it.
        if (c.loader_module) {
        // closes the files and nulls the pointer to the module
    #if defined(_WIN32)
            FreeLibrary(static_cast<HMODULE>(c.loader_module));
    #else
            dlclose(c.loader_module);
    #endif
            c.loader_module = nullptr;
        }
    
        // 4) Clear function pointers so a buggy late call can't jump into freed DLL!
        // instance functions
        c.vkGetInstanceProcAddr = nullptr;
        c.vkCreateInstance = nullptr;
        c.vkDestroyInstance = nullptr;
        c.vkEnumeratePhysicalDevices = nullptr;
        c.vkGetPhysicalDeviceProperties = nullptr;
        c.vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
        c.vkCreateDevice = nullptr;
        c.vkDestroyDevice = nullptr;
        c.vkGetDeviceQueue = nullptr;

        // buffer and memory functions
        c.vkCreateBuffer = nullptr;
        c.vkDestroyBuffer = nullptr;
        c.vkGetBufferMemoryRequirements = nullptr;
        c.vkAllocateMemory = nullptr;
        c.vkFreeMemory = nullptr;
        c.vkBindBufferMemory = nullptr;
        c.vkMapMemory = nullptr;
        c.vkUnmapMemory = nullptr;
        c.vkGetPhysicalDeviceMemoryProperties = nullptr;

        // Command pool, command buffer, copy, and sync functions
        c.vkCreateCommandPool = nullptr;
        c.vkDestroyCommandPool = nullptr;
        c.vkAllocateCommandBuffers = nullptr;
        c.vkFreeCommandBuffers = nullptr;
        c.vkResetCommandBuffer = nullptr;
        c.vkBeginCommandBuffer = nullptr;
        c.vkEndCommandBuffer = nullptr;
        c.vkCmdCopyBuffer = nullptr;
        c.vkCreateFence = nullptr;
        c.vkDestroyFence = nullptr;
        c.vkQueueSubmit = nullptr;
        c.vkWaitForFences = nullptr;
        c.vkResetFences = nullptr;
    
        c.queue_family = 0;
        c.device_name.clear();
        c.ready = false;
    }

} // namespace anonymous

    // to lock the init and shutdown functions aswell as any thread unsafe gpu functions
    static std::mutex& gpu_mutex() {
        static std::mutex m;
        return m;
    }

    Context& context() {
        static Context ctx;
        return ctx;
    }

    const std::string& device_name() {
        try {
            init(); // try to initialize (noops if already initialized)
            return context().device_name;
        } catch (const std::exception& e) {
            throw std::runtime_error("cthreads: " + std::string(e.what())); // this could be cleaner but isnt relevant for now
        }
    }

    bool available() {
        try {
            init(); // try to initialize (noops if already initialized)
            return context().ready; // return success/failure
        } catch (...) {
            return false; // initialization failed
        }
    }

    void init() {
        Context& c = context();
        if (c.ready) return;
    
        std::lock_guard<std::mutex> lock(gpu_mutex());
        if (c.ready) return;
    
        try {
            open_loader(c);
            create_instance_and_device(c);
        } catch (...) {
            shutdown_unlocked(c);
            throw;  // original exception, nothing stored
        }
    }


    void shutdown() {
        std::lock_guard<std::mutex> lock(gpu_mutex());
        shutdown_unlocked(context());
    }

} // namespace cthreads::gpu