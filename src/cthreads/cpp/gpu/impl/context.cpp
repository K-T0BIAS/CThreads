#include "../headers/context.hpp"
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <string>
#include <mutex>
#include <vector>

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
        c.ready = true;
    }

    void shutdown_unlocked(Context& c) {
    
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
        c.vkGetInstanceProcAddr = nullptr;
        c.vkCreateInstance = nullptr;
        c.vkDestroyInstance = nullptr;
        c.vkEnumeratePhysicalDevices = nullptr;
        c.vkGetPhysicalDeviceProperties = nullptr;
        c.vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
        c.vkCreateDevice = nullptr;
        c.vkDestroyDevice = nullptr;
        c.vkGetDeviceQueue = nullptr;
    
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