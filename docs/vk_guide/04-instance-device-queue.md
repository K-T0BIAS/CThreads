# 04 — Instance, physical device, logical device, queue

This is the "plug the GPU in" chapter. In cthreads this lives in `Context`.

## The four names people mix up

| Name | What it is | Analogy |
|------|------------|---------|
| **Loader** | DLL/SO that finds drivers | "Phone book + switchboard" |
| **Instance** (`VkInstance`) | Your app's connection to Vulkan | "Logged into the Vulkan system" |
| **Physical device** (`VkPhysicalDevice`) | One GPU the loader can see | "The hardware card" |
| **Logical device** (`VkDevice`) | Your open session on that GPU | "File handle / open connection" |
| **Queue** (`VkQueue`) | Submission port for work | "Inbox where GPU jobs go" |

You always need: loader -> instance -> pick physical device -> create logical device -> get queue.

## Step by step (what `init()` does)

### Step A — Open the loader

Windows: `LoadLibraryA("vulkan-1.dll")`  
Linux: `dlopen("libvulkan.so.1", …)`

Then get **one** symbol by name: `vkGetInstanceProcAddr`.

Everything else is resolved through that function.

### Step B — Resolve `vkCreateInstance` (global)

Some functions can be queried with a **null instance**:

- `vkCreateInstance`
- (and a few enumerate-instance helpers we do not need yet)

```text
get_fn(NULL, "vkCreateInstance")
```

### Step C — Create the instance

You fill `VkApplicationInfo` (app name, engine name, API version) and `VkInstanceCreateInfo`.

We request **Vulkan 1.1** (`VK_API_VERSION_1_1`). That is enough for our compute path.

Out parameter: `c.instance`.

### Step D — Resolve instance-level functions

**Important rule (easy to get wrong):**

After you have an instance, resolve functions like:

- `vkDestroyInstance`
- `vkEnumeratePhysicalDevices`
- `vkGetPhysicalDeviceProperties`
- `vkCreateDevice`
- …

with **`c.instance`**, not `NULL`.

If you pass `NULL`, many loaders return nullptr for those names. That is a common failure mode (`missing vkDestroyInstance` / similar).

### Step E — Enumerate physical devices

Vulkan uses a two-call pattern constantly:

1. Call with `nullptr` data pointer to get **count**.
2. Allocate a vector of that size.
3. Call again to fill the vector.

```text
enumerate(instance, &count, nullptr)
vector<VkPhysicalDevice> devices(count)
enumerate(instance, &count, devices.data())
```

### Step F — Pick a device + queue family

For each physical device:

1. Read properties (name, discrete vs integrated).
2. Read queue family properties.
3. Find a family whose flags include `VK_QUEUE_COMPUTE_BIT`.
4. Prefer `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` (score higher).

Store:

- `physical_device`
- `queue_family` (index)
- `device_name` (string for Python)

### Step G — Create the logical device

You request queues:

```text
VkDeviceQueueCreateInfo: family index = queue_family, count = 1, priority = 1.0
VkDeviceCreateInfo: those queue infos
vkCreateDevice(physical_device, ...)
```

Out: `c.device`.

### Step H — Get the queue handle

```text
vkGetDeviceQueue(device, queue_family, 0, &queue)
```

Queue index `0` means "first queue of that family."

### Step I — Mark ready

`c.ready = true`.

On any failure: destroy what you created and unload the loader (`shutdown_unlocked`).

## Thread safety in our Context

`init` / `shutdown` take a mutex. Double-checked locking:

- Fast path: if `ready`, return.
- Lock, check again, then initialize once.

`available()` tries `init()` and returns false on failure (no throw).
`device_name()` tries `init()` and may throw mapped errors.

## Shutdown order

1. Destroy logical device (also invalidates queues).
2. Destroy instance.
3. Clear physical device handle.
4. `FreeLibrary` / `dlclose` the loader.
5. Null all function pointers so a late call cannot jump into unloaded code.

Command pools, fences, and buffers must be destroyed **before** destroying the logical device.

## What the Context layer does *not* do

- No buffers
- No command pools
- No shaders
- No descriptors

It only proves: **we can talk to a compute-capable GPU and print its name.**
