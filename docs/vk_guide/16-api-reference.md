# 16 — Vulkan types, structs, enums, and functions (cthreads compute)

This is a **reference chapter** for Vulkan symbols that matter to the cthreads
GPU compute path (Context, memory/transfers, and the planned launch path:
descriptors, pipelines, dispatch). It is not the full Vulkan API.

Conventions used below:

- **Handle** means an opaque Vulkan object id (do not dereference it as a C pointer).
- **`sType`** on create/info structs must be set to the matching `VK_STRUCTURE_TYPE_*`.
- **`pNext`** is usually `nullptr` in cthreads (extension chaining).
- **`pAllocator`** is always `nullptr` in cthreads (use the default allocator).
- **In / out** describes whether the caller fills the argument or the driver writes it.

Related conceptual chapters: [02](./02-mental-model.md), [04](./04-instance-device-queue.md),
[06](./06-buffers-and-memory.md), [08](./08-commands-fences.md), [09](./09-descriptors-ssbo.md),
[11](./11-spirv-pipelines-dispatch.md).

---

## Table of contents

1. [Scalar and handle types](#1-scalar-and-handle-types)
2. [Results and booleans](#2-results-and-booleans)
3. [Structure type tags (`sType`)](#3-structure-type-tags-stype)
4. [Version helpers](#4-version-helpers)
5. [Instance and device bootstrap](#5-instance-and-device-bootstrap)
6. [Physical device queries](#6-physical-device-queries)
7. [Queues](#7-queues)
8. [Buffers and device memory](#8-buffers-and-device-memory)
9. [Command pools, command buffers, copies](#9-command-pools-command-buffers-copies)
10. [Fences and submit](#10-fences-and-submit)
11. [Launch path: descriptors](#11-launch-path-descriptors)
12. [Launch path: shaders, pipelines, dispatch](#12-launch-path-shaders-pipelines-dispatch)
13. [Launch path: barriers (overview)](#13-launch-path-barriers-overview)
14. [Function pointer typedefs (`PFN_*`)](#14-function-pointer-typedefs-pfn_)
15. [Quick index by name](#15-quick-index-by-name)

---

## 1. Scalar and handle types

### `VkBool32`

- **What:** Vulkan boolean. Not C++ `bool`.
- **Values:** `VK_TRUE` (1), `VK_FALSE` (0).
- **Where used:** e.g. `vkWaitForFences(..., waitAll, ...)`.

### `VkDeviceSize`

- **What:** Unsigned 64-bit size/offset type for buffer sizes, copy sizes, memory offsets.
- **Why not `size_t`:** Vulkan wants a fixed width across platforms.
- **cthreads:** `GpuBuffer.size`, upload/download byte counts.

### `VK_NULL_HANDLE`

- **What:** Sentinel meaning "no object" for handle types.
- **Analogy:** Like `nullptr` for Vulkan handles.
- **cthreads:** Initial value of all handles on `Context` / `GpuBuffer`.

### `VK_WHOLE_SIZE`

- **What:** Special size meaning "the rest of the resource" (often used with map/descriptor ranges).
- **cthreads:** May appear when mapping whole allocations or descriptor buffer ranges.

### Opaque handles (object ids)

Handles are opaque ids created by `vkCreate*` / allocate functions and destroyed by
matching destroy/free functions. Destroy children before parents.

Below is what each object is **for** (purpose), not only what the typedef is called.

#### `VkInstance`

**Purpose:** The process-wide "logged into Vulkan" object. It owns the connection to
the loader and is required before enumerating GPUs or creating a device.

**How it works:** After `vkCreateInstance`, the loader knows this application exists
and can route instance-level calls. Almost every later bootstrap call takes the
instance (or a device created from it). Destroying the instance last tears down that
connection. cthreads keeps one instance on `Context`.

#### `VkPhysicalDevice`

**Purpose:** A handle representing one GPU the loader can see (hardware or software ICD).

**How it works:** It is not "opened" yet. It is used only to **query** capabilities
(name, queue families, memory types) and as input to `vkCreateDevice`. Destroying the
instance invalidates physical device handles. cthreads picks one compute-capable
physical device (preferring discrete).

#### `VkDevice` (logical device)

**Purpose:** The opened session on a chosen GPU. Almost all GPU work objects (buffers,
pipelines, command pools) are created from a device.

**How it works:** `vkCreateDevice` tells the driver which queue families this app will
use. After that, device-level entry points operate on this handle. Destroying the
device invalidates all objects created from it. cthreads uses one logical device.

#### `VkQueue`

**Purpose:** A submission port. Recorded command buffers are sent here to run on the GPU.

**How it works:** Queues belong to a queue family (graphics/compute/transfer).
`vkGetDeviceQueue` returns a handle owned by the device (no separate destroy).
`vkQueueSubmit` is asynchronous: it returns when work is queued, not when finished.
cthreads uses one compute-capable queue for copies and (later) dispatches.

#### `VkBuffer`

**Purpose:** A GPU resource that represents a contiguous byte region with allowed uses
(copy, storage buffer, and so on).

**How it works:** Creating a buffer only creates the *object*. Bytes live in
`VkDeviceMemory` after `vkBindBufferMemory`. Shaders never see the C++ handle directly;
descriptors bind the buffer to a binding number. cthreads uses staging buffers and
device-local SSBOs (`GpuBuffer`).

#### `VkDeviceMemory`

**Purpose:** An allocated slab of memory from a chosen memory type (host-visible,
device-local, …).

**How it works:** `vkAllocateMemory` reserves memory; `vkBindBufferMemory` attaches it
to a buffer. Host-visible memory can be `vkMapMemory`'d for CPU `memcpy`. Device-local
memory is typically filled via GPU copies from staging. Free with `vkFreeMemory`
after unmapping and after the bound buffer is destroyed or no longer needs it
(cthreads destroys buffer then frees memory).

#### `VkCommandPool`

**Purpose:** An allocator/owner for command buffers tied to one queue family.

**How it works:** Command buffers must come from a pool that matches the queue that
will submit them. Pools can be optimized for short-lived ("transient") buffers.
Destroying a pool frees its command buffers. cthreads currently creates a transient
pool per copy; a reused transfer engine would keep one pool alive on Context.

#### `VkCommandBuffer`

**Purpose:** A recorded recipe of GPU commands (copies, binds, dispatches).

**How it works:** The CPU calls `vkBeginCommandBuffer`, then `vkCmd*` functions, then
`vkEndCommandBuffer`. Nothing runs on the GPU until `vkQueueSubmit`. After submit, the
CPU can continue; completion is tracked with a fence (or semaphores). Think of it as
building a batch job, then mailing it to the GPU.

#### `VkFence`

**Purpose:** A CPU-visible "this GPU submit is finished" signal.

**How it works:** A fence starts unsignaled (unless created signaled). Passing it to
`vkQueueSubmit` asks the driver to signal it when that submit completes. The CPU calls
`vkWaitForFences` to block until signaled, then may safely read staging memory,
destroy temporary buffers, or mark a job joined. `vkResetFences` returns it to
unsignaled for reuse. Unlike semaphores (GPU-to-GPU), fences are the usual tool for
**CPU waiting on GPU** in cthreads.

#### `VkSemaphore`

**Purpose:** GPU-side synchronization between queue submits (and present in graphics).

**How it works:** One submit can signal a semaphore; a later submit can wait on it at a
pipeline stage. The CPU does not typically `wait` on semaphores the way it waits on
fences. cthreads compute/copy path can ignore semaphores while everything uses one
queue and fence waits.

#### `VkDescriptorSetLayout`

**Purpose:** The schema for a descriptor set: which binding numbers exist and what
types they are (e.g. binding 0 = storage buffer).

**How it works:** Created once for a shader interface. Pipeline layouts reference it.
It does not point at real buffers yet; it only describes the shape.

#### `VkDescriptorPool`

**Purpose:** A pool from which concrete descriptor sets are allocated.

**How it works:** Sized by how many sets and how many descriptors of each type may be
allocated. Destroying the pool frees its sets (unless using free-individual flags).

#### `VkDescriptorSet`

**Purpose:** One instance of a layout filled with real resources (which `VkBuffer` is
binding 1, and so on).

**How it works:** Allocate from a pool, then `vkUpdateDescriptorSets` to point bindings
at buffers. Before dispatch, `vkCmdBindDescriptorSets` attaches the set to the command
buffer so the shader's `layout(binding=N)` reads the right memory.

#### `VkShaderModule`

**Purpose:** Holds SPIR-V code for a shader stage.

**How it works:** Created from SPIR-V bytes. Referenced when creating a pipeline. The
module itself is not "run"; the pipeline binds the compiled stage. Safe to destroy the
module after the pipeline is created (the pipeline keeps what it needs), depending on
driver rules; many apps keep modules until shutdown.

#### `VkPipelineLayout`

**Purpose:** Declares the interface a pipeline expects: descriptor set layouts and
optional push-constant ranges.

**How it works:** Must match what the shader bindings and push constants use.
`vkCmdBindDescriptorSets` and `vkCmdPushConstants` take this layout.

#### `VkPipeline` (compute)

**Purpose:** A ready-to-bind compute program: shader stage + layout (+ state).

**How it works:** `vkCreateComputePipelines` compiles/links the compute stage for the
device. Recording `vkCmdBindPipeline` then `vkCmdDispatch` runs it. Creating pipelines
is relatively expensive, so cthreads should cache them by shader hash.

#### `VkPipelineCache`

**Purpose:** Optional cache so pipeline creation can reuse previous driver compilations.

**How it works:** Pass a cache into `vkCreateComputePipelines`. Can be saved to disk
across runs. Not required for correctness.

---

## 2. Results and booleans

### `VkResult` (enum)

Most Vulkan functions return `VkResult`.

| Value | Meaning for cthreads |
|-------|----------------------|
| `VK_SUCCESS` | Call succeeded |
| `VK_NOT_READY` | Not done yet (fences/queries) |
| `VK_TIMEOUT` | Wait timed out |
| `VK_EVENT_SET` / `VK_EVENT_RESET` | Event status (unused here) |
| `VK_INCOMPLETE` | Enumeration truncated (rare if count pattern is used correctly) |
| `VK_ERROR_OUT_OF_HOST_MEMORY` | CPU OOM |
| `VK_ERROR_OUT_OF_DEVICE_MEMORY` | GPU OOM |
| `VK_ERROR_INITIALIZATION_FAILED` | Init failed |
| `VK_ERROR_DEVICE_LOST` | Device lost (serious) |
| `VK_ERROR_MEMORY_MAP_FAILED` | `vkMapMemory` failed |
| `VK_ERROR_LAYER_NOT_PRESENT` | Requested layer missing |
| `VK_ERROR_EXTENSION_NOT_PRESENT` | Requested extension missing |
| `VK_ERROR_FEATURE_NOT_PRESENT` | Feature not available |
| `VK_ERROR_INCOMPATIBLE_DRIVER` | Driver too old / incompatible |

**cthreads pattern:** treat anything other than `VK_SUCCESS` as failure and throw `cthreads.gpu.VulkanInitFailed: …` (or a more specific mapped error).

---

## 3. Structure type tags (`sType`)

### `VkStructureType` (enum, partial)

Every create/info struct starts with:

```text
sType: VkStructureType
pNext: const void*   // usually nullptr
```

`sType` tells the driver which struct layout follows. Wrong `sType` is undefined behavior / validation errors.

Values cthreads uses (or will use):

| Enum | Struct |
|------|--------|
| `VK_STRUCTURE_TYPE_APPLICATION_INFO` | `VkApplicationInfo` |
| `VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO` | `VkInstanceCreateInfo` |
| `VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO` | `VkDeviceQueueCreateInfo` |
| `VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO` | `VkDeviceCreateInfo` |
| `VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO` | `VkBufferCreateInfo` |
| `VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO` | `VkMemoryAllocateInfo` |
| `VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO` | `VkCommandPoolCreateInfo` |
| `VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO` | `VkCommandBufferAllocateInfo` |
| `VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO` | `VkCommandBufferBeginInfo` |
| `VK_STRUCTURE_TYPE_FENCE_CREATE_INFO` | `VkFenceCreateInfo` |
| `VK_STRUCTURE_TYPE_SUBMIT_INFO` | `VkSubmitInfo` |
| `VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO` | `VkShaderModuleCreateInfo` |
| `VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO` | `VkPipelineLayoutCreateInfo` |
| `VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO` | `VkComputePipelineCreateInfo` |
| `VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO` | `VkPipelineShaderStageCreateInfo` |
| `VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO` | `VkDescriptorSetLayoutCreateInfo` |
| `VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO` | `VkDescriptorPoolCreateInfo` |
| `VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO` | `VkDescriptorSetAllocateInfo` |
| `VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET` | `VkWriteDescriptorSet` |
| `VK_STRUCTURE_TYPE_MEMORY_BARRIER` / `BUFFER_MEMORY_BARRIER` / … | Barrier structs |

**Rule:** always zero the struct (`{}`), then set `sType`, then set needed fields.

---

## 4. Version helpers

### `VK_MAKE_VERSION(major, minor, patch)`

Packs a version into a `uint32_t` for `VkApplicationInfo`.

### `VK_API_VERSION_1_0` / `VK_API_VERSION_1_1` / …

API version requested in `VkApplicationInfo.apiVersion`.

**cthreads:** requests `VK_API_VERSION_1_1` in Context init.

---

## 5. Instance and device bootstrap

**Purpose of this layer:** turn "Vulkan exists on this machine" into a usable
`VkDevice` + `VkQueue`. Without it, no buffers or submits are possible.

Flow: create `VkInstance` -> enumerate `VkPhysicalDevice`s -> create `VkDevice` with
a compute queue family -> `vkGetDeviceQueue`. See handle purposes in section 1.

### `VkApplicationInfo`

Metadata about the application (mostly for tools/drivers).

| Field | Type | Meaning |
|-------|------|---------|
| `sType` | `VkStructureType` | Must be `APPLICATION_INFO` |
| `pNext` | `const void*` | Usually `nullptr` |
| `pApplicationName` | `const char*` | App name string |
| `applicationVersion` | `uint32_t` | App version (`VK_MAKE_VERSION`) |
| `pEngineName` | `const char*` | Engine name (`"cthreads"`) |
| `engineVersion` | `uint32_t` | Engine version |
| `apiVersion` | `uint32_t` | Highest Vulkan API version the app uses |

### `VkInstanceCreateInfo`

Arguments to `vkCreateInstance`.

| Field | Type | Meaning |
|-------|------|---------|
| `sType` | `VkStructureType` | `INSTANCE_CREATE_INFO` |
| `pNext` | `const void*` | Extensions via pNext (unused in basic cthreads) |
| `flags` | `VkInstanceCreateFlags` | Usually 0 |
| `pApplicationInfo` | `const VkApplicationInfo*` | Pointer to app info |
| `enabledLayerCount` | `uint32_t` | Validation layers count (0 unless debugging) |
| `ppEnabledLayerNames` | `const char* const*` | Layer name list |
| `enabledExtensionCount` | `uint32_t` | Instance extensions (0 for basic compute) |
| `ppEnabledExtensionNames` | `const char* const*` | Extension name list |

### `vkCreateInstance`

```text
VkResult vkCreateInstance(
  const VkInstanceCreateInfo* pCreateInfo,  // in: create parameters
  const VkAllocationCallbacks* pAllocator,  // in: nullptr in cthreads
  VkInstance* pInstance                     // out: created instance handle
)
```

Creates the Vulkan instance. Resolve this entry point with a **null** instance via `vkGetInstanceProcAddr`.

### `vkDestroyInstance`

```text
void vkDestroyInstance(
  VkInstance instance,                      // in: instance to destroy
  const VkAllocationCallbacks* pAllocator   // in: nullptr
)
```

Destroys the instance. Resolve with the **real** instance handle (not null).

### `vkEnumeratePhysicalDevices`

```text
VkResult vkEnumeratePhysicalDevices(
  VkInstance instance,                 // in
  uint32_t* pPhysicalDeviceCount,      // in/out: count
  VkPhysicalDevice* pPhysicalDevices   // out: array, or nullptr to query count only
)
```

**Two-call idiom:** first call with `pPhysicalDevices == nullptr` to get count; allocate; second call to fill.

### `VkDeviceQueueCreateInfo`

Requests queues when creating a logical device.

| Field | Type | Meaning |
|-------|------|---------|
| `sType` | `VkStructureType` | `DEVICE_QUEUE_CREATE_INFO` |
| `pNext` | `const void*` | Usually `nullptr` |
| `flags` | `VkDeviceQueueCreateFlags` | Usually 0 |
| `queueFamilyIndex` | `uint32_t` | Which family (cthreads compute family) |
| `queueCount` | `uint32_t` | How many queues (cthreads: 1) |
| `pQueuePriorities` | `const float*` | Array of priorities in `[0,1]` (cthreads: `{1.0f}`) |

### `VkDeviceCreateInfo`

| Field | Type | Meaning |
|-------|------|---------|
| `sType` | `VkStructureType` | `DEVICE_CREATE_INFO` |
| `pNext` | `const void*` | Features/extensions via pNext if needed |
| `flags` | `VkDeviceCreateFlags` | Usually 0 |
| `queueCreateInfoCount` | `uint32_t` | Number of queue infos |
| `pQueueCreateInfos` | `const VkDeviceQueueCreateInfo*` | Queue requests |
| `enabledLayerCount` | `uint32_t` | Deprecated for device; usually 0 |
| `ppEnabledLayerNames` | `const char* const*` | Usually unused |
| `enabledExtensionCount` | `uint32_t` | Device extensions (0 for basic path) |
| `ppEnabledExtensionNames` | `const char* const*` | Extension names |
| `pEnabledFeatures` | `const VkPhysicalDeviceFeatures*` | Optional features; can be `nullptr` |

### `vkCreateDevice`

```text
VkResult vkCreateDevice(
  VkPhysicalDevice physicalDevice,          // in: chosen GPU
  const VkDeviceCreateInfo* pCreateInfo,    // in
  const VkAllocationCallbacks* pAllocator,  // in: nullptr
  VkDevice* pDevice                         // out
)
```

### `vkDestroyDevice`

```text
void vkDestroyDevice(
  VkDevice device,
  const VkAllocationCallbacks* pAllocator
)
```

Destroys the logical device. All device-child objects must already be destroyed.

### `vkGetDeviceQueue`

```text
void vkGetDeviceQueue(
  VkDevice device,              // in
  uint32_t queueFamilyIndex,    // in
  uint32_t queueIndex,          // in: 0 for first queue of that family
  VkQueue* pQueue               // out
)
```

Does not create a queue; retrieves a handle owned by the device.

---

## 6. Physical device queries

**Purpose of this layer:** ask a `VkPhysicalDevice` what it can do before opening it.
cthreads uses these queries to (1) prefer a discrete GPU with a compute queue, and
(2) pick legal memory types for staging vs device-local buffers.

### `VkPhysicalDeviceType` (enum)

| Value | Meaning |
|-------|---------|
| `VK_PHYSICAL_DEVICE_TYPE_OTHER` | Other / unknown |
| `VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU` | iGPU (CPU die) |
| `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` | Discrete card (preferred by cthreads scoring) |
| `VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU` | Virtualized |
| `VK_PHYSICAL_DEVICE_TYPE_CPU` | CPU fallback implementation |

### `VkPhysicalDeviceProperties`

Large struct. Fields cthreads cares about:

| Field | Type | Meaning |
|-------|------|---------|
| `apiVersion` | `uint32_t` | Max API version supported |
| `driverVersion` | `uint32_t` | Vendor driver version |
| `vendorID` | `uint32_t` | PCI vendor id |
| `deviceID` | `uint32_t` | PCI device id |
| `deviceType` | `VkPhysicalDeviceType` | Discrete vs integrated, etc. |
| `deviceName` | `char[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]` | Human-readable GPU name |
| `pipelineCacheUUID` | `uint8_t[VK_UUID_SIZE]` | Cache identity |
| `limits` | `VkPhysicalDeviceLimits` | Many limits (max bindings, etc.) |

### `vkGetPhysicalDeviceProperties`

```text
void vkGetPhysicalDeviceProperties(
  VkPhysicalDevice physicalDevice,           // in
  VkPhysicalDeviceProperties* pProperties    // out
)
```

### `VkQueueFlagBits` / `VkQueueFlags`

Bit flags describing what a queue family can do:

| Flag | Meaning |
|------|---------|
| `VK_QUEUE_GRAPHICS_BIT` | Graphics commands |
| `VK_QUEUE_COMPUTE_BIT` | Compute dispatches (required by cthreads) |
| `VK_QUEUE_TRANSFER_BIT` | Dedicated transfer (often also implied by graphics/compute) |
| `VK_QUEUE_SPARSE_BINDING_BIT` | Sparse memory (unused) |

### `VkQueueFamilyProperties`

| Field | Type | Meaning |
|-------|------|---------|
| `queueFlags` | `VkQueueFlags` | Capability bits |
| `queueCount` | `uint32_t` | How many queues in this family |
| `timestampValidBits` | `uint32_t` | Timestamp support |
| `minImageTransferGranularity` | `VkExtent3D` | Image copy granularity |

### `vkGetPhysicalDeviceQueueFamilyProperties`

Two-call idiom (count then array), same as device enumeration.

### `VkMemoryPropertyFlagBits` / `VkMemoryPropertyFlags`

| Flag | Meaning |
|------|---------|
| `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` | GPU-fast memory (VRAM-like). cthreads device-local SSBOs |
| `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` | CPU can map. cthreads staging |
| `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` | No manual flush for CPU/GPU visibility. cthreads staging |
| `VK_MEMORY_PROPERTY_HOST_CACHED_BIT` | CPU cached (optional) |
| `VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT` | Transient attachments (graphics; unused) |

### `VkMemoryType`

| Field | Type | Meaning |
|-------|------|---------|
| `propertyFlags` | `VkMemoryPropertyFlags` | What this type can do |
| `heapIndex` | `uint32_t` | Which heap it comes from |

### `VkMemoryHeap`

| Field | Type | Meaning |
|-------|------|---------|
| `size` | `VkDeviceSize` | Heap size in bytes |
| `flags` | `VkMemoryHeapFlags` | e.g. device-local heap |

### `VkPhysicalDeviceMemoryProperties`

| Field | Type | Meaning |
|-------|------|---------|
| `memoryTypeCount` | `uint32_t` | Number of types |
| `memoryTypes` | `VkMemoryType[]` | Types array |
| `memoryHeapCount` | `uint32_t` | Number of heaps |
| `memoryHeaps` | `VkMemoryHeap[]` | Heaps array |

### `vkGetPhysicalDeviceMemoryProperties`

```text
void vkGetPhysicalDeviceMemoryProperties(
  VkPhysicalDevice physicalDevice,
  VkPhysicalDeviceMemoryProperties* pMemoryProperties
)
```

Used by `find_memory_type` together with `memoryTypeBits` from buffer requirements.

---

## 7. Queues

**Purpose:** a `VkQueue` is where work enters the GPU. The CPU records command
buffers, then `vkQueueSubmit` hands them to a queue. Execution is asynchronous
relative to the CPU.

Queues are obtained with `vkGetDeviceQueue` (section 5). There is no
`vkDestroyQueue`; destroying the device invalidates queues.

cthreads uses a single compute-capable queue for buffer copies and later for
compute dispatches.

---

## 8. Buffers and device memory

**Purpose of this layer:** give the GPU (and optionally the CPU) a place to store
bytes. A `VkBuffer` is the typed resource; `VkDeviceMemory` is the backing store.
Binding connects them. Staging memory is host-visible for `memcpy`; device-local
memory is what shaders should use for hot data.

See also the handle write-ups for `VkBuffer` and `VkDeviceMemory` in section 1.

### `VkBufferUsageFlagBits` / `VkBufferUsageFlags`

| Flag | Meaning |
|------|---------|
| `VK_BUFFER_USAGE_TRANSFER_SRC_BIT` | May be source of `vkCmdCopyBuffer` |
| `VK_BUFFER_USAGE_TRANSFER_DST_BIT` | May be destination of a copy |
| `VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT` | UBO (cthreads prefers SSBO for mutable pack) |
| `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` | SSBO for compute read/write |
| `VK_BUFFER_USAGE_INDEX_BUFFER_BIT` | Graphics index buffer (unused) |
| `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT` | Graphics vertex buffer (unused) |
| `VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT` | Indirect dispatch/draw (optional later) |

**cthreads Staging:** `TRANSFER_SRC | TRANSFER_DST`  
**cthreads DeviceLocal:** `STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST`

### `VkSharingMode` (enum)

| Value | Meaning |
|-------|---------|
| `VK_SHARING_MODE_EXCLUSIVE` | One queue family owns it (cthreads default) |
| `VK_SHARING_MODE_CONCURRENT` | Multiple families; must list indices |

### `VkBufferCreateInfo`

| Field | Type | Meaning |
|-------|------|---------|
| `sType` | `VkStructureType` | `BUFFER_CREATE_INFO` |
| `pNext` | `const void*` | Usually `nullptr` |
| `flags` | `VkBufferCreateFlags` | Sparse flags etc.; usually 0 |
| `size` | `VkDeviceSize` | Size in bytes (> 0 in cthreads) |
| `usage` | `VkBufferUsageFlags` | Allowed uses |
| `sharingMode` | `VkSharingMode` | Exclusive vs concurrent |
| `queueFamilyIndexCount` | `uint32_t` | Needed if concurrent |
| `pQueueFamilyIndices` | `const uint32_t*` | Families if concurrent |

### `vkCreateBuffer`

```text
VkResult vkCreateBuffer(
  VkDevice device,
  const VkBufferCreateInfo* pCreateInfo,
  const VkAllocationCallbacks* pAllocator,
  VkBuffer* pBuffer
)
```

Creates the buffer object only. Memory is separate.

### `vkDestroyBuffer`

```text
void vkDestroyBuffer(
  VkDevice device,
  VkBuffer buffer,
  const VkAllocationCallbacks* pAllocator
)
```

### `VkMemoryRequirements`

| Field | Type | Meaning |
|-------|------|---------|
| `size` | `VkDeviceSize` | Bytes to allocate (may exceed create size due to alignment) |
| `alignment` | `VkDeviceSize` | Required alignment of the bind offset |
| `memoryTypeBits` | `uint32_t` | Bit i set => memory type i is legal |

### `vkGetBufferMemoryRequirements`

```text
void vkGetBufferMemoryRequirements(
  VkDevice device,
  VkBuffer buffer,
  VkMemoryRequirements* pMemoryRequirements
)
```

### `VkMemoryAllocateInfo`

| Field | Type | Meaning |
|-------|------|---------|
| `sType` | `VkStructureType` | `MEMORY_ALLOCATE_INFO` |
| `pNext` | `const void*` | Usually `nullptr` |
| `allocationSize` | `VkDeviceSize` | Usually `mem_reqs.size` |
| `memoryTypeIndex` | `uint32_t` | From `find_memory_type` |

### `vkAllocateMemory`

```text
VkResult vkAllocateMemory(
  VkDevice device,
  const VkMemoryAllocateInfo* pAllocateInfo,
  const VkAllocationCallbacks* pAllocator,
  VkDeviceMemory* pMemory
)
```

### `vkFreeMemory`

```text
void vkFreeMemory(
  VkDevice device,
  VkDeviceMemory memory,
  const VkAllocationCallbacks* pAllocator
)
```

### `vkBindBufferMemory`

```text
VkResult vkBindBufferMemory(
  VkDevice device,
  VkBuffer buffer,
  VkDeviceMemory memory,
  VkDeviceSize memoryOffset   // must respect alignment; cthreads uses 0 with dedicated alloc
)
```

Associates memory with the buffer. A buffer is bound at most once (without sparse extensions).

### `VkMemoryMapFlags`

Usually `0` for `vkMapMemory`.

### `vkMapMemory`

```text
VkResult vkMapMemory(
  VkDevice device,
  VkDeviceMemory memory,
  VkDeviceSize offset,        // start offset in the allocation
  VkDeviceSize size,          // bytes to map, or VK_WHOLE_SIZE
  VkMemoryMapFlags flags,     // usually 0
  void** ppData               // out: CPU pointer
)
```

Only valid for **host-visible** memory. cthreads maps staging buffers.

### `vkUnmapMemory`

```text
void vkUnmapMemory(
  VkDevice device,
  VkDeviceMemory memory
)
```

---

## 9. Command pools, command buffers, copies

**Purpose of this layer:** build and own GPU "to-do lists." A command buffer is the
list; a command pool allocates those lists for a specific queue family;
`vkCmdCopyBuffer` is one kind of item on the list. Recording is CPU-side; running
happens only after submit.

See handle write-ups for `VkCommandPool` and `VkCommandBuffer` in section 1.

### `VkCommandPoolCreateFlagBits`

| Flag | Meaning |
|------|---------|
| `VK_COMMAND_POOL_CREATE_TRANSIENT_BIT` | Short-lived CBs (cthreads one-shot copies) |
| `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT` | Allow resetting individual CBs |

### `VkCommandPoolCreateInfo`

| Field | Type | Meaning |
|-------|------|---------|
| `sType` | `VkStructureType` | `COMMAND_POOL_CREATE_INFO` |
| `pNext` | `const void*` | Usually `nullptr` |
| `flags` | `VkCommandPoolCreateFlags` | Transient / reset bits |
| `queueFamilyIndex` | `uint32_t` | Must match the queue that will submit |

### `vkCreateCommandPool` / `vkDestroyCommandPool`

Standard create/destroy pair on a `VkDevice`.

### `VkCommandBufferLevel` (enum)

| Value | Meaning |
|-------|---------|
| `VK_COMMAND_BUFFER_LEVEL_PRIMARY` | Can be submitted to a queue (cthreads) |
| `VK_COMMAND_BUFFER_LEVEL_SECONDARY` | Can be called from primary (advanced) |

### `VkCommandBufferAllocateInfo`

| Field | Type | Meaning |
|-------|------|---------|
| `sType` | `VkStructureType` | `COMMAND_BUFFER_ALLOCATE_INFO` |
| `pNext` | `const void*` | Usually `nullptr` |
| `commandPool` | `VkCommandPool` | Pool to allocate from |
| `level` | `VkCommandBufferLevel` | Primary in cthreads |
| `commandBufferCount` | `uint32_t` | How many to allocate |

### `vkAllocateCommandBuffers` / `vkFreeCommandBuffers`

```text
VkResult vkAllocateCommandBuffers(
  VkDevice device,
  const VkCommandBufferAllocateInfo* pAllocateInfo,
  VkCommandBuffer* pCommandBuffers   // out array
)

void vkFreeCommandBuffers(
  VkDevice device,
  VkCommandPool commandPool,
  uint32_t commandBufferCount,
  const VkCommandBuffer* pCommandBuffers
)
```

### `VkCommandBufferUsageFlagBits`

| Flag | Meaning |
|------|---------|
| `VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT` | Record, submit once (copies) |
| `VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT` | Graphics secondary (unused) |
| `VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT` | Allow concurrent resubmit (careful) |

### `VkCommandBufferBeginInfo`

| Field | Type | Meaning |
|-------|------|---------|
| `sType` | `VkStructureType` | `COMMAND_BUFFER_BEGIN_INFO` |
| `pNext` | `const void*` | Usually `nullptr` |
| `flags` | `VkCommandBufferUsageFlags` | e.g. one-time |
| `pInheritanceInfo` | `const VkCommandBufferInheritanceInfo*` | For secondary; nullptr for primary |

### `vkBeginCommandBuffer` / `vkEndCommandBuffer`

```text
VkResult vkBeginCommandBuffer(
  VkCommandBuffer commandBuffer,
  const VkCommandBufferBeginInfo* pBeginInfo
)

VkResult vkEndCommandBuffer(
  VkCommandBuffer commandBuffer
)
```

Recording happens between begin and end. `vkCmd*` calls are only valid while recording.

### `vkResetCommandBuffer`

```text
VkResult vkResetCommandBuffer(
  VkCommandBuffer commandBuffer,
  VkCommandBufferResetFlags flags   // usually 0
)
```

Clears recorded commands so the buffer can be recorded again (pool must allow reset, or reset the whole pool).

### `VkBufferCopy`

| Field | Type | Meaning |
|-------|------|---------|
| `srcOffset` | `VkDeviceSize` | Byte offset in source |
| `dstOffset` | `VkDeviceSize` | Byte offset in destination |
| `size` | `VkDeviceSize` | Bytes to copy |

### `vkCmdCopyBuffer`

```text
void vkCmdCopyBuffer(
  VkCommandBuffer commandBuffer,     // recording CB
  VkBuffer srcBuffer,
  VkBuffer dstBuffer,
  uint32_t regionCount,
  const VkBufferCopy* pRegions
)
```

Records a GPU-side copy. Does not run until the CB is submitted and the GPU executes it.

Both buffers need appropriate `TRANSFER_SRC` / `TRANSFER_DST` usage bits.

---

## 10. Fences and submit

**Purpose of this layer:** (1) send recorded work to the GPU (`vkQueueSubmit`), and
(2) let the CPU know when that work is done (`VkFence`).

### How fences work (detail)

The GPU and CPU run on different timelines. After `vkQueueSubmit`, the CPU might
immediately try to `memcpy` from a staging buffer that a download copy has not
finished writing. That is a race.

A **fence** closes that gap:

1. Create a fence (usually **unsignaled**).
2. Pass it as the last argument to `vkQueueSubmit`.
3. When the GPU finishes that submit, the driver **signals** the fence.
4. `vkWaitForFences` blocks the CPU until the fence is signaled (or times out).
5. After the wait returns successfully, it is safe to read staging memory, free
   temporary buffers used by that submit, or complete a Python `join()`.
6. `vkResetFences` sets it back to unsignaled before the next submit if reusing it.

Mental model: the fence is a doorbell the GPU rings when a batch of work is done;
the CPU sleeps on `vkWaitForFences` until it hears the ring.

**Fence vs semaphore:** fences are for **CPU waits**. Semaphores are for **GPU waits**
between submits. cthreads upload/download and `GpuJob.join` are CPU-wait problems,
so fences are the right tool.

### `VkFenceCreateFlagBits`

| Flag | Meaning |
|------|---------|
| `0` | Created unsignaled (typical) |
| `VK_FENCE_CREATE_SIGNALED_BIT` | Created already signaled |

### `VkFenceCreateInfo`

| Field | Type | Meaning |
|-------|------|---------|
| `sType` | `VkStructureType` | `FENCE_CREATE_INFO` |
| `pNext` | `const void*` | Usually `nullptr` |
| `flags` | `VkFenceCreateFlags` | Usually 0 |

### `vkCreateFence` / `vkDestroyFence`

Standard create/destroy on device.

### `vkResetFences`

```text
VkResult vkResetFences(
  VkDevice device,
  uint32_t fenceCount,
  const VkFence* pFences
)
```

Returns fences to unsignaled so they can be reused on the next submit.

### `vkWaitForFences`

```text
VkResult vkWaitForFences(
  VkDevice device,
  uint32_t fenceCount,
  const VkFence* pFences,
  VkBool32 waitAll,          // VK_TRUE: wait until all signal
  uint64_t timeout           // nanoseconds; UINT64_MAX = forever
)
```

CPU blocks until the fence(s) signal (or timeout).

### `VkSubmitInfo`

| Field | Type | Meaning |
|-------|------|---------|
| `sType` | `VkStructureType` | `SUBMIT_INFO` |
| `pNext` | `const void*` | Usually `nullptr` |
| `waitSemaphoreCount` | `uint32_t` | GPU waits (0 in basic cthreads copies) |
| `pWaitSemaphores` | `const VkSemaphore*` | Semaphores to wait on |
| `pWaitDstStageMask` | `const VkPipelineStageFlags*` | Stages to wait at |
| `commandBufferCount` | `uint32_t` | How many CBs |
| `pCommandBuffers` | `const VkCommandBuffer*` | CBs to execute |
| `signalSemaphoreCount` | `uint32_t` | Semaphores to signal (often 0) |
| `pSignalSemaphores` | `const VkSemaphore*` | Signal list |

### `vkQueueSubmit`

```text
VkResult vkQueueSubmit(
  VkQueue queue,
  uint32_t submitCount,
  const VkSubmitInfo* pSubmits,
  VkFence fence                 // optional; signals when this submit finishes
)
```

Returns when the submit is **queued**, not when the GPU finished. Use the fence to wait on the CPU.

---

## 11. Launch path: descriptors

**Purpose of this layer:** connect shader binding numbers to real `VkBuffer`s.
Without descriptors, a compute shader has no way to know which device memory is
`binding = 1`. Layouts describe the shape; sets hold the actual pointers/handles;
updates and binds install them for a dispatch.

See handle write-ups for descriptor layout/pool/set in section 1.

These are required once compute shaders bind GpuPack buffers.

### `VkDescriptorType` (enum, partial)

| Value | Meaning |
|-------|---------|
| `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` | UBO |
| `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` | SSBO (cthreads scalars + lists) |
| `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC` | Dynamic-offset UBO |
| `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC` | Dynamic-offset SSBO |

### `VkDescriptorSetLayoutBinding`

| Field | Type | Meaning |
|-------|------|---------|
| `binding` | `uint32_t` | Binding number (0 = scalars, 1..N = lists) |
| `descriptorType` | `VkDescriptorType` | `STORAGE_BUFFER` for cthreads |
| `descriptorCount` | `uint32_t` | Usually 1 |
| `stageFlags` | `VkShaderStageFlags` | `VK_SHADER_STAGE_COMPUTE_BIT` |
| `pImmutableSamplers` | `const VkSampler*` | nullptr for buffers |

### `VkShaderStageFlagBits`

| Flag | Meaning |
|------|---------|
| `VK_SHADER_STAGE_COMPUTE_BIT` | Compute shader |
| `VK_SHADER_STAGE_VERTEX_BIT` | Graphics (unused) |
| `VK_SHADER_STAGE_ALL` | All stages |

### `VkDescriptorSetLayoutCreateInfo`

Holds an array of `VkDescriptorSetLayoutBinding`.

### `vkCreateDescriptorSetLayout` / `vkDestroyDescriptorSetLayout`

Create/destroy the layout object on a device.

### `VkDescriptorPoolSize`

| Field | Type | Meaning |
|-------|------|---------|
| `type` | `VkDescriptorType` | e.g. storage buffer |
| `descriptorCount` | `uint32_t` | How many of that type the pool can allocate |

### `VkDescriptorPoolCreateInfo`

| Field | Type | Meaning |
|-------|------|---------|
| `flags` | `VkDescriptorPoolCreateFlags` | e.g. free-set bit if freeing individual sets |
| `maxSets` | `uint32_t` | Max sets allocatable |
| `poolSizeCount` | `uint32_t` | Length of pool sizes |
| `pPoolSizes` | `const VkDescriptorPoolSize*` | Capacities per type |

### `vkCreateDescriptorPool` / `vkDestroyDescriptorPool`

### `VkDescriptorSetAllocateInfo`

| Field | Type | Meaning |
|-------|------|---------|
| `descriptorPool` | `VkDescriptorPool` | Pool |
| `descriptorSetCount` | `uint32_t` | How many sets |
| `pSetLayouts` | `const VkDescriptorSetLayout*` | Layout per set |

### `vkAllocateDescriptorSets` / `vkFreeDescriptorSets`

### `VkDescriptorBufferInfo`

| Field | Type | Meaning |
|-------|------|---------|
| `buffer` | `VkBuffer` | Buffer to bind |
| `offset` | `VkDeviceSize` | Start offset |
| `range` | `VkDeviceSize` | Bytes, or `VK_WHOLE_SIZE` |

### `VkWriteDescriptorSet`

| Field | Type | Meaning |
|-------|------|---------|
| `dstSet` | `VkDescriptorSet` | Set to update |
| `dstBinding` | `uint32_t` | Binding index |
| `dstArrayElement` | `uint32_t` | Usually 0 |
| `descriptorCount` | `uint32_t` | Usually 1 |
| `descriptorType` | `VkDescriptorType` | Must match layout |
| `pImageInfo` | `const VkDescriptorImageInfo*` | For images; nullptr for buffers |
| `pBufferInfo` | `const VkDescriptorBufferInfo*` | Buffer info |
| `pTexelBufferView` | `const VkBufferView*` | Texel buffers; nullptr |

### `vkUpdateDescriptorSets`

```text
void vkUpdateDescriptorSets(
  VkDevice device,
  uint32_t descriptorWriteCount,
  const VkWriteDescriptorSet* pDescriptorWrites,
  uint32_t descriptorCopyCount,
  const VkCopyDescriptorSet* pDescriptorCopies
)
```

CPU-side update; no command buffer needed.

### `vkCmdBindDescriptorSets`

```text
void vkCmdBindDescriptorSets(
  VkCommandBuffer commandBuffer,
  VkPipelineBindPoint pipelineBindPoint,  // COMPUTE
  VkPipelineLayout layout,
  uint32_t firstSet,
  uint32_t descriptorSetCount,
  const VkDescriptorSet* pDescriptorSets,
  uint32_t dynamicOffsetCount,
  const uint32_t* pDynamicOffsets
)
```

---

## 12. Launch path: shaders, pipelines, dispatch

**Purpose of this layer:** turn SPIR-V into something the GPU can run, then launch it.

- **Shader module:** raw SPIR-V wrapped as a Vulkan object.
- **Pipeline layout:** declares descriptor/push-constant interface.
- **Compute pipeline:** prepared program ready to bind.
- **Dispatch:** "run N workgroups of this pipeline."

Recording order in a command buffer is typically: bind pipeline -> bind descriptor
sets -> (optional push constants) -> `vkCmdDispatch`. Then submit + fence wait as in
section 10.

### `VkShaderModuleCreateInfo`

| Field | Type | Meaning |
|-------|------|---------|
| `flags` | `VkShaderModuleCreateFlags` | Usually 0 |
| `codeSize` | `size_t` | SPIR-V size in **bytes** |
| `pCode` | `const uint32_t*` | SPIR-V words |

### `vkCreateShaderModule` / `vkDestroyShaderModule`

### `VkPipelineBindPoint` (enum)

| Value | Meaning |
|-------|---------|
| `VK_PIPELINE_BIND_POINT_COMPUTE` | Compute pipeline |
| `VK_PIPELINE_BIND_POINT_GRAPHICS` | Graphics (unused) |

### `VkPipelineShaderStageCreateInfo`

| Field | Type | Meaning |
|-------|------|---------|
| `stage` | `VkShaderStageFlagBits` | `COMPUTE_BIT` |
| `module` | `VkShaderModule` | Shader module |
| `pName` | `const char*` | Entry point, usually `"main"` |
| `pSpecializationInfo` | `const VkSpecializationInfo*` | Optional constants |

### `VkPipelineLayoutCreateInfo`

| Field | Type | Meaning |
|-------|------|---------|
| `setLayoutCount` | `uint32_t` | Number of descriptor set layouts |
| `pSetLayouts` | `const VkDescriptorSetLayout*` | Layouts |
| `pushConstantRangeCount` | `uint32_t` | 0 if unused |
| `pPushConstantRanges` | `const VkPushConstantRange*` | Optional |

### `vkCreatePipelineLayout` / `vkDestroyPipelineLayout`

### `VkComputePipelineCreateInfo`

| Field | Type | Meaning |
|-------|------|---------|
| `flags` | `VkPipelineCreateFlags` | Usually 0 |
| `stage` | `VkPipelineShaderStageCreateInfo` | Compute stage |
| `layout` | `VkPipelineLayout` | Pipeline layout |
| `basePipelineHandle` | `VkPipeline` | For derivatives; often null |
| `basePipelineIndex` | `int32_t` | Or -1 |

### `vkCreateComputePipelines`

```text
VkResult vkCreateComputePipelines(
  VkDevice device,
  VkPipelineCache pipelineCache,   // optional VK_NULL_HANDLE
  uint32_t createInfoCount,
  const VkComputePipelineCreateInfo* pCreateInfos,
  const VkAllocationCallbacks* pAllocator,
  VkPipeline* pPipelines
)
```

### `vkDestroyPipeline`

### `vkCmdBindPipeline`

```text
void vkCmdBindPipeline(
  VkCommandBuffer commandBuffer,
  VkPipelineBindPoint pipelineBindPoint,
  VkPipeline pipeline
)
```

### `vkCmdDispatch`

```text
void vkCmdDispatch(
  VkCommandBuffer commandBuffer,
  uint32_t groupCountX,
  uint32_t groupCountY,
  uint32_t groupCountZ
)
```

Launches `groupCountX * groupCountY * groupCountZ` workgroups.
Each workgroup size comes from the shader `layout(local_size_x = …)`.

Rough element coverage for 1D:

```text
groupsX = ceil(elementCount / local_size_x)
```

### `vkCmdPushConstants` (optional)

```text
void vkCmdPushConstants(
  VkCommandBuffer commandBuffer,
  VkPipelineLayout layout,
  VkShaderStageFlags stageFlags,
  uint32_t offset,
  uint32_t size,
  const void* pValues
)
```

Small fast uniforms. In cthreads, scalar SSBO remains writeback source of truth if both are used.

---

## 13. Launch path: barriers (overview)

**Purpose:** force ordering and memory visibility between GPU commands. The GPU may
overlap or reorder work. A barrier says: "finish these earlier writes, then allow
these later reads/writes."

**When fences are enough:** if the CPU submits a copy-only command buffer and
`vkWaitForFences` before touching staging memory, the CPU wait provides the needed
ordering for that CPU read. No barrier required for that pattern.

**When barriers are required:** if the **same** command buffer (or back-to-back GPU
work without a CPU wait) does `copy into buffer Y` then `dispatch that reads Y`,
insert a pipeline barrier between them so the shader cannot read stale data.

### `VkPipelineStageFlagBits` (partial)

| Flag | Meaning |
|------|---------|
| `VK_PIPELINE_STAGE_TRANSFER_BIT` | Copy engine |
| `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` | Compute shader |
| `VK_PIPELINE_STAGE_HOST_BIT` | Host access |
| `VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT` | Earliest |
| `VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT` | Latest |

### `VkAccessFlagBits` (partial)

| Flag | Meaning |
|------|---------|
| `VK_ACCESS_TRANSFER_WRITE_BIT` | Copy write |
| `VK_ACCESS_TRANSFER_READ_BIT` | Copy read |
| `VK_ACCESS_SHADER_READ_BIT` | Shader read |
| `VK_ACCESS_SHADER_WRITE_BIT` | Shader write |
| `VK_ACCESS_HOST_READ_BIT` | CPU read |
| `VK_ACCESS_HOST_WRITE_BIT` | CPU write |

### `vkCmdPipelineBarrier`

Records a barrier between earlier and later commands in the same CB. Exact struct
fields (`VkMemoryBarrier`, `VkBufferMemoryBarrier`) are filled when implementing
the launch path; until then, CPU fence waits after submit are enough for
upload/download helpers that do not dispatch in the same CB.

---

## 14. Function pointer typedefs (`PFN_*`)

Vulkan headers define:

```text
typedef <ReturnType> (VKAPI_PTR *PFN_vkXxx)(<args...>);
```

Examples:

| Typedef | Points to |
|---------|-----------|
| `PFN_vkGetInstanceProcAddr` | `vkGetInstanceProcAddr` |
| `PFN_vkCreateInstance` | `vkCreateInstance` |
| `PFN_vkCreateBuffer` | `vkCreateBuffer` |
| `PFN_vkCmdCopyBuffer` | `vkCmdCopyBuffer` |
| … | every entry point |

cthreads stores these on `Context` and loads them through `vkGetInstanceProcAddr`
(see [05-dynamic-loading.md](./05-dynamic-loading.md)).

### `vkGetInstanceProcAddr`

```text
PFN_vkVoidFunction vkGetInstanceProcAddr(
  VkInstance instance,   // NULL for a few globals; real instance otherwise
  const char* pName      // e.g. "vkCreateBuffer"
)
```

Returns a generic function pointer or `nullptr` if unavailable.

---

## 15. Quick index by name

### Handles / scalars

`VkBool32`, `VkDeviceSize`, `VK_NULL_HANDLE`, `VK_WHOLE_SIZE`,  
`VkInstance`, `VkPhysicalDevice`, `VkDevice`, `VkQueue`,  
`VkBuffer`, `VkDeviceMemory`, `VkCommandPool`, `VkCommandBuffer`, `VkFence`,  
`VkDescriptorSetLayout`, `VkDescriptorPool`, `VkDescriptorSet`,  
`VkShaderModule`, `VkPipelineLayout`, `VkPipeline`, `VkPipelineCache`

### Enums / flags (high traffic)

`VkResult`, `VkStructureType`, `VkPhysicalDeviceType`,  
`VkQueueFlagBits`, `VkMemoryPropertyFlagBits`, `VkBufferUsageFlagBits`,  
`VkSharingMode`, `VkCommandBufferLevel`, `VkCommandPoolCreateFlagBits`,  
`VkCommandBufferUsageFlagBits`, `VkDescriptorType`, `VkShaderStageFlagBits`,  
`VkPipelineBindPoint`, `VkPipelineStageFlagBits`, `VkAccessFlagBits`

### Structs (high traffic)

`VkApplicationInfo`, `VkInstanceCreateInfo`,  
`VkDeviceQueueCreateInfo`, `VkDeviceCreateInfo`,  
`VkPhysicalDeviceProperties`, `VkQueueFamilyProperties`,  
`VkPhysicalDeviceMemoryProperties`, `VkMemoryType`, `VkMemoryHeap`,  
`VkBufferCreateInfo`, `VkMemoryRequirements`, `VkMemoryAllocateInfo`,  
`VkCommandPoolCreateInfo`, `VkCommandBufferAllocateInfo`, `VkCommandBufferBeginInfo`,  
`VkBufferCopy`, `VkFenceCreateInfo`, `VkSubmitInfo`,  
`VkDescriptorSetLayoutBinding`, `VkDescriptorBufferInfo`, `VkWriteDescriptorSet`,  
`VkShaderModuleCreateInfo`, `VkPipelineShaderStageCreateInfo`,  
`VkPipelineLayoutCreateInfo`, `VkComputePipelineCreateInfo`

### Functions already used in cthreads GPU code

`vkGetInstanceProcAddr`,  
`vkCreateInstance`, `vkDestroyInstance`,  
`vkEnumeratePhysicalDevices`,  
`vkGetPhysicalDeviceProperties`, `vkGetPhysicalDeviceQueueFamilyProperties`,  
`vkGetPhysicalDeviceMemoryProperties`,  
`vkCreateDevice`, `vkDestroyDevice`, `vkGetDeviceQueue`,  
`vkCreateBuffer`, `vkDestroyBuffer`, `vkGetBufferMemoryRequirements`,  
`vkAllocateMemory`, `vkFreeMemory`, `vkBindBufferMemory`,  
`vkMapMemory`, `vkUnmapMemory`,  
`vkCreateCommandPool`, `vkDestroyCommandPool`,  
`vkAllocateCommandBuffers`, `vkFreeCommandBuffers`, `vkResetCommandBuffer`,  
`vkBeginCommandBuffer`, `vkEndCommandBuffer`, `vkCmdCopyBuffer`,  
`vkCreateFence`, `vkDestroyFence`, `vkQueueSubmit`, `vkWaitForFences`, `vkResetFences`

### Functions needed for the launch path (not all may be loaded yet)

`vkCreateDescriptorSetLayout`, `vkDestroyDescriptorSetLayout`,  
`vkCreateDescriptorPool`, `vkDestroyDescriptorPool`,  
`vkAllocateDescriptorSets`, `vkFreeDescriptorSets`, `vkUpdateDescriptorSets`,  
`vkCreateShaderModule`, `vkDestroyShaderModule`,  
`vkCreatePipelineLayout`, `vkDestroyPipelineLayout`,  
`vkCreateComputePipelines`, `vkDestroyPipeline`,  
`vkCmdBindPipeline`, `vkCmdBindDescriptorSets`, `vkCmdDispatch`,  
`vkCmdPipelineBarrier`, `vkCmdPushConstants` (optional)

When adding any of these in C++, also add the matching `PFN_*` field on `Context`,
resolve it in init, and clear it in shutdown.

---

## How to use this file

1. When reading `context.cpp` / `memory.cpp`, look up each `Vk*` / `vk*` here.
2. When implementing the launch path, start from sections 11-13 and extend Context PFNs.
3. For intuition (why fences exist, why staging exists), return to chapters 02-08.

This reference intentionally over-explains. Prefer it over skimming random internet snippets that mix graphics-only APIs into compute work.
