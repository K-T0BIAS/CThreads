# 05 — Dynamic loading and `PFN_*` function pointers

## Why we do not link `vulkan-1.lib` by default

If you link the Vulkan import library:

- Every machine that imports `_ext` needs the loader present **or** load fails at process start.
- CPU-only users would pay a Vulkan dependency they do not need.

cthreads design:

- Default build: no Vulkan.
- `CTHREADS_GPU=ON`: compile GPU code against **headers only**, resolve symbols at runtime.

## What a `PFN_` is

Vulkan headers define typedefs like:

```cpp
typedef VkResult (VKAPI_PTR *PFN_vkCreateBuffer)(VkDevice, const VkBufferCreateInfo*, ...);
```

So `PFN_vkCreateBuffer` means: "pointer to a function with that signature."

In `Context` we store:

```cpp
PFN_vkCreateBuffer vkCreateBuffer = nullptr;
```

After init:

```cpp
context.vkCreateBuffer(device, &info, nullptr, &buffer);
```

That is a normal indirect call through a function pointer.

## The bootstrap chain

```text
1. LoadLibrary / dlopen
2. GetProcAddress / dlsym("vkGetInstanceProcAddr")
3. vkGetInstanceProcAddr(instance_or_null, "vkCreateInstance")
4. create instance
5. vkGetInstanceProcAddr(real_instance, "vkCreateBuffer")  // etc.
```

The Context helper `get_fn<PFN>(context, instance, "name")`:

1. Calls `vkGetInstanceProcAddr`.
2. Throws if null (`cthreads.gpu.VulkanInitFailed: missing …`).
3. Casts to the typed `PFN_*`.

## Global vs instance vs device level (practical rules)

You do not need the full spec table. Use these project rules:

1. **Before instance exists:** only resolve truly global entry points (`vkCreateInstance`, …) with `VK_NULL_HANDLE`.
2. **After instance exists:** resolve everything else we need with `c.instance`.
3. **After device exists:** we still resolve device commands via `vkGetInstanceProcAddr(instance, name)` (loader trampoline). That matches our code. (Advanced apps often use `vkGetDeviceProcAddr`; not required for our path.)

## Why so many pointers on `Context`?

Buffer, command, and fence functions are loaded once during Context init and cleared on shutdown.

Groups in `context.hpp`:

1. Instance / device bootstrap
2. Buffer + memory
3. Command pool / command buffer / copy / fence / submit

When adding a new Vulkan call in `memory.cpp`, first check: is its `PFN_` on `Context` and loaded in `create_instance_and_device`? If not, add it there.

## Errors contributors will see

| Message prefix | Meaning |
|----------------|---------|
| `VulkanLoaderNotFound` | DLL/SO missing |
| `VulkanInitFailed: missing X` | `get_fn` returned null for name `X` |
| `VulkanNoDevice` | No compute-capable GPU |
| `VulkanInitFailed: vkCreate… failed` | Driver rejected create |

Python maps these string prefixes to exception types in `cthreads.gpu.frontend.errors`.
