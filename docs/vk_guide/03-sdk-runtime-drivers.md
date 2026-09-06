# 03 — SDK vs runtime vs drivers

This chapter clears the most common confusion: **"Do end users need the Vulkan SDK?"**

**Short answer for cthreads:**  

- **End users:** GPU drivers only (they typically already have `vulkan-1.dll` / `libvulkan.so.1`).  
- **Contributors building with `CTHREADS_GPU=ON`:** Vulkan SDK (headers) + drivers.  
- **Default CPU wheels do not hard-link** `vulkan-1.lib`.

## Three different things

### 1. GPU driver (required to *run* Vulkan apps)

NVIDIA / AMD / Intel install a **Vulkan Installable Client Driver (ICD)** with the graphics driver.

That is what makes games and compute apps find a GPU.

Sanity checks on Windows:

- Device Manager shows the GPU.
- `vulkaninfo` (from SDK tools) lists the GPU if the runtime works.

### 2. Vulkan loader (`vulkan-1.dll` / `libvulkan.so.1`)

The **loader** is a small library that:

- finds ICDs
- exposes `vkGetInstanceProcAddr`
- dispatches calls to the right driver

On Windows it is typically `vulkan-1.dll` (comes with the driver / runtime).
On Linux it is `libvulkan.so.1` (package like `vulkan-icd-loader` + vendor ICD).

The Context bootstrap does:

```text
LoadLibrary("vulkan-1.dll") / dlopen("libvulkan.so.1")
-> get vkGetInstanceProcAddr
-> resolve every other function by name
```

So the **process must find that DLL/SO at runtime**. Linking the import library at build time is optional; we chose **not** to require it for default CPU builds.

### 3. Vulkan SDK (LunarG) — mainly for *developers*

The SDK gives you:

| Piece | Use |
|-------|-----|
| Headers (`vulkan/vulkan.h`) | Compile C++ that mentions `VkBuffer`, `PFN_vkCreateInstance`, … |
| `vulkaninfo` | Diagnose devices/extensions |
| Validation layers | Catch API misuse while developing |
| Shader tools (`glslangValidator`, etc.) | Compile GLSL -> SPIR-V (we may use shaderc in-library later) |

**End users of cthreads do not install the SDK** just to `pip install` a CPU wheel.
**Contributors** install the SDK so `find_package(Vulkan)` can see headers when `CTHREADS_GPU=ON`.

## How this shows up in our CMake

When `CTHREADS_GPU=ON`:

- CMake runs `find_package(Vulkan REQUIRED)` — needs SDK/headers on the machine building `_ext`.
- We compile `context.cpp`, `memory.cpp`, …
- We define `CTHREADS_WITH_GPU=1`.
- We still **do not** need to link `Vulkan::Vulkan` if we resolve symbols dynamically (our design).

When `CTHREADS_GPU=OFF`:

- GPU sources are not compiled.
- Python `cthreads.gpu` soft-fails (`available()` false / `VulkanNotBuiltError`).

## Practical Windows setup (contributors)

1. Install recent GPU drivers.
2. Install [LunarG Vulkan SDK](https://vulkan.lunarg.com/).
3. Open **x64 Native Tools** / VS Dev Cmd.
4. Build:

```bat
set CMAKE_ARGS=-DCTHREADS_GPU=ON
pip install -e . -v
```

Look for CMake status: `cthreads GPU: ON (Vulkan)`.

5. Smoke:

```python
from cthreads import gpu
print(gpu.available(), gpu.device_name() if gpu.available() else None)
```

## What can go wrong

| Symptom | Likely cause |
|---------|----------------|
| `VulkanLoaderNotFound` | No `vulkan-1.dll` on PATH / not installed with driver |
| `VulkanNoDevice` | Loader ok but no compute-capable ICD / bad driver |
| `missing vkDestroyInstance` style bugs | Resolved instance functions with null instance (we fixed this) |
| CMake cannot find Vulkan | SDK not installed or env not visible to the build |
| Tiny wheel / no GPU log | `CTHREADS_GPU` not actually ON (wrong shell env, or cached CMake) |

## Mental picture

```text
Your C++ (_ext)
    |  dynamic LoadLibrary
    v
vulkan-1.dll  (LOADER)
    |  finds
    v
nvoglv64.dll / amdvlk / igvx64 ...  (ICD / DRIVER)
    |
    v
GPU hardware
```

Headers from the SDK are only a **compile-time** description of that API.
They are not the DLL.
