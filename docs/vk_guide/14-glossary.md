# 14 — Glossary

| Term | Meaning in this project |
|------|-------------------------|
| **Vulkan** | Low-level cross-vendor GPU API |
| **Loader** | `vulkan-1.dll` / `libvulkan.so.1` — finds drivers, exports `vkGetInstanceProcAddr` |
| **ICD** | Installable Client Driver — vendor Vulkan implementation inside the GPU driver |
| **SDK** | LunarG Vulkan SDK — headers and tools for *developers* |
| **Instance** | App-wide Vulkan connection (`VkInstance`) |
| **Physical device** | A GPU enumerated by the loader |
| **Logical device** | Opened GPU connection (`VkDevice`) |
| **Queue family** | Group of queues with shared capabilities (compute/graphics/…) |
| **Queue** | Submission port (`VkQueue`) |
| **Handle** | Opaque id for a Vulkan object (`VkBuffer`, …) |
| **PFN_*** | Typed function pointer to a Vulkan entry point |
| **Buffer (`VkBuffer`)** | Buffer resource object (needs memory bound) |
| **Device memory** | Allocated GPU memory slab (`VkDeviceMemory`) |
| **Host-visible** | Memory the CPU can map |
| **Host-coherent** | Mapped writes/reads without manual flush |
| **Device-local** | GPU-fast memory; not our persistently mapped SSBO path |
| **Staging buffer** | Host-visible buffer used only for upload/download traffic |
| **Map** | Get a CPU pointer into host-visible device memory |
| **SSBO** | Storage buffer — shader-readable/writable buffer block |
| **Descriptor** | Binding that tells a shader which buffer is at binding i |
| **std430** | Packing rules for storage buffer structs/arrays |
| **Command pool** | Owns command buffers for a queue family |
| **Command buffer** | Recorded list of GPU commands |
| **Submit** | Give a command buffer to a queue for execution |
| **Fence** | CPU-waitable completion signal for a submit |
| **Semaphore** | GPU-side sync (we mostly ignore for now) |
| **Barrier** | GPU ordering/visibility constraint between commands |
| **SPIR-V** | Portable shader IR Vulkan consumes |
| **Shader module** | Vulkan object holding SPIR-V |
| **Pipeline (compute)** | Prepared compute shader + layout |
| **Dispatch** | Launch compute workgroups (`vkCmdDispatch`) |
| **Workgroup / local size** | Group of invocations that run together |
| **GpuPack** | Per-launch scalar SSBO + list SSBOs (binding convention) |
| **Marshal** | Copy Python values into native/GPU pack storage |
| **Writeback** | Copy native/GPU results into the same Python objects |
| **Join** | Wait for GPU job completion then writeback |

## Abbreviations

| Short | Full |
|-------|------|
| H2D | Host to device (upload) |
| D2H | Device to host (download) |
| SSBO | Shader Storage Buffer Object |
| GIPA | `vkGetInstanceProcAddr` |
| BDA | Buffer device address (rejected for v1) |
| AoS | Array of structures (Threadable lists later) |
