# 15 — Checklist: concepts contributors should be able to explain

Use this after reading. If a concept cannot be explained in plain language, revisit that chapter.

## Foundations

- [ ] Why cthreads GPU uses normal Python types, not a public DeviceBuffer
- [ ] Difference between loader, SDK, and GPU driver
- [ ] Why we dynamic-load Vulkan instead of linking for default wheels
- [ ] What `VK_NULL_HANDLE` means

## Context layer

- [ ] Instance vs physical device vs logical device vs queue
- [ ] Why instance-level functions must be resolved with a real instance
- [ ] How we pick a compute queue family and prefer discrete GPUs
- [ ] Shutdown order (device before instance before FreeLibrary)

## Memory / transfer layer

- [ ] Why a buffer needs both `VkBuffer` and `VkDeviceMemory`
- [ ] What `find_memory_type` does with `type_bits` and property flags
- [ ] Staging vs device-local (`BufferKind`)
- [ ] Why device-local SSBOs are not persistently mapped
- [ ] Upload path: memcpy staging -> GPU copy -> device-local
- [ ] Download path reverse
- [ ] Why size 0 buffers are rejected
- [ ] What a fence wait guarantees after `vkQueueSubmit`

## Launch path / GpuPack / shaders

- [ ] What an SSBO is in GLSL
- [ ] cthreads binding 0 = scalars, 1..N = lists convention
- [ ] What std430 padding is and why host/GPU must match
- [ ] SPIR-V vs GLSL vs pipeline vs dispatch
- [ ] What GpuPack contains (binding convention)
- [ ] Why GPU jobs are launch/wait only (no mid-run sync)

## When stuck

1. Find the term in [14-glossary.md](./14-glossary.md).
2. Open the matching chapter from [README.md](./README.md).
3. Open the matching file in [13-map-to-our-code.md](./13-map-to-our-code.md).
4. If still unclear, ask with the chapter number + expected vs observed behavior in code.
