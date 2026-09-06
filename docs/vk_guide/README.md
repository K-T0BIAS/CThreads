# Vulkan guide for cthreads (compute path)

This folder is a **project-specific** Vulkan tutorial for **cthreads contributors**.
It covers the compute path used by the GPU backend: buffers, copies, descriptors,
shaders, and launch/wait — **not** the full graphics stack (swapchains, render
passes, images, and so on).

Architecture choices here match the cthreads GPU design (same Python types as the
CPU backend, GpuPack option 5, device-local + staging). Implementation status in
the tree may move faster or slower than any particular roadmap document; treat
this guide as the conceptual reference, and the source under `src/cthreads/cpp/gpu/`
as ground truth for what is already landed.

## Audience

Contributors who know C++ and the cthreads CPU model (`@Thread`, pack, marshal,
writeback), and who may know little or no Vulkan. That is a normal starting point.

## How to read

Read in order the first time. Later use the glossary and the "map to the codebase"
chapter as a reference.

| # | File | Topics |
|---|------|--------|
| 00 | [00-read-me-first.md](./00-read-me-first.md) | Goals, out of scope, Vulkan vs CUDA/OpenGL |
| 01 | [01-cthreads-gpu-big-picture.md](./01-cthreads-gpu-big-picture.md) | How GPU fits next to CPU pack/marshal |
| 02 | [02-mental-model.md](./02-mental-model.md) | Objects, handles, explicit control |
| 03 | [03-sdk-runtime-drivers.md](./03-sdk-runtime-drivers.md) | SDK vs drivers vs `vulkan-1.dll` |
| 04 | [04-instance-device-queue.md](./04-instance-device-queue.md) | Connecting to a GPU (`Context`) |
| 05 | [05-dynamic-loading.md](./05-dynamic-loading.md) | `LoadLibrary` / `PFN_*` resolution |
| 06 | [06-buffers-and-memory.md](./06-buffers-and-memory.md) | `VkBuffer`, memory types, bind |
| 07 | [07-staging-upload-download.md](./07-staging-upload-download.md) | Device-local + staging copies |
| 08 | [08-commands-fences.md](./08-commands-fences.md) | Command buffers, submit, fences |
| 09 | [09-descriptors-ssbo.md](./09-descriptors-ssbo.md) | How shaders see buffers |
| 10 | [10-std430-layouts.md](./10-std430-layouts.md) | Scalar struct packing |
| 11 | [11-spirv-pipelines-dispatch.md](./11-spirv-pipelines-dispatch.md) | Shaders, pipelines, `dispatch` |
| 12 | [12-gpupack-marshal.md](./12-gpupack-marshal.md) | Option 5 pack end-to-end |
| 13 | [13-map-to-our-code.md](./13-map-to-our-code.md) | Files in `src/cthreads/cpp/gpu/` |
| 14 | [14-glossary.md](./14-glossary.md) | Terms in one place |
| 15 | [15-checklist.md](./15-checklist.md) | Concepts a contributor should be able to explain |

## Official docs (optional later)

- [Vulkan Guide](https://vkguide.dev/) — general tutorial (more graphics-heavy)
- [Vulkan Tutorial](https://vulkan-tutorial.com/) — classic; skip swapchain chapters for this project
- [Khronos Vulkan Spec](https://registry.khronos.org/vulkan/) — reference, not a textbook

This guide is intentionally longer and more hand-holding than those, and tied to
**cthreads** architecture decisions rather than a generic hello-triangle path.
