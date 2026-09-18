# 11 — SPIR-V, pipelines, and compute dispatch

## Shader languages in our stack

| Form | Role |
|------|------|
| GLSL compute (human-written or emitted) | Source you can read |
| SPIR-V | Binary IR Vulkan drivers consume |
| `VkShaderModule` | Vulkan object wrapping SPIR-V bytes |
| `VkPipeline` (compute) | Compiled shader + layout ready to bind |

Flow:

```text
GLSL -> (shaderc / glslang at library build or runtime) -> SPIR-V bytes
     -> vkCreateShaderModule
     -> vkCreateComputePipelines
```

Users never run `glslangValidator` themselves. Either:

- we commit `.spv` files, or
- we compile with shaderc inside `_ext` when `CTHREADS_GPU=ON`.

## Compute shader shape (intuition)

```glsl
#version 450
layout(local_size_x = 64) in;  // workgroup size

layout(set=0, binding=0, std430) buffer Scalars { int n; float a; } scalars;
layout(set=0, binding=1, std430) buffer X { float data[]; } x;
layout(set=0, binding=2, std430) buffer Y { float data[]; } y;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(scalars.n)) return;
    y.data[i] = scalars.a * x.data[i] + y.data[i];
}
```

### Workgroups and local size

- `local_size_x = 64` means each workgroup runs 64 invocations.
- `vkCmdDispatch(groupCountX, 1, 1)` launches that many workgroups.
- Global id roughly: `groupIndex * local_size + localIndex`.

For `n` elements:

```text
groups = ceil(n / 64)
vkCmdDispatch(groups, 1, 1)
```

cthreads may also support serial-looking `@Gpu` bodies later (`local_size=1`). The product API stays launch/wait; the emitter chooses a parallelism model and documents it.

## Pipeline layout

Connects:

- descriptor set layouts (which bindings exist)
- push constant ranges (optional)

Even if you use a scalar SSBO, you might later mirror hot scalars into push constants for speed. Writeback truth remains the scalar SSBO.

## Creating a compute pipeline (conceptual)

```text
VkShaderModuleCreateInfo <- SPIR-V code
vkCreateShaderModule

VkPipelineShaderStageCreateInfo stage: COMPUTE, module, entry "main"
VkComputePipelineCreateInfo: stage + pipelineLayout
vkCreateComputePipelines
```

Pipelines should be cached by shader hash so launches do not rebuild every time.

## Dispatch command buffer (full story)

```text
Begin command buffer
  // ensure uploads visible to compute (barrier) when needed
  vkCmdBindPipeline(COMPUTE, pipeline)
  vkCmdBindDescriptorSets(... pack's set ...)
  vkCmdDispatch(groupsX, 1, 1)
End
vkQueueSubmit(..., fence)
```

`GpuJob.join()`:

```text
vkWaitForFences
download GpuPack -> Python writeback
```

## Barriers in one sentence

A **pipeline barrier** tells the GPU: finish these earlier commands' memory effects before later commands use the data.

Example: copy into `y` finished before compute reads/writes `y`.

CPU-side round-trips can rely on fence waits.
In-GPU sequences need barriers between copy and dispatch (and between dispatches in a batch).

## What to skip in graphics tutorials

When reading vkguide / vulkan-tutorial, skip:

- swapchain
- render pass
- framebuffer
- vertex buffers for drawing

Keep:

- buffer creation
- command buffers
- descriptors
- compute pipeline / dispatch (some tutorials bury this)
