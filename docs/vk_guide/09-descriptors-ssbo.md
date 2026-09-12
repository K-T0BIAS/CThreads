# 09 — Descriptors and SSBOs (how shaders see buffers)

The memory layer creates buffers. The launch path lets shaders read them. This chapter is the bridge.

## The problem descriptors solve

You have device-local buffers on the CPU/C++ side (`VkBuffer` handles).
A shader is SPIR-V running on the GPU. It cannot see your C++ variables.

You need a **binding table**: "shader binding 0 is this buffer, binding 1 is that buffer."

That table is built with:

- Descriptor set layout (the schema: binding i is a storage buffer)
- Descriptor pool + descriptor set (an instance of that schema)
- Descriptor writes (`vkUpdateDescriptorSets`) pointing at actual `VkBuffer`s
- `vkCmdBindDescriptorSets` before dispatch

## SSBO = storage buffer

In GLSL compute:

```glsl
layout(set = 0, binding = 1, std430) buffer XBlock {
    float data[];
} x;
```

That means:

- descriptor set 0
- binding 1
- storage buffer
- std430 packing
- unsized array of floats at the end of the block

On the C++ side, binding 1's descriptor must reference the `VkBuffer` that holds those floats.

## cthreads binding convention

| Binding | Contents |
|---------|----------|
| 0 | Scalar SSBO (`std430` struct: `n`, `a`, …) |
| 1 | First list SSBO |
| 2 | Second list SSBO |
| … | More lists |

Saxpy example:

```glsl
layout(set=0, binding=0, std430) buffer Scalars { int n; float a; } scalars;
layout(set=0, binding=1, std430) buffer X { float data[]; } x;
layout(set=0, binding=2, std430) buffer Y { float data[]; } y;
```

No pointers inside `Scalars`. Bindings do the wiring.

## Descriptor types we care about

| Vulkan type | GLSL idea |
|-------------|-----------|
| `STORAGE_BUFFER` | `buffer { ... }` (read/write) |
| `UNIFORM_BUFFER` | `uniform` block (we avoid for mutable pack) |

cthreads uses **storage buffers for scalars and lists** (binding convention = all SSBO).

## Lifecycle sketch (launch path)

```text
1. Create descriptor set layout (bindings 0..N as STORAGE_BUFFER)
2. Create pipeline layout (includes that set layout)
3. Create compute pipeline (shader module + layout)
4. Create descriptor pool; allocate one set
5. For each launch:
     write descriptors to point at this GpuPack's buffers
     record: bind pipeline, bind set, dispatch
6. On shutdown: destroy sets/pool/pipeline/layout/module (order matters)
```

## Why update descriptors per launch?

Because each job has its own GpuPack buffers (or at least different list buffers).
The layout stays cached; the **buffer handles** inside the set change.

CB reuse and pipeline caching optimize recording/submit; the descriptor model stays.

## Ranges

When writing a descriptor for a buffer you specify offset + range (often `VK_WHOLE_SIZE` or `buffer.size`).
For our tightly packed arrays, whole buffer / exact size both work if the buffer was created for that array only.

## Mental model

```text
Shader source says: binding 1 is float array x
Descriptor set says: binding 1 -> VkBuffer handle of list x
Dispatch runs: loads/stores hit that memory
```

If descriptors are wrong, you get garbage or crashes — validation layers help.
