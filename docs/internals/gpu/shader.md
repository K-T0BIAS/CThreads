# Shader cache and create_entry

Sources:

- `src/cthreads/cpp/gpu/headers/shader_cache.hpp`
- `src/cthreads/cpp/gpu/headers/shader.hpp`
- `src/cthreads/cpp/gpu/impl/shader_cache.cpp`
- `src/cthreads/cpp/gpu/impl/shader.cpp`

Namespace: `cthreads::gpu::shader`.

Depends on: [Context](./context.md). Used by: [Descriptors](./descriptors.md) (set layout), [Module / launch](./module.md).

## Purpose

Building a compute pipeline from SPIR-V is expensive. Doing it on every launch would waste time. The shader cache stores, per kernel symbol, the reusable Vulkan objects that stay identical across launches:

- Shader module (SPIR-V wrapped for Vulkan)
- Descriptor set layout (binding convention schema)
- Pipeline layout (how sets attach to the pipeline)
- Compute pipeline (compiled program ready to bind)
- Binding count metadata

Per-launch objects (GpuPack buffers, descriptor sets, fences) are not stored here.

## Access rights

- Writers (registry / testing, via `add` once friended): insert new entries.
- Everyone else: `get` returns a const reference.
- Context shutdown: `clear` destroys all Vulkan objects, then empties the map.

Entries are not mutated in place after insert. Duplicate `add` of the same key throws.

## Technical terms

- SPIR-V: binary intermediate language for shaders. Vulkan drivers consume SPIR-V, not GLSL text, at runtime.
- Shader module: Vulkan object created from SPIR-V bytes (`VkShaderModule`).
- Compute pipeline: prepared compute program plus layout (`VkPipeline` with compute bind point).
- Pipeline layout: declares which descriptor set layouts (and optional push constants) a pipeline uses.
- Descriptor set layout: schema of bindings; see [Descriptors](./descriptors.md).
- Entry point name: function name inside the shader. cthreads uses `"main"`.
- Push constants: tiny values pushed in the command buffer without a buffer object. Not used in create_entry today; scalars live in binding 0.

## Struct `ShaderCacheEntry`

Move-only. Copying would duplicate Vulkan handles and double-destroy them.

Fields:

- `shader_module`: may remain non-null after pipeline create (kept for simplicity; `clear` destroys it).
- `set_layout`: storage buffer bindings `0 .. binding_count-1` (scalars then lists).
- `pipeline_layout`: layout used when creating the pipeline.
- `pipeline`: compute pipeline handle.
- `binding_count`: number of storage buffer bindings (at least 1).

## Class `ShaderCache`

Process-wide singleton via `getInstance()`.

### `add(key, entry)` (private until registry friend exists)

Moves the entry into an internal `unordered_map`. Returns a const reference to the map node. Throws if the key already exists.

### `get(key)`

Returns a const reference to an existing entry. Throws if missing.

### `clear(context)`

Destroys every entry's Vulkan objects through `destroy_entry`, then clears the map. Called from Context shutdown before the logical device is destroyed.

### Destructor

Only drops the map. Shutdown must have already cleared handles, because static destruction order versus Context is undefined.

## Function `create_entry`

Declared in `shader.hpp`, implemented in `shader.cpp`.

Builds a complete `ShaderCacheEntry` from SPIR-V words and a binding count. Does not insert into the cache; the caller (registry or test) calls `add`.

### Parameters

- `context`: ready Context with create and destroy entry points.
- `spirv`: pointer to SPIR-V code as `uint32_t` words.
- `spirv_word_count`: number of words (byte size divided by 4).
- `binding_count`: storage buffer bindings (`>= 1`).

### Steps

1. Validate ready device, non-empty SPIR-V, binding count, and create PFNs.
2. `vkCreateShaderModule` from the SPIR-V bytes.
3. Build `binding_count` layout bindings, each `STORAGE_BUFFER`, compute stage, then `vkCreateDescriptorSetLayout`.
4. `vkCreatePipelineLayout` with that single set layout and no push constant ranges.
5. `vkCreateComputePipelines` with stage compute, module, entry `"main"`, and the pipeline layout.
6. On any failure after partial success, call `destroy_entry` and throw.

### Return

Owned `ShaderCacheEntry`. Move it into `ShaderCache::add`, or destroy it with `destroy_entry` if you abandon it.

## Function `destroy_entry`

Destroys pipeline, pipeline layout, set layout, and shader module in that order (children before parents), then nulls handles. Used by `ShaderCache::clear` and by fail paths in `create_entry`.

If the device handle is already gone, it only nulls fields (shutdown edge case).

## Key Vulkan calls

| Call | Role |
|-|-|
| `vkCreateShaderModule` | Wrap SPIR-V |
| `vkCreateDescriptorSetLayout` | Binding convention schema |
| `vkCreatePipelineLayout` | Attach set layout to pipeline interface |
| `vkCreateComputePipelines` | Compile compute pipeline |
| Matching `vkDestroy*` | Tear down in `destroy_entry` / `clear` |

## Where SPIR-V comes from

Issue 3 feeds committed SPIR-V (or library-built smoke shaders) under `_ext.gpu.testing`. Later, `@Gpu` compilation will produce SPIR-V and call the same `create_entry` path.

## What create_entry does not do

- Allocate descriptor sets or pools
- Point bindings at a GpuPack
- Record dispatch
- Register the entry in the cache (caller must `add`)

Those are separate steps on the launch timeline.

## Relationship to the cache key

The cache key is a string symbol (stable kernel name). Long term, content hashing of SPIR-V may be stored inside the entry for invalidation. Today the contract is: one symbol maps to one immutable entry for the process lifetime after `add`.
