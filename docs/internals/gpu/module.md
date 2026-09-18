# Launch path (`SpawnedGpuKernel`)

Sources:

- `src/cthreads/cpp/gpu/headers/module.hpp`
- `src/cthreads/cpp/gpu/impl/module.cpp`

Namespace: `cthreads::gpu`.

Depends on: [Context](./context.md), [Pack](./pack.md), [Descriptors](./descriptors.md), [Shader](./shader.md).

## Purpose

`launch_gpu_kernel` is the GPU analogue of CPU `spawn_from_meta`: build a `GpuPack`, wire descriptors, record bind+dispatch, submit with a fence, and return a job handle. `SpawnedGpuKernel::join` waits on that fence, downloads ref lists into the same Python objects, then releases Vulkan state. There is no OS worker thread and no mid-run `sync_state`.

Public `gpu()` / `@Gpu` (later) will call these same types. Product pybind:
`_ext.gpu.launch_gpu_kernel` + `_ext.gpu.GpuJob`. Tests register smoke SPIR-V
via `_ext.gpu.testing.register_smoke_saxpy`, then launch on the product path.

## Technical terms

- Fence: CPU waits until the submitted dispatch has finished.
- Writeback: copy device list SSBOs back into the kept Python `list` objects (`pass_as` ref).
- Per-job command buffer + fence: checked out from Context `LaunchEngine` for the job lifetime; returned on join (supports overlapping launches). The command **pool** is process-lifetime on Context.

## Struct `SpawnedGpuKernel`

Owns per-launch GPU handles plus writeback inputs:

| Field | Role |
|-|-|
| `pack` | Device-local scalar + list SSBOs |
| `descriptor_pool` / `descriptor_set` | Wired to this pack |
| `command_pool` / `command_buffer` | Recorded dispatch |
| `fence` | Signals when submit completes |
| `values_keep` | Python args kept alive for list writeback |
| `writeback_lists` | Plan of ref list slots to download on join |

## Function `launch_gpu_kernel`

Takes `meta` + `ordered_values` (same shape as the docstring on `module.hpp`). Registers nothing in the shader cache; the symbol must already exist. Returns `shared_ptr<SpawnedGpuKernel>` without waiting.

## Method `join`

1. `vkWaitForFences` on the job fence  
2. Compute → transfer barrier (TransferEngine)  
3. For each ref list: `download_container` → fill the kept `py::list` in place  
4. `release_inflight` (free CB/pool/set/fence/pack, drop `values_keep`)

Value scalars are not written back. Threadable/schema marshal is later.

## Testing

`register_smoke_saxpy` puts committed saxpy SPIR-V in ShaderCache (test-only).
Pytest drives product `launch_gpu_kernel` + `GpuJob.join` and asserts `y`:
`tests/unit/test_gpu_shader.py::test_live_launch_saxpy_product_path`.
