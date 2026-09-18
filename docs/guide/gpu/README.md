# GPU guides (cthreads 0.2.0)

These guides cover the **public** Vulkan compute path: `@Gpu` kernels, `gpu()` launch,
`GpuArena` residency, and workgroup barriers. They assume you know Python and are
comfortable with the CPU side of cthreads (`@Thread`, jobs, join). They do **not**
assume prior GPU or Vulkan experience.

Internals for contributors (Vulkan substrate, C++ modules) live under
[docs/vk_guide](../../vk_guide/README.md) and [docs/internals/gpu](../../internals/gpu/README.md).

## Version note

| Release | GPU surface |
|---------|-------------|
| **0.2.0** | `@Gpu` / `gpu()` / `GpuJob`, index builtins, list writeback, `GpuArena`, workgroup `__sync_threads` / `Barrier.arrive_and_wait()` |
| **0.2.1 (planned)** | Workgroup shared memory (tiles / cooperative reductions) |

Shared memory is intentionally out of scope for 0.2.0. Barriers are documented so the
API is stable; cooperative tile patterns become useful once shared memory lands.

## Reading order

| Start here | Why |
|------------|-----|
| [concepts.md](./concepts.md) | Host vs device, workgroups, writeback, what a GPU kernel is |
| [quickstart.md](./quickstart.md) | First end-to-end `saxpy` |
| [kernels.md](./kernels.md) | `@Gpu` rules, types, language subset |
| [indexes.md](./indexes.md) | `GlobalIdx` / `ThreadIdx` / dispatch sizing |
| [launch.md](./launch.md) | `prepare`, `gpu()`, `GpuJob.join` |
| [arena.md](./arena.md) | Keep lists on device across launches |
| [sync.md](./sync.md) | Workgroup barriers (and what they are not) |
| [best_practices.md](./best_practices.md) | Patterns that stay correct and fast |
| [examples.md](./examples.md) | Worked examples |
| [errors.md](./errors.md) | Probe API, exceptions, troubleshooting |
| [api.md](./api.md) | Compact API reference |

## Install and probe

| Command | Result |
|---------|--------|
| `pip install cthreads` | CPU wheel |
| `pip install "cthreads[gpu]"` | CPU package + **`cthreads-gpu`** (GPU `_ext`) |
| `pip install cthreads-gpu` | GPU wheel only |

See [install.md](../../install.md#gpu-vulkan-compute) and [release.md](../../release.md).

Always probe before launching:

```python
from cthreads import gpu

if not gpu.available():
    raise SystemExit("No usable Vulkan compute device for cthreads.gpu")
print(gpu.device_name())
```

## Product rules (locked)

1. **Python types only** on the default path: scalars and `list[...]` of scalars. No public `DeviceBuffer` type for ordinary kernels.
2. **Launch then join.** There is no mid-run Python observe of GPU state (unlike CPU `__sync_state`).
3. **Results are list writeback.** Kernels are `-> None`. Scalars are inputs only.
4. **Workgroup barriers are workgroup-local.** They do not synchronize the whole launch grid.

CPU `@Thread` and GPU `@Gpu` are separate backends. The same process can use both;
a CPU kernel cannot yet launch a GPU kernel (see [gpu_future_cpu_to_gpu.md](../../gpu_future_cpu_to_gpu.md)).
