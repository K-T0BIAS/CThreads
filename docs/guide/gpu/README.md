# GPU guides (cthreads)

These guides cover the public Vulkan compute path: `@Gpu` kernels, `gpu()` launch, `GpuArena` residency, and workgroup barriers. They assume you know Python and are comfortable with the CPU side of cthreads (`@Thread`, jobs, join). They do not assume prior GPU or Vulkan experience.

Start here: [quickstart.md](./quickstart.md).

## Version note

| Release | GPU surface |
|---------|-------------|
| **0.2.0+** | `@Gpu` / `gpu()` / `GpuJob`, index builtins, list writeback, `GpuArena`, workgroup `__sync_threads` / `Barrier.arrive_and_wait()` |
| **Later** | Workgroup shared memory (tiles / cooperative reductions) |

Workgroup shared memory is not in the current public dialect. Barriers are documented so the API stays stable. Cooperative tile patterns become useful once shared memory lands. Packaging fix **0.2.1** (kernel headers in wheels) is separate from that GPU feature.

## Reading order

| Start here | Why |
|------------|-----|
| [quickstart.md](./quickstart.md) | Probe, first `saxpy`, arena sketch, tool map |
| [concepts.md](./concepts.md) | Host vs device, workgroups, writeback |
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
| `pip install cthreads-gpu` | Full package with GPU `_ext` (do not also install `cthreads`) |

See [install.md](../../install.md#gpu-vulkan-compute).

Always probe before launching:

```python
from cthreads import gpu

print(gpu.available())
if gpu.available():
    print(gpu.device_name())
```
