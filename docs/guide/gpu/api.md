# GPU API reference (0.2.0)

Compact reference for the public `cthreads.gpu` surface and related sync entry
points. Narrative guides: [README.md](./README.md).

# Contents

- [Package imports](#package-imports)
- [`@Gpu`](#gpu)
- [Launch](#launch)
- [Indexes](#indexes)
- [GpuArena](#gpuarena)
- [Probe / lifecycle](#probe--lifecycle)
- [Errors](#errors)
- [Sync barriers](#sync-barriers)
- [Kernel language (summary)](#kernel-language-summary)

# Package imports

```python
from cthreads.gpu import (
    Gpu,
    gpu,
    prepare,
    compile,
    GpuJob,
    GpuArena,
    GlobalIdx,
    ThreadIdx,
    BlockIdx,
    BlockDim,
    GridDim,
    available,
    device_name,
    init,
    shutdown,
)

from cthreads.sync import Barrier, __sync_threads
```

# `@Gpu`

```python
@Gpu
def kernel(...) -> None: ...

@Gpu(log=True)
def kernel_logged(...) -> None: ...
```

- Validates GPU type allowlist and registers the function.
- Requires GPU availability at decorate time.
- Does not launch.

Allowed parameter types: `int`, `float`, `bool`, `list[int]`, `list[float]`,
`list[bool]`. Return must be `None`.

# Launch

### `prepare(force: bool = False) -> dict`

Ensure Vulkan is usable and compile registered `@Gpu` kernels.

### `compile(force: bool = False) -> dict`

Emit / refresh SPIR-V artifacts for registered kernels.

### `gpu(fn, *args, force: bool = False) -> GpuJob`

Launch a `@Gpu` function. Positional args only. Auto-prepares when needed.

### `GpuJob`

| Method | Signature | Notes |
|--------|-----------|-------|
| `join` | `join(download: bool = True) -> None` | Wait; download ref lists when `download` is true |
| `result` | `result() -> None` | Always `None` |

# Indexes

Markers with `.x` / `.y` / `.z`:

| Name | Meaning |
|------|---------|
| `GlobalIdx` | Global invocation id (prefer for 1D maps) |
| `ThreadIdx` | Id within workgroup |
| `BlockIdx` | Workgroup id |
| `BlockDim` | Workgroup size |
| `GridDim` | Number of workgroups |

Default workgroup size X: **64**. `group_count_x` defaults to `ceil(n / 64)` with
`n` from parameter `n` or longest list length.

# GpuArena

```python
with GpuArena() as arena:
    arena.bind(x=x, y=y)
    gpu(fn, n, x, y).join(download=False)
    arena.sync()           # or arena.sync("y")
# arena.release() on exit
```

| Method | Role |
|--------|------|
| `bind(**lists)` | Upload / register lists by keyword slot name |
| `sync(*names)` | Download slots into Python lists |
| `release()` | Destroy buffers; clear registrations |
| `names()` | Bound slot names |

Residency is keyed by list object identity + length.

# Probe / lifecycle

| Function | Returns | Raises |
|----------|---------|--------|
| `available()` | `bool` | No (soft fail) |
| `device_name()` | `str` | Mapped GPU errors / not built |
| `init()` | `None` | Mapped GPU errors / not built |
| `shutdown()` | `None` | Soft if not built |

# Errors

`CThreadsGPUError` and subclasses:

- `VulkanNotBuiltError`
- `VulkanLoaderNotFound`
- `VulkanNoDevice`
- `VulkanInitFailed`
- `VulkanOutOfMemory`
- `GpuInvalidArgument`
- `GpuUseAfterDestroy`
- `GPUNotAvailable`

Details: [errors.md](./errors.md).

# Sync barriers

Inside `@Gpu` only:

```python
__sync_threads()
Barrier.arrive_and_wait()
```

Workgroup scope. `Barrier(...)` construction inside `@Gpu` raises `TypeError`.
Guide: [sync.md](./sync.md).

# Kernel language (summary)

Supported: annotated locals, `if`/`else`, `while`, `for i in range(...)`, early
`return`, list indexing, arithmetic/comparisons, `sqrt` / `floor` / `int(...)`,
barrier calls.

Not supported: list `for-in`, rich Python types, valued returns, keyword
`gpu(...)` args, mid-run Python observe.

Full rules: [kernels.md](./kernels.md).
