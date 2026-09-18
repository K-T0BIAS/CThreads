# GPU quickstart

End-to-end path from install probe to a working element-wise kernel. For the mental
model behind the steps, see [concepts.md](./concepts.md).

# Contents

- [1. Confirm GPU availability](#1-confirm-gpu-availability)
- [2. Write a `@Gpu` kernel](#2-write-a-gpu-kernel)
- [3. Launch and join](#3-launch-and-join)
- [4. Read results](#4-read-results)
- [5. Optional: keep data on device](#5-optional-keep-data-on-device)
- [Common first mistakes](#common-first-mistakes)

# 1. Confirm GPU availability

```python
from cthreads import gpu

print(gpu.available())
if gpu.available():
    print(gpu.device_name())
```

If `available()` is `False`, stop here and check
[install.md](../../install.md#gpu-vulkan-compute) and [errors.md](./errors.md).
Decorating `@Gpu` or calling `gpu()` raises `GPUNotAvailable` when the path is not
usable.

# 2. Write a `@Gpu` kernel

Rules that matter for the first kernel:

- Annotate every parameter and use `-> None`.
- Use only `int`, `float`, `bool`, and `list` of those scalars.
- Introduce locals with annotated assignment (`i: int = ...`).
- Index with `GlobalIdx.x` for 1D element-wise work.
- Guard out-of-range invocations.

```python
from cthreads.gpu import Gpu, GlobalIdx, gpu

@Gpu
def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
    """
    y[i] = a * x[i] + y[i] for i in 0 .. n-1
    """
    i: int = GlobalIdx.x
    if i >= n:
        return
    y[i] = a * x[i] + y[i]
```

Calling `saxpy(...)` as ordinary Python still runs the Python body. Device
execution happens only through `gpu(saxpy, ...)`.

# 3. Launch and join

```python
x: list[float] = [1.0, 2.0, 3.0, 4.0]
y: list[float] = [10.0, 20.0, 30.0, 40.0]
n: int = len(x)
a: float = 2.0

job = gpu(saxpy, n, a, x, y)
job.join()
```

What `gpu()` does on first use in the process:

1. Ensures Vulkan is usable.
2. Compiles registered `@Gpu` functions to SPIR-V (cached afterward).
3. Uploads arguments and records a compute dispatch.
4. Returns a `GpuJob` that is already started.

`join()` waits for the GPU fence and, by default, downloads ref lists into the
Python lists you passed.

# 4. Read results

```python
print(y)  # [12.0, 24.0, 36.0, 48.0]
```

There is no scalar return value: `job.result()` is always `None`. Outputs live in
the list arguments you marked for writeback (all lists today).

# 5. Optional: keep data on device

When the same lists are reused for many launches, bind them once:

```python
from cthreads.gpu import GpuArena

with GpuArena() as arena:
    arena.bind(x=x, y=y)
    for _ in range(100):
        gpu(saxpy, n, a, x, y).join(download=False)
    arena.sync()  # download into x and y when you need Python to see values
```

Details: [arena.md](./arena.md).

# Common first mistakes

| Mistake | What happens | Fix |
|---------|--------------|-----|
| Missing `if i >= n: return` | Out-of-range writes / undefined behavior on padded invocations | Always bounds-check |
| Expecting `job.result()` to hold an array | Always `None` | Read the list args after join |
| Using `dict` / `@Threadable` / `str` args | `TypeError` at decorate or compile | Stick to scalars and scalar lists |
| Calling `__sync_threads()` from plain Python | `RuntimeError` | Only inside `@Gpu` bodies |
| Assuming a barrier syncs the whole array | Only one workgroup waits | Use multiple launches for global phases |

## Next

- [kernels.md](./kernels.md) - full language and type rules
- [indexes.md](./indexes.md) - how `GlobalIdx` relates to workgroups
- [examples.md](./examples.md) - more complete samples
