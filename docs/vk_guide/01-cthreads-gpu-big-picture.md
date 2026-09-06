# 01 — cthreads GPU big picture

## The CPU path

A typical typed kernel looks like:

```python
@Thread
def add(n: int, x: list[float], y: list[float]) -> None:
    i: int = 0
    while i < n:
        y[i] = x[i] + y[i]
        i = i + 1

job = thread(add, 4, [1,2,3,4], [10,10,10,10]).start()
job.join()
```

Roughly:

1. **Marshal** copies Python values into a C++ **pack** (args struct).
2. A worker thread runs the compiled C++ body on that pack.
3. **Writeback** copies mutated pack fields into the **same** Python lists.

Scalars live as fields in the pack. Lists live as native containers owned by / pointed from the pack.

## The GPU path (same story, different storage)

Same Python. Different backend:

1. **GPU marshal** creates a **GpuPack**:
   - one **device-local** buffer holding all scalars as a `std430` struct
   - one **device-local** buffer per list (tight array of floats/ints)
2. Host bytes go into those buffers via **staging + GPU copy** (upload).
3. A **compute shader** reads/writes those buffers through **descriptor bindings**.
4. After the GPU finishes (`join`): **download** + writeback into the same Python objects.

```text
                 CPU world                         GPU world
            -----------------                 --------------------
Python list[float]  --upload-->  staging  --copy-->  device-local SSBO
Python int/float    --upload-->  staging  --copy-->  scalar SSBO (struct)
                                                      |
                                                 compute shader
                                                      |
Python list[float]  <--writeback-- staging <--copy-- device-local SSBO
```

## Why not one giant buffer for everything?

cthreads uses **option 5**:

| Piece | Where it lives |
|-------|----------------|
| Scalars (`n`, `a`, …) | One small SSBO interpreted as a C-like struct |
| Each `list[...]` | Its own SSBO |

Reasons:

- Matches how compute shaders usually declare `buffer` blocks (one binding per array).
- Easy to reuse the same list buffer across multiple dispatches in a batch.
- Layout for arrays stays simple (tight packed floats).
- Scalar writeback is one download of a small blob.

The design **rejects** putting GPU pointers inside the scalar struct (that needs buffer device address). Bindings connect the shader to buffers instead.

## Job contract (important product rule)

On CPU, `__sync_state` can mirror pack -> Python mid-run.

On the GPU path: **only launch then wait**.

- No observing Python lists while the shader runs.
- Results appear after `GpuJob.join()` (fence wait + download + writeback).

That matches how GPUs want to work: record work, submit, wait once.

## Layers of the GPU stack

| Layer | Responsibility |
|-------|----------------|
| Context | Talk to Vulkan: instance, device, queue, entry points |
| Memory helpers | Allocate buffers, staging upload/download |
| GpuPack | Option 5: scalar SSBO + per-list SSBOs; marshal/writeback |
| Launch path | Descriptors, pipeline, dispatch, `GpuJob.join` |
| `@Gpu` / `gpu()` | Compile and run user kernels (same typing model as CPU) |
| Workloads / packaging | Real numeric steps, docs, CI, capability gates |

## Mental link: pack vs GpuPack

| CPU pack | GpuPack |
|----------|---------|
| One C++ struct in process memory | Several `VkBuffer`s on the device |
| Field `int n` | Bytes at offset 0 in scalar SSBO |
| Field `vector<float> x` | Separate SSBO whose bytes are the floats |
| Kernel gets C++ references | Shader gets bindings 0, 1, 2, … |

Same **roles**. Different **implementation**.

## What "done" looks like for a library user

```python
from cthreads import Gpu, gpu

@Gpu
def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
    ...

x = [1.0, 2.0, 3.0, 4.0]
y = [10.0, 10.0, 10.0, 10.0]
gpu(saxpy, 4, 2.0, x, y).join()
# y is updated in place, like a CPU Thread job
```

No `DeviceBuffer` in that story. Vulkan is an implementation detail of `_ext`.
