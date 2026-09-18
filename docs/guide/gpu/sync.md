# GPU sync (workgroup barriers)

Device-side rendezvous for invocations **inside one workgroup**. This is not the
CPU `Barrier(parties)` class, and it is not a whole-grid barrier.

# Contents

- [Why barriers exist](#why-barriers-exist)
- [API: two names, one lowering](#api-two-names-one-lowering)
- [Scope: workgroup only](#scope-workgroup-only)
- [CPU Barrier vs GPU barrier](#cpu-barrier-vs-gpu-barrier)
- [What you can do in 0.2.0](#what-you-can-do-in-020)
- [Shared memory (0.2.1)](#shared-memory-021)
- [Rejected forms](#rejected-forms)
- [Host-side phases instead of grid barriers](#host-side-phases-instead-of-grid-barriers)

# Why barriers exist

Invocations in a workgroup can run ahead of each other. When algorithm step B must
see results that step A wrote into **memory shared by that workgroup**, every
invocation in the group must reach a meeting point first. That meeting point is a
**barrier**.

On Vulkan/GLSL this is a shader builtin (`barrier` plus a shared-memory memory
barrier), not a host lock and not a spinloop you write by hand.

# API: two names, one lowering

```python
from cthreads.sync import Barrier, __sync_threads
from cthreads.gpu import Gpu, GlobalIdx, ThreadIdx

@Gpu
def example(n: int, data: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    # ... writes that peers in this workgroup must observe ...
    __sync_threads()
    # equivalent:
    # Barrier.arrive_and_wait()
    # ... reads of those writes ...
```

| Form | Audience | GPU effect |
|------|----------|------------|
| `__sync_threads()` | CUDA-familiar | Workgroup barrier + shared memory barrier |
| `Barrier.arrive_and_wait()` | Same idea, method-shaped name | Identical lowering |

Both are compile-only on the GPU path. Calling `__sync_threads()` from ordinary
Python raises `RuntimeError`.

# Scope: workgroup only

```text
Workgroup 0 invocations  <-->  barrier waits here only
Workgroup 1 invocations  <-->  separate barrier domain
```

A barrier does **not**:

- Wait for all workgroups in the launch
- Make Python see updated lists mid-run
- Replace a second `gpu()` launch when your algorithm needs a global phase

Default workgroup size is 64 along X. At most those peers participate in one
barrier instance.

# CPU Barrier vs GPU barrier

| | CPU `cthreads.sync.Barrier` | GPU forms above |
|--|----------------------------|-----------------|
| Construction | `Barrier(parties)` then instance `arrive_and_wait()` | No construction in `@Gpu` |
| Parties | Explicit count you choose | Implicit: workgroup size |
| Machine | OS threads / host mutex | GPU shader builtin |
| Import package | Same `cthreads.sync` | Same `cthreads.sync` |

Sharing the import surface is intentional. Sharing the implementation is not:
the compiler backend chooses the lowering.

# What you can do in 0.2.0

Barriers are available so cooperative patterns can be authored against a stable
API. **Without workgroup shared memory**, the practical uses inside a single
kernel are limited: there is no user-declared shared array to stage tiles into
yet. You can still:

- Call the barrier forms (they lower correctly).
- Structure multi-pass algorithms as **multiple `gpu()` launches** on the host
  (global phases).
- Use residency (`GpuArena`) so those phases do not thrash host memory.

# Shared memory (0.2.1)

Planned follow-up: declare workgroup-local shared storage and use barriers between
produce/consume steps inside the group (classic tiled reductions, stencils, and
so on). Until that lands, prefer host multi-pass for algorithms that need
cross-element staging beyond ordinary list buffers.

# Rejected forms

Inside `@Gpu`:

```python
Barrier(64)           # TypeError: construction not valid inside @Gpu
Barrier()             # same
b = Barrier(64)       # not a GPU pattern
b.arrive_and_wait()   # instance path is CPU-oriented
```

Use only:

```python
__sync_threads()
Barrier.arrive_and_wait()
```

# Host-side phases instead of grid barriers

Example: two global steps over the same arrays.

```python
from cthreads.gpu import Gpu, GlobalIdx, GpuArena, gpu

@Gpu
def step_a(n: int, x: list[float], y: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    y[i] = x[i] * 2.0

@Gpu
def step_b(n: int, y: list[float], z: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    z[i] = y[i] + 1.0

n = len(x)
with GpuArena() as arena:
    arena.bind(x=x, y=y, z=z)
    gpu(step_a, n, x, y).join(download=False)
    gpu(step_b, n, y, z).join(download=False)
    arena.sync("z")
```

Each `gpu()` launch completes before the next begins when you `join` in between.
That is the supported way to get **grid-wide** ordering in 0.2.0.

## See also

- [concepts.md](./concepts.md#workgroups-and-the-grid)
- [CPU sync guide](../sync.md)
- [best_practices.md](./best_practices.md)
