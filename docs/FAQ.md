# FAQ

Common questions and pitfalls for **cthreads**. Answers use full sentences. Code samples include comments. For install detail see [install.md](./install.md). For multithreading basics see [concepts.md](./concepts.md).

#### Technical terms (project-specific)

- **GIL**: Global Interpreter Lock. A process-wide lock in CPython so only one thread runs Python bytecode at a time.
- **kernel**: A `@Thread` or `@Gpu` function compiled to native code (C++ or Vulkan SPIR-V).
- **pack**: The C++ copy of job arguments. The kernel mutates the pack. Python objects update on writeback.
- **writeback**: Copying mutable pack fields back into the Python objects you passed in (on `join` or sync).
- **SharedHost**: Per-pool (or default) store for `Shared[T]` values so many jobs see one native object.
- **SPIR-V**: Shader Intermediate Representation. What Vulkan compute pipelines consume after GLSL lowering.

---

## What is cthreads

### What does this library do

**cthreads** compiles a typed subset of Python into C++ so work can run on real operating-system threads without holding the GIL. You keep a Python-shaped control flow with jobs, pools, and sync tools. An optional second package adds Vulkan GPU kernels with `@Gpu`.

It is not a drop-in replacement for all Python. It is not a full NumPy or Numba substitute. It is aimed at people who want threaded (and optionally GPU) kernels while staying close to ordinary Python call style.

### What should I install

For CPU kernels use:

```bash
pip install cthreads
```

For CPU plus Vulkan `@Gpu` support use the other project name instead:

```bash
pip install cthreads-gpu
```

Do not install both. Both ship the import name `cthreads` and the native `_ext` module. Installing both overwrites `_ext` and leads to confusing failures.

More detail: [install.md](./install.md).

---

## Install and first run

### Why do I still need a C++ compiler after pip install

PyPI **wheels** ship a prebuilt `_ext` on supported platforms. That covers the runtime bridge.

Your `@Thread` bodies are compiled later on the machine that runs them. The first `prepare` or `cthreads.thread(...)` (or an explicit compile for pools) needs a C++17 compiler such as MSVC Build Tools, `g++`, or `clang++`.

Google Colab needs `g++` installed in the notebook (the Mandelbrot demo does this). Local Linux often uses `build-essential`. Windows usually uses Visual Studio Build Tools with the C++ workload.

### Does every platform have a wheel

Published wheels target Linux and Windows **x86_64** for common CPython versions. Other platforms may fall back to an sdist build. That path needs CMake and a compiler up front. macOS and some ARM builds are not the first-class wheel story yet.

### Why is the first launch slow

The first successful kernel build pays for codegen and linking a shared library (for example `cthreads_kernels.dll` or `.so`). Later launches reuse the cache when the annotated source is **unchanged**.

Warm up once before you time anything (or simply count the time of the second run):

```python
import cthreads
from cthreads import Thread

@Thread
def add(a: int, b: int) -> int:
    # Locals and parameters need types in @Thread bodies.
    s: int = a + b
    return s

# First call may compile. Do not put this in a benchmark timer.
cthreads.thread(add, 1, 2).join()

# Steady-state timing starts here.
job = cthreads.thread(add, 3, 4)
job.join()
print(job.result())  # 7
```

### ThreadPool failed with missing `__kernel_meta__`

`cthreads.thread(...)` can auto-run prepare and load kernels when nothing is loaded yet. **`ThreadPool.submit` / `group` do not.** Compile and load before you submit pool jobs.

```python
from cthreads import Thread, ThreadPool, prepare, load_kernels

@Thread
def add(a: int, b: int) -> int:
    return a + b

# Build kernels then load the shared library into the process.
binary = prepare()
load_kernels(str(binary))

pool = ThreadPool(4).start()
try:
    job = pool.submit(add, 1, 2)
    print(job.result())  # 3
finally:
    pool.stop()
```

See [pools.md](./guide/pools.md).

---

## Performance pitfalls

### Why was my `@Thread` kernel slower than pure Python

A common cause is marshaling a very large `list` of scalars on every launch. At spawn time each element is packed into C++ through per-element accessors. On `join` each element is written back the same way. For hundreds of thousands of ints that cost can dominate the useful work.

Pure Python can mutate a list in place with no pack or writeback. A single `@Thread` that only rewrites a huge `list[int]` can lose that comparison even though the math inside C++ is faster.

Prefer one of these patterns for large buffers:

- `TBufferI64` (or other TBuffer forms) so workers share a native buffer by handle
- `Shared[list[T]]` for cooperative state that many jobs must see (keep the list small if you pass it often)
- Many strip jobs on a `ThreadPool` with disjoint index ranges

The Colab Mandelbrot demo uses pool plus Shared plus TBuffer on purpose. Open it from the README badge or [Mandelbrot.ipynb](../colab/Mandelbrot.ipynb).

### Will Python `threading` speed up my CPU-bound Mandelbrot

Usually no. CPU-bound pure Python still takes turns on the GIL. Extra threads add scheduling cost without using multiple cores for bytecode. That is why the Colab baseline stays single-threaded on the Python side.

**cthreads** kernels run as compiled C++ off the GIL so they can overlap on multiple cores when you split work across a pool.

### Is cthreads always faster than NumPy

No. NumPy already runs much work in optimized native code. **cthreads** helps when you want Python-shaped threaded kernels, custom loops, jobs and pools, or the Vulkan `@Gpu` path. It is not a claim that every array expression beats NumPy.

---

## Language and types

### What types are allowed inside `@Thread`

Parameters, returns, and locals need annotations from the whitelist. Typical allowed forms include `int`, `float`, `bool`, `str`, nested `list` and `dict` of allowed types, `@Threadable` classes, and internal types from `cthreads` (sync, TBuffer helpers, linalg where documented).

Arbitrary Python objects are not supported in kernels. `*args` and `**kwargs` are not supported on `@Thread` functions.

See [thread_and_threadable.md](./guide/thread_and_threadable.md) and [concepts.md](./concepts.md).

### Why do my Python lists look unchanged while the job runs

The kernel edits the **pack** (the C++ copy). By default the live Python objects update on `join` writeback (or when you sync). Mid-run visibility needs `__sync_state()` / `job.sync_state()`, `Shared[T]`, or a **TBuffer** generation path.

See [sync.md](./guide/sync.md) and [concepts.md](./concepts.md).

---

## Shared memory and pools

### How do several jobs share one list

A plain `list` argument is packed per job. Each job gets its own pack copy unless you use cooperative sharing.

Use `Shared[list[T]]` (or other `Shared[T]`) so values live on a **SharedHost**. On a pool use `pool.group(...)` or `with pool.submit_queue():` so the host stays pinned for the whole submit wave. A bare loop of `pool.submit` with Shared data can free the host too early.

```python
from cthreads import Thread, Shared, ThreadPool, prepare, load_kernels

@Thread
def bump_at(head: Shared[list[int]], i: int) -> None:
    # All jobs in this wave see the same native list on the pool SharedHost.
    head[i] = head[i] + 1

binary = prepare()
load_kernels(str(binary))

head: list[int] = [0, 0, 0, 0]
pool = ThreadPool(4).start()
try:
    # group pins Shared for the batch. Prefer this over bare submit loops.
    group = pool.group(bump_at, [(head, i) for i in range(4)])
    group.results()
    print(head)  # [1, 1, 1, 1] after writeback
finally:
    pool.stop()
```

See [pools.md](./guide/pools.md) and [marshal_and_module.md](./guide/marshal_and_module.md).

### When should I use TBuffer

Use a **TBuffer** when you want a fixed-capacity native buffer and publish snapshots without writing every element back through Python list marshal. Workers can write disjoint indices. Call `publish()` after the wave finishes if multiple threads shared one write slot. Then read with `read_copy()` (or generation polling) on the host.

See [sync.md](./guide/sync.md).

---

## GPU

### I installed cthreads but `@Gpu` fails

The default **cthreads** wheel is built with GPU support off. Install **cthreads-gpu** instead (and uninstall plain cthreads). Confirm with:

```python
from cthreads import gpu

# Soft probe. False means no usable GPU path in this install or environment.
print(gpu.available())
if gpu.available():
    print(gpu.device_name())
```

You still need a Vulkan-capable device and working drivers or ICD on the machine.

### Does `@Gpu` support everything `@Thread` supports

No. The public GPU path focuses on void kernels with scalars and lists of scalars, launch then `join`, and related tools such as `GpuArena`. Workgroup barriers exist. Workgroup shared memory and several richer dialects are planned later. Read [guide/gpu/README.md](./guide/gpu/README.md) before you design a large GPU port.

---

## Platform and versions

### Which Python versions are supported

**Python 3.10+** per packaging metadata. Use a version that has a matching wheel when possible.

### What about the old missing `shared_host.hpp` error after pip install

Some **0.2.0** wheels shipped `_ext` but not the kernel C++ headers. JIT compile then failed with a missing `shared_host.hpp`. That packaging bug is fixed in **0.2.1 and later**. Upgrade:

```bash
pip install -U "cthreads>=0.2.1"
```

If you still see it, confirm you are not on a yanked or cached old wheel.

---

## Project status and where to ask

### Is this production ready

The project is early (alpha-style). The CPU demo path (pool, Shared, TBuffer) is the best-tested showcase. APIs can still change. Use judgment before you depend on it in critical production systems.

Maintenance is primarily one maintainer (If you want to help contact [@K-T0BIAS](https://github.com/K-T0BIAS)). Response times vary. There may be quiet periods. Please check this FAQ and the Colab demo before opening an issue. 

### Where should I ask a question

- **Bug with a clear repro**: open a GitHub Issue.
- **How do I… / design question**: prefer GitHub Discussions (Q and A) 
- **Feature idea**: Discussions (Ideas) or an Issue labeled as a request.
- **Unsure?**: ask me [@K-T0BIAS](https://github.com/K-T0BIAS) (I will try to respond asap. Please be patient with my response times) (prefer the GitHub Discussions (Q&A))

### How do I try the CPU demo quickly

Use the Colab notebook linked from the README (Open in Colab badge). It installs a recent **cthreads**, installs `g++`, and runs the Mandelbrot comparison with pool plus Shared plus TBuffer.

---

## See also

| Topic | Doc |
|-------|-----|
| Install and toolchain | [install.md](./install.md) |
| GIL and pack model | [concepts.md](./concepts.md) |
| Pools and Shared waves | [guide/pools.md](./guide/pools.md) |
| Sync and TBuffer | [guide/sync.md](./guide/sync.md) |
| GPU user guides | [guide/gpu/quickstart.md](./guide/gpu/quickstart.md) |
| Guides catalog | [index.md](./index.md) |
| CPU quickstart | [quickstart.md](./quickstart.md) |
