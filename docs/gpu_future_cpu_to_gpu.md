# Future work: CPU kernels calling GPU

**Status:** Planned after the public Python GPU package (landed in **0.2.0**:
`@Gpu`, `gpu()` / `GpuJob`, marshal, launch, join, writeback, `GpuArena`,
workgroup barriers). Shared memory is tracked separately for **0.2.1**.

This note records the intended design so we do not bolt on a second launch stack later. Write it as if the GPU runtime and Python `gpu()` path already exist; this feature only adds a native caller on top.

---

## Idea

GPU work is started from **Python** today (`gpu(fn, *args) -> GpuJob`).

A **CPU** `@Thread` kernel (native C++ in `kernels.dll`) should be able to launch the same GPU work and wait on it, without going back through Python for every dispatch.

Example intent (shape, not final API):

```text
@Thread
def step(...):
    # CPU work
    gpu_job = launch_gpu(some_gpu_fn, ...)   # C++ handle
    # more CPU work overlapping GPU
    gpu_job.join()
```

Same GpuPack / descriptor / pipeline / writeback machinery as `gpu()`. Only the **caller** changes: Python host vs native CPU kernel.

---

## Why this is a separate step

1. Python `gpu()` is the place that proves the permanent launch types and memory model.
2. Native CPU->GPU needs a stable **C++ `GpuJob` handle** and a native launch entry that mirrors Python marshal.
3. Overlapping CPU threads calling GPU will stress **transfer** and **queue submit** concurrency. That is easier to size once the single-caller path is solid.

---

## Assumptions (GPU system already done)

Before this work, the GPU package already provides:

- `Context`, TransferEngine (or an engine pool API), `memory::`, `GpuPack`
- Descriptor and pipeline path, `GpuJob.join`, writeback into caller-owned buffers / objects
- Kernel cache keyed by symbol (layout, pipeline, SPIR-V)
- Inflight state owned by each job (pack, descriptor set, fence)
- Python `gpu()` as a thin wrapper over that C++ surface

---

## What this needs

### A. C++ job handle (GPU side)

This may already exist for Python bindings. Confirm it is first-class native, not Python-only.

- Type like `cthreads::gpu::GpuJob`.
- API roughly: create/launch from packed args or from a native marshal helper; `join()`; optional `result()`; destroy / RAII.
- Safe to store and `join` from a CPU `@Thread` worker (OS thread; no GIL required for the wait itself).
- Overlapping jobs: two launches of the same GPU symbol get distinct inflight rows (symbol + inflight index), shared **kernel cache** entry.

### B. Native launch entry (GPU side)

Something CPU code can call without pybind, for example:

```text
GpuJob launch(symbol_or_pipeline_key, GpuPack&& pack, /* writeback plan */)
```

or a small C ABI used by generated CPU kernels (same spirit as `Fn__call`).

Marshal from C++ values into `GpuPack` (lists/scalars already in native memory). Do not go through `cthreads.marshal` / ctypes.

### C. Transfer concurrency (GPU side)

CPU threads will upload/download without Python sequencing.

- Prefer a **pool of TransferEngines** (or a checkout API) on `Context`, not one global engine held for the whole job lifetime without a plan.
- Mutex around a single engine is a minimum; a pool is the better fit once CPU->GPU is real.
- Compute queue submit may need a mutex if multiple host threads submit at once.

### D. Changes to the cthreads CPU / codegen system

- Allow CPU kernels to **call into** the GPU runtime (link against `_ext` GPU symbols or a shared `cthreads_gpu` API exported for kernels).
- Codegen / validator: a controlled way to express "launch this `@Gpu` from `@Thread`" (name binding, arity, types). Reject anything that implies mid-run Python sync on the GPU job.
- Job lifetime: CPU `SpawnedKernel` may own or await a child `GpuJob`. Define destroy order (GPU join before CPU pack free if they share buffers). Prefer **separate** packs unless explicitly aliased.
- Pools: a CPU pool worker launching GPU must not assume the GIL; completion is fence-based like Python `join`.
- Optional: register GPU child jobs for debugging / `Job` trees. Not required for the first version of this feature.

### E. What not to invent

- A second descriptor/pipeline stack for "native only."
- A public `DeviceBuffer` type.
- GPU `__sync_state` / mid-run Python observe.
- Replacing Python `gpu()`. It stays the host entry; native launch is an additional caller.

---

## Suggested sequencing

1. Confirm `GpuJob` is a complete C++ type used by Python bindings.
2. Add TransferEngine pool / checkout if not already there.
3. Add native `launch` + C++ marshal-into-GpuPack for supported types.
4. Extend CPU codegen/validator for an approved call shape.
5. Tests: CPU `@Thread` launches `@Gpu`, overlaps, joins; two CPU workers launch GPU concurrently.
6. Docs: call rules, lifetime, no GIL assumptions.

---

## Relation to other later work

This is an **additive** product feature on top of the finished Python GPU path. It is not a rewrite of pack, descriptors, pipelines, or launch. Track it separately from atomics, MoltenVK, and shared IR (those stay in the Later list in `todo.md`).
