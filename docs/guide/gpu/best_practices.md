# GPU best practices

Practical guidance for correct, maintainable, and efficient use of `cthreads.gpu`
in 0.2.0. Pair with [concepts.md](./concepts.md) and the worked samples in
[examples.md](./examples.md).

# Contents

- [Choose CPU or GPU deliberately](#choose-cpu-or-gpu-deliberately)
- [Keep kernels regular](#keep-kernels-regular)
- [Always bounds-check](#always-bounds-check)
- [Prefer SoA lists](#prefer-soa-lists)
- [Name an explicit `n`](#name-an-explicit-n)
- [Minimize host round-trips](#minimize-host-round-trips)
- [Phase on the host for global ordering](#phase-on-the-host-for-global-ordering)
- [Treat barriers carefully](#treat-barriers-carefully)
- [Respect float32](#respect-float32)
- [Fail fast on availability](#fail-fast-on-availability)
- [Keep orchestration in Python](#keep-orchestration-in-python)
- [Testing habits](#testing-habits)
- [Performance expectations](#performance-expectations)

# Choose CPU or GPU deliberately

| Prefer `@Gpu` when | Prefer `@Thread` when |
|--------------------|------------------------|
| Large arrays, similar work per index | Irregular control flow / sparse work |
| Throughput matters more than latency of tiny n | Small n where launch overhead dominates |
| Outputs fit list writeback | You need return values or rich types |
| No mid-run Python observe | You need `__sync_state` / locks / TBuffer |

A slow GPU path is often a small problem size, accidental per-iteration download,
or an algorithm that needs atomics/shared memory that 0.2.0 does not provide yet.

# Keep kernels regular

Write kernels so most invocations do the same kind of work:

- Same loop trip counts when possible.
- Avoid large `if` trees that split the workgroup into many divergent paths.
- Push rare special cases to a separate pass or to the CPU.

Regular code maps cleanly to SIMD-style GPU execution.

# Always bounds-check

```python
i: int = GlobalIdx.x
if i >= n:
    return
```

Dispatch is rounded up to workgroup multiples. Skipping the guard is undefined for
your arrays' logical length.

# Prefer SoA lists

Use parallel arrays:

```python
px: list[float]
py: list[float]
vx: list[float]
vy: list[float]
```

rather than objects or nested structures. The GPU allowlist is scalar lists only.
SoA also keeps memory access patterns predictable.

# Name an explicit `n`

```python
def kernel(n: int, ..., out: list[float]) -> None:
```

An `int` parameter named `n` drives `group_count_x` inference. Keep `n` equal to
the logical length you intend to process, and keep list lengths consistent with it.

# Minimize host round-trips

Default launch uploads and downloads lists. For iterative algorithms:

1. `GpuArena.bind(...)` once.
2. Loop with `gpu(...).join(download=False)`.
3. `arena.sync()` when Python must read results.

Re-bind (or upload via bind) when the **host** changes list contents that the
device must see.

# Phase on the host for global ordering

If step B must see every index's result from step A across the whole problem:

```text
gpu(step_a).join(download=False)
gpu(step_b).join(download=False)
```

Do not expect `__sync_threads()` to provide that guarantee. Barriers are
workgroup-local ([sync.md](./sync.md)).

# Treat barriers carefully

- Use `__sync_threads()` or `Barrier.arrive_and_wait()` only inside `@Gpu`.
- Do not construct `Barrier(...)` in device code.
- In 0.2.0, barriers are forward-compatible API; shared-memory tile recipes arrive
  in 0.2.1. Prefer host multi-pass for real algorithms until then.

# Respect float32

GPU `float` is 32-bit. CPU `@Thread` `float` is double-backed. Mixing CPU and GPU
passes can accumulate different rounding. If you need higher precision end-to-end,
validate tolerances explicitly or keep sensitive reductions on CPU for now.

# Fail fast on availability

At process start:

```python
from cthreads import gpu

if not gpu.available():
    # choose CPU fallback or exit with a clear message
    ...
```

Decorating `@Gpu` when the path is unavailable raises. Probe first in applications
that must run on machines without Vulkan compute.

# Keep orchestration in Python

Good split of responsibility:

- **Python:** allocate lists, choose launches, arena lifetime, I/O, plotting, tests.
- **`@Gpu`:** arithmetic and indexed reads/writes over buffers.

Avoid trying to rebuild a full application framework inside the shader language
subset.

# Testing habits

1. Compare GPU outputs to a pure-Python or `@Thread` reference on modest `n`.
2. Include non-multiples of 64 for `n` to exercise bounds checks.
3. Test arena loops with a final `sync` and assert against a single-launch baseline.
4. Assert `gpu.available()` or skip in environments without a device (CI often has
   no GPU).

# Performance expectations

What usually helps:

- Larger `n` (hundreds of thousands to millions of elements).
- Resident buffers for multi-pass loops.
- Fewer, thicker kernels rather than tiny launches in a Python loop without arena.

What usually does not match CUDA driver-enqueue folklore:

- Expecting nanosecond host enqueue with full list marshal every call.
- Tiny `n` where compile/upload dominate.

Measure with your data sizes; microbenchmarks on `n=16` rarely predict production.

## Checklist

- [ ] `available()` checked
- [ ] `-> None` and allowed types only
- [ ] `GlobalIdx` + `i >= n` guard
- [ ] Explicit `n` when possible
- [ ] Arena for hot multi-pass loops
- [ ] Global phases via multiple launches, not workgroup barriers alone
- [ ] Tolerances aware of float32
