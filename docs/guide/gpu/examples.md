# GPU examples

Worked examples for `cthreads.gpu` 0.2.0. Each sample is self-contained. Adapt
sizes and names to your project. These are documentation samples, not a shipped
demo suite.

# Contents

- [Probe device](#probe-device)
- [Fill a buffer](#fill-a-buffer)
- [SAXPY](#saxpy)
- [Clamp / activation-style map](#clamp--activation-style-map)
- [Pairwise combine into an output list](#pairwise-combine-into-an-output-list)
- [Integer histogram bins (no atomics)](#integer-histogram-bins-no-atomics)
- [Stencil 1D (host multi-pass safe pattern)](#stencil-1d-host-multi-pass-safe-pattern)
- [Resident multi-pass loop](#resident-multi-pass-loop)
- [Two-phase pipeline with arena](#two-phase-pipeline-with-arena)
- [CPU reference check](#cpu-reference-check)
- [Graceful CPU fallback sketch](#graceful-cpu-fallback-sketch)
- [Workgroup barrier call shape](#workgroup-barrier-call-shape)

# Probe device

```python
from cthreads import gpu

def main() -> None:
    if not gpu.available():
        print("cthreads.gpu is not usable on this machine")
        return
    gpu.init()
    print("device:", gpu.device_name())

if __name__ == "__main__":
    main()
```

# Fill a buffer

```python
from cthreads.gpu import Gpu, GlobalIdx, gpu

@Gpu
def fill(n: int, value: float, out: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    out[i] = value

def main() -> None:
    n = 10_000
    out = [0.0] * n
    gpu(fill, n, 3.14, out).join()
    assert out[0] == 3.14
    assert out[n - 1] == 3.14

if __name__ == "__main__":
    main()
```

# SAXPY

```python
from cthreads.gpu import Gpu, GlobalIdx, gpu

@Gpu
def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    y[i] = a * x[i] + y[i]

def main() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [10.0, 20.0, 30.0, 40.0, 50.0]
    gpu(saxpy, len(x), 2.0, x, y).join()
    print(y)  # [12.0, 24.0, 36.0, 48.0, 60.0]

if __name__ == "__main__":
    main()
```

# Clamp / activation-style map

```python
from cthreads.gpu import Gpu, GlobalIdx, gpu

@Gpu
def clamp01(n: int, x: list[float], y: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    v: float = x[i]
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    y[i] = v

def main() -> None:
    x = [-1.0, 0.25, 2.0]
    y = [0.0, 0.0, 0.0]
    gpu(clamp01, len(x), x, y).join()
    print(y)  # [0.0, 0.25, 1.0]

if __name__ == "__main__":
    main()
```

# Pairwise combine into an output list

```python
from cthreads.gpu import Gpu, GlobalIdx, gpu

@Gpu
def hadamard(n: int, a: list[float], b: list[float], out: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    out[i] = a[i] * b[i]

def main() -> None:
    a = [1.0, 2.0, 3.0, 4.0]
    b = [2.0, 2.0, 2.0, 2.0]
    out = [0.0] * len(a)
    gpu(hadamard, len(a), a, b, out).join()
    print(out)  # [2.0, 4.0, 6.0, 8.0]

if __name__ == "__main__":
    main()
```

# Integer histogram bins (no atomics)

Without atomics, invocations must not contend on the same output slot. A safe
teaching pattern is **one output per input** (category id), then aggregate on the
host -- or give each invocation a private region. Example: map values to bin ids
on GPU, count on CPU.

```python
import math
from cthreads.gpu import Gpu, GlobalIdx, gpu

@Gpu
def value_to_bin(
    n: int,
    n_bins: int,
    lo: float,
    inv_width: float,
    x: list[float],
    bins: list[int],
) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    # floor((x - lo) * inv_width), clamped into [0, n_bins-1]
    t: float = (x[i] - lo) * inv_width
    b: int = int(math.floor(t))
    if b < 0:
        b = 0
    if b >= n_bins:
        b = n_bins - 1
    bins[i] = b

def main() -> None:
    x = [0.1, 0.4, 0.9, 1.2]
    n_bins = 4
    lo = 0.0
    hi = 1.0
    inv_width = float(n_bins) / (hi - lo)
    bin_of = [0] * len(x)
    gpu(value_to_bin, len(x), n_bins, lo, inv_width, x, bin_of).join()

    counts = [0] * n_bins
    for b in bin_of:
        counts[b] += 1
    print(bin_of, counts)

if __name__ == "__main__":
    main()
```

When device atomics land in a later release, in-kernel histogram accumulation
becomes more direct. Until then, keep contested reductions on the host or design
conflict-free writes.

# Stencil 1D (host multi-pass safe pattern)

Neighbor reads from a **read-only** input list into a separate output list avoid
hazards inside one launch:

```python
from cthreads.gpu import Gpu, GlobalIdx, gpu

@Gpu
def smooth(n: int, src: list[float], dst: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    left: float = src[i]
    right: float = src[i]
    if i > 0:
        left = src[i - 1]
    if i + 1 < n:
        right = src[i + 1]
    dst[i] = 0.25 * left + 0.5 * src[i] + 0.25 * right

def main() -> None:
    src = [0.0, 1.0, 0.0, 1.0, 0.0]
    dst = [0.0] * len(src)
    gpu(smooth, len(src), src, dst).join()
    print(dst)

if __name__ == "__main__":
    main()
```

For iterative smoothing, swap roles across launches (or use an arena and two
buffers) rather than reading and writing the same list unsafely in one pass.

# Resident multi-pass loop

```python
from cthreads.gpu import Gpu, GlobalIdx, GpuArena, gpu

@Gpu
def damp(n: int, factor: float, y: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    y[i] = y[i] * factor

def main() -> None:
    y = [1.0] * 100_000
    n = len(y)
    with GpuArena() as arena:
        arena.bind(y=y)
        for _ in range(50):
            gpu(damp, n, 0.99, y).join(download=False)
        arena.sync("y")
    print(y[0])

if __name__ == "__main__":
    main()
```

# Two-phase pipeline with arena

```python
from cthreads.gpu import Gpu, GlobalIdx, GpuArena, gpu

@Gpu
def square(n: int, x: list[float], y: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    y[i] = x[i] * x[i]

@Gpu
def add_const(n: int, c: float, y: list[float], z: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    z[i] = y[i] + c

def main() -> None:
    x = [1.0, 2.0, 3.0, 4.0]
    y = [0.0] * len(x)
    z = [0.0] * len(x)
    n = len(x)
    with GpuArena() as arena:
        arena.bind(x=x, y=y, z=z)
        gpu(square, n, x, y).join(download=False)
        gpu(add_const, n, 1.0, y, z).join(download=False)
        arena.sync("z")
    print(z)  # [2.0, 5.0, 10.0, 17.0]

if __name__ == "__main__":
    main()
```

# CPU reference check

```python
from cthreads.gpu import Gpu, GlobalIdx, gpu

@Gpu
def scale(n: int, a: float, x: list[float], y: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    y[i] = a * x[i]

def ref_scale(a: float, x: list[float]) -> list[float]:
    return [a * v for v in x]

def main() -> None:
    x = [float(i) for i in range(1000)]
    y = [0.0] * len(x)
    a = 1.5
    gpu(scale, len(x), a, x, y).join()
    expect = ref_scale(a, x)
    for i in range(len(x)):
        err = abs(y[i] - expect[i])
        if err > 1e-5:
            raise AssertionError(f"mismatch at {i}: {y[i]} vs {expect[i]}")
    print("ok")

if __name__ == "__main__":
    main()
```

# Graceful CPU fallback sketch

```python
from cthreads import gpu as gpu_api
from cthreads.gpu import Gpu, GlobalIdx, gpu

@Gpu
def scale_gpu(n: int, a: float, x: list[float], y: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    y[i] = a * x[i]

def scale_cpu(a: float, x: list[float], y: list[float]) -> None:
    for i in range(len(x)):
        y[i] = a * x[i]

def scale(a: float, x: list[float], y: list[float]) -> None:
    if gpu_api.available():
        gpu(scale_gpu, len(x), a, x, y).join()
    else:
        scale_cpu(a, x, y)
```

Note: defining `@Gpu` requires availability at decorate time. For binaries that
must import on GPU-less hosts, gate the decorated definitions behind
`if gpu_api.available():` or keep GPU kernels in a module imported only after the
probe succeeds.

# Workgroup barrier call shape

Illustrates the legal call forms. Without shared memory, this is primarily an API
shape example; see [sync.md](./sync.md).

```python
from cthreads.sync import Barrier, __sync_threads
from cthreads.gpu import Gpu, GlobalIdx

@Gpu
def barrier_shape(n: int, data: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    data[i] = data[i] + 1.0
    __sync_threads()
    # Same device sync:
    Barrier.arrive_and_wait()
    data[i] = data[i] * 1.0
```

## See also

- [best_practices.md](./best_practices.md)
- [arena.md](./arena.md)
- [sync.md](./sync.md)
