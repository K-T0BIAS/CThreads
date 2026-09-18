# Indexes and dispatch

How each invocation knows which element to process, and how cthreads chooses how
many workgroups to launch.

# Contents

- [The five builtins](#the-five-builtins)
- [Prefer `GlobalIdx` for 1D kernels](#prefer-globalidx-for-1d-kernels)
- [Workgroup size](#workgroup-size)
- [How `group_count_x` is inferred](#how-group_count_x-is-inferred)
- [Bounds checks](#bounds-checks)
- [2D and 3D axes](#2d-and-3d-axes)
- [Mapping to GLSL names](#mapping-to-glsl-names)

# The five builtins

Import from `cthreads.gpu`:

```python
from cthreads.gpu import (
    ThreadIdx,  # index inside the workgroup
    BlockIdx,   # which workgroup
    BlockDim,   # workgroup size
    GridDim,    # number of workgroups
    GlobalIdx,  # global index across the grid
)
```

Each exposes `.x`, `.y`, and `.z` axis markers. In 0.2.0 the public launch path is
effectively **1D in X** for sizing (`group_count_y` / `group_count_z` default to 1),
so most kernels only need `.x`.

Relationship (same as CUDA 1D):

```text
GlobalIdx.x == BlockIdx.x * BlockDim.x + ThreadIdx.x
```

# Prefer `GlobalIdx` for 1D kernels

Element-wise kernels almost always look like this:

```python
from cthreads.gpu import Gpu, GlobalIdx

@Gpu
def axpy(n: int, a: float, x: list[float], y: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    y[i] = a * x[i] + y[i]
```

Use `ThreadIdx` / `BlockIdx` when you intentionally structure work per workgroup
(for example future shared-memory tiles). Today, without shared memory, most
user code can stay on `GlobalIdx`.

# Workgroup size

Default `local_size_x` is **64**. That is the number of invocations in one
workgroup along X.

Implications:

- Launch count is rounded up: for `n = 100`, you get enough workgroups that
  `group_count_x * 64 >= 100`, so some invocations have `i >= n`.
- Workgroup barriers wait for those 64 (or fewer active) peers in one group, not
  for all `n` elements.

# How `group_count_x` is inferred

When you call `gpu(fn, *args)`, cthreads sets `group_count_x` if metadata does not
already fix it:

1. If there is an `int` parameter named **`n`**, use that value as the problem size.
2. Otherwise use the length of the longest list argument.
3. Then `group_count_x = ceil(n / local_size_x)` (at least 1).

Recommendation: **pass an explicit `n: int`** as the first scalar and keep list
lengths consistent with `n`. That makes dispatch intent obvious in the signature.

```python
@Gpu
def fill(n: int, value: float, out: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    out[i] = value

out = [0.0] * 1000
gpu(fill, len(out), 1.0, out).join()
```

# Bounds checks

Always guard:

```python
i: int = GlobalIdx.x
if i >= n:
    return
```

Without the guard, padded invocations can write past the logical end of your
arrays. This is one of the most common GPU correctness bugs for newcomers; it is
mechanical once you treat dispatch rounding as normal.

# 2D and 3D axes

`.y` and `.z` exist on the builtins for dialect completeness and for future
multi-dimensional launches. The high-level `gpu()` path currently fills
`group_count_y = 1` and `group_count_z = 1` when unset. Prefer 1D indexing with
`GlobalIdx.x` unless you have a clear reason to use other axes and matching
launch metadata.

# Mapping to GLSL names

| cthreads | GLSL built-in |
|----------|----------------|
| `ThreadIdx` | `gl_LocalInvocationID` |
| `BlockIdx` | `gl_WorkGroupID` |
| `BlockDim` | `gl_WorkGroupSize` |
| `GridDim` | `gl_NumWorkGroups` |
| `GlobalIdx` | `gl_GlobalInvocationID` |

You do not write GLSL by hand for ordinary kernels; the table is for readers who
already know Vulkan/GLSL or are debugging lowered shaders.

## See also

- [concepts.md](./concepts.md#workgroups-and-the-grid)
- [sync.md](./sync.md) - barriers are workgroup-scoped
- [examples.md](./examples.md)
