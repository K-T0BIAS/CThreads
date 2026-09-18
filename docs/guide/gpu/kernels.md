# `@Gpu` kernels

How to author device functions: types, language subset, math helpers, and how
decoration differs from launch.

# Contents

- [Decorating](#decorating)
- [Allowed types](#allowed-types)
- [Return type and outputs](#return-type-and-outputs)
- [Locals and annotations](#locals-and-annotations)
- [Language subset](#language-subset)
- [Math and casts](#math-and-casts)
- [Imports inside kernels](#imports-inside-kernels)
- [What is rejected](#what-is-rejected)
- [Decorate vs launch vs call-as-Python](#decorate-vs-launch-vs-call-as-python)

# Decorating

```python
from cthreads.gpu import Gpu, GlobalIdx

@Gpu
def fill(n: int, value: float, out: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    out[i] = value
```

(`GlobalIdx` is documented in [indexes.md](./indexes.md).)

Optional logging of the assigned device name:

```python
@Gpu(log=True)
def fill_logged(n: int, value: float, out: list[float]) -> None:
    ...
```

Decoration:

1. Checks that the GPU path is available (`gpu.available()`).
2. Validates annotations against the GPU allowlist.
3. Builds launch metadata on the function (`__gpu_kernel_meta__`).
4. Registers the function for later SPIR-V compile.

Decoration does **not** upload data or run the kernel.

# Allowed types

| Annotation | Role | Notes |
|------------|------|-------|
| `int` | Scalar input | Packed into the scalar storage buffer |
| `float` | Scalar input | Device `float` (32-bit). CPU `@Thread` uses double; GPU follows GLSL float |
| `bool` | Scalar input | Stored as 32-bit on the device packing path |
| `list[int]` | Buffer (writeback) | Same Python list updated on join |
| `list[float]` | Buffer (writeback) | Same |
| `list[bool]` | Buffer (writeback) | Same |

Not supported on the GPU path (raise at meta build / decorate):

- `str`, `dict[...]`
- `@Threadable` classes
- `cthreads.sync` lock / event / TBuffer annotations as kernel parameters
- Nested lists (`list[list[float]]`)
- `set`, optional types, `Any`

Structure-of-arrays is the intended style: parallel lists of scalars
(`pos_x`, `pos_y`, `vel_x`, ...) rather than a list of particle objects.

# Return type and outputs

Every `@Gpu` function must be annotated `-> None`.

Outputs are **in-place list mutations**. After `join()`, inspect those lists on
the host. Scalar parameters are not written back.

```python
@Gpu
def add_scaled(
    n: int,
    alpha: float,
    src: list[float],
    dst: list[float],
) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    dst[i] = dst[i] + alpha * src[i]
```

# Locals and annotations

Locals must use annotated assignment, same discipline as `@Thread`:

```python
i: int = GlobalIdx.x
tmp: float = x[i] * a
ok: bool = i < n
```

Bare `i = GlobalIdx.x` is not part of the supported subset.

# Language subset

Supported control flow (compiled to GLSL):

| Construct | Support |
|-----------|---------|
| `if` / `else` | Yes |
| `while` | Yes (no `while`/`else`) |
| `for i in range(...)` | Yes (`range` with 1-3 args; no keywords) |
| `return` | Early exit only (`return;` / bare `return`) |
| `break` / `continue` | Via the shared flow helpers (same spirit as CPU) |
| Expression statements that are calls | Yes (math, barriers) |

Not supported:

- `for x in some_list:` (no list iteration)
- `for`/`else`, `while`/`else`
- Rebinding an existing name as a `for` target
- Arbitrary Python calls, comprehensions, generators, `try`/`except`, classes, nested `def`

Index expressions on list parameters (`y[i] = ...`) are the primary memory API.

# Math and casts

The GPU CallPlugin path currently lowers:

| Python form | Device |
|-------------|--------|
| `sqrt(x)` or `math.sqrt(x)` | GLSL `sqrt` |
| `floor(x)` or `math.floor(x)` | GLSL `floor` (toward -inf) |
| `int(x)` | GLSL `int` (truncate toward zero) |

Example:

```python
import math
from cthreads.gpu import Gpu, GlobalIdx

@Gpu
def cell_of(n: int, inv_h: float, x: list[float], cell: list[int]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    c: float = math.floor(x[i] * inv_h)
    cell[i] = int(c)
```

Richer math will grow over time; do not assume the full `math` module is available
inside `@Gpu`. Prefer the helpers above or express formulas with `+ - * /` and
comparisons.

# Imports inside kernels

You may import names that the translator recognizes (`math` for `math.sqrt` /
`math.floor`, sync barrier names, index builtins). The compiled kernel does not
run Python imports on the device; matching is by AST shape at compile time.

Typical host-side imports for authoring:

```python
from cthreads.gpu import Gpu, GlobalIdx, BlockDim, ThreadIdx
from cthreads.sync import Barrier, __sync_threads
```

# What is rejected

| Pattern | Why |
|---------|-----|
| `Barrier(4)` inside `@Gpu` | Host-style construction; use `Barrier.arrive_and_wait()` |
| Valued `return x` | Kernels are void; use list writeback |
| Keyword args to `gpu(fn, x=...)` | Not supported yet; positional only |
| Mid-run Python reads of lists | No GPU `sync_state`; join or `arena.sync` |
| Assuming `float` is IEEE double | Device float is 32-bit |

# Decorate vs launch vs call-as-Python

| Action | Effect |
|--------|--------|
| `@Gpu` on a function | Validate + register |
| Direct `fn(*args)` in Python | Runs the Python source as ordinary Python (useful for tiny checks; not the device path) |
| `gpu(fn, *args)` | Compile if needed, upload, dispatch, return `GpuJob` |
| `job.join()` | Wait (+ download lists by default) |

For production device work, always go through `gpu(...)`.

## See also

- [indexes.md](./indexes.md)
- [launch.md](./launch.md)
- [sync.md](./sync.md)
