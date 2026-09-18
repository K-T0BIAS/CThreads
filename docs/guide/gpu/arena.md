# GpuArena (resident lists)

`GpuArena` keeps named Python lists bound to process-wide device buffers so repeated
`gpu()` launches can skip alloc/upload when you pass the **same list objects**.

# Contents

- [When to use it](#when-to-use-it)
- [Basic pattern](#basic-pattern)
- [Option B: same list objects](#option-b-same-list-objects)
- [API](#api)
- [Host edits between launches](#host-edits-between-launches)
- [Length and type stability](#length-and-type-stability)
- [Release and context managers](#release-and-context-managers)
- [Pitfalls](#pitfalls)

# When to use it

| Situation | Recommendation |
|-----------|----------------|
| One or few launches | Default `gpu(...).join()` is enough |
| Many launches over the same arrays | `GpuArena` + `join(download=False)` + occasional `sync()` |
| Need Python to read results every iteration | Default join download, or `sync()` each time (you pay the transfer) |

Arena does not change kernel source. It only changes how launch finds buffers.

# Basic pattern

```python
from cthreads.gpu import Gpu, GlobalIdx, GpuArena, gpu

@Gpu
def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
    i: int = GlobalIdx.x
    if i >= n:
        return
    y[i] = a * x[i] + y[i]

x = [1.0] * 1_000_000
y = [0.0] * 1_000_000
n = len(x)
a = 1.0001

with GpuArena() as arena:
    arena.bind(x=x, y=y)
    for _ in range(200):
        gpu(saxpy, n, a, x, y).join(download=False)
    arena.sync()  # download x and y into the Python lists
```

# Option B: same list objects

Launch recognizes residency by **object identity** (`id(list)`) plus length checks:

1. `arena.bind(x=x, ...)` uploads and registers `x`.
2. Later `gpu(..., x, ...)` looks up `id(x)`.
3. If found and length still matches, launch **borrows** the resident buffer
   instead of allocating a fresh one.

Consequences:

- Pass the **same** list instance you bound. A copy (`x2 = list(x)`) is a different
  object and will not hit the resident path.
- Do not replace the list variable with a new list of the same values without
  rebinding.

# API

### `GpuArena()`

Creates a session with a unique id. Prefer `with GpuArena() as arena:`.

### `arena.bind(**named_lists)`

```python
arena.bind(positions=pos, velocities=vel)
```

- Keyword names are slot names inside the arena (`"positions"`, ...).
- Values must be non-empty `list` objects (`int` / `float` / `bool` elements).
- Element kind is inferred from the first element (`bool` before `int`).
- Rebinding the same slot name with the same list, length, and kind re-uploads.
- Rebinding with a different shape destroys and recreates the device buffer.
- A given list object cannot be bound under two different slots/arenas at once.

Returns `self` for chaining: `GpuArena().bind(x=x).bind(y=y)` is valid, though one
`bind(x=x, y=y)` call is clearer.

### `arena.sync(*names)`

Downloads device bytes into the bound Python lists.

```python
arena.sync()          # all slots
arena.sync("y")       # one slot by bind name
arena.sync("x", "y")
```

### `arena.release()`

Destroys device buffers owned by this arena and clears identity registrations.
Called automatically from `__exit__`.

### `arena.names()` / `name in arena`

Introspection helpers for bound slot names.

# Host edits between launches

There is no proxy that watches Python list mutations. If you change host list
contents in Python between launches and need the device to see them:

```python
# host updated y in Python
arena.bind(y=y)  # re-upload that slot
gpu(kernel, n, x, y).join(download=False)
```

If only the device mutates the lists, re-bind is unnecessary; just launch again.

# Length and type stability

If a bound list's length changes, launch raises `GpuInvalidArgument` and asks you
to bind again. Shrinking/growing arrays mid-session means: rebuild the list, bind
again, continue.

Element kind must keep matching the kernel parameter (`list[float]` vs inferred
kind from the host list).

# Release and context managers

```python
arena = GpuArena()
try:
    arena.bind(x=x)
    gpu(fn, n, x).join(download=False)
    arena.sync("x")
finally:
    arena.release()
```

After `release()`, further `bind` / `sync` raise `GpuInvalidArgument`.

# Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| `join()` with default download every iteration | Slow loop; residency benefits muted | `join(download=False)` + `sync` when needed |
| New list each iteration | Misses resident lookup | Reuse bound objects |
| Read Python lists without `sync` | Stale host values | `arena.sync()` or downloading join |
| Empty list at bind | `GpuInvalidArgument` | Bind non-empty lists |
| Two arenas bind the same list | `GpuInvalidArgument` | One bind owner at a time |

## See also

- [launch.md](./launch.md#join-and-download)
- [best_practices.md](./best_practices.md)
- [examples.md](./examples.md#resident-multi-pass-loop)
