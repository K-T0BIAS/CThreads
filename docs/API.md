# CTHreads API

Compile `@Thread` / `@Threadable` Python into native C++ kernels and run them **off the GIL**.

```text
decorate → prepare/load → thread(...) → await job  (or join)
```

---

## 1. Imports

```python
from cthreads import (
    Thread,
    Threadable,
    prepare,
    thread,
    spawn,
    Job,
    compile,
    build,
    load_kernels,
    unload_kernels,
    kernel_path,
    sync,
    math,
)
```

Native helpers (same pattern for both):

```python
from cthreads import sync, math

lock = sync.Lock()
x = math.abs(-1.0)
```

Do **not** add shadow `sync.py` / `math.py` modules or assign into `sys.modules["cthreads.sync"]` — that can double-init the extension.

---

## 2. Allowed types

Used on `@Threadable` fields and `@Thread` parameters / returns.

| Python | C++ (approx.) | Notes |
|--------|----------------|--------|
| `int` | `int` | |
| `float` | `double` | |
| `bool` | `bool` | |
| `str` | `std::string` | |
| `list[T]` | `std::vector<T>` | `T` must itself be allowed |
| `dict[K, V]` | `std::unordered_map<K,V>` | keys: `str` or `int` for dispatch |
| `@Threadable` class | generated `struct` | nestable |
| `cthreads.sync.Lock` / `Event` / `RWLock` | native sync types | marked internal |

**Not supported (yet):** `set[...]`, most other stdlib / third-party types, untyped / `Any`.

Every parameter and return (except `-> None`) needs a resolvable type hint.

---

## 3. `@Thread` — free functions

Mark a function to compile into a native kernel.

### Rules
- Type-hint all parameters and the return (or `-> None`).
- Only allowed types (see above).
- No `*args` / `**kwargs` / keyword-only-only oddities in the supported subset.
- Locals must be introduced with **annotated** assignment (`x: int = 0`), not bare `x = 0`.
- Body language is a **subset** of Python (see §7).

### Example

```python
from cthreads import Thread, thread

@Thread
def add(a: int, b: int) -> int:
    return a + b

# first call may prepare + load kernels (cached afterward)
job = thread(add, 2, 3)
result = await job          # preferred (async)
# sync:
# job.start(); job.join(); print(job.result())
```

---

## 4. `@Threadable` — data + methods

A `@Threadable` class becomes a C++ `struct`. Fields are class annotations. Methods that run off the GIL are marked `@Thread`.

### Rules
- **No `__init__`** — declare fields with annotations only.
- Field types must be allowed (see §2).
- `@Thread` methods: first parameter is `self`; other args/return follow `@Thread` rules.
- Nested `@Threadable` types and `list[SomeThreadable]` are allowed (including self-refs via quotes / postponed evaluation where needed).

### Example

```python
from cthreads import Thread, Threadable, thread

@Threadable
class Particle:
    x: float
    y: float
    velocity: float

    @Thread
    def step(self, dt: float) -> None:
        self.x += self.velocity * dt

# Python-side instance (attribute bag)
p = Particle()
p.x = 0.0
p.y = 0.0
p.velocity = 1.5

# Methods: pass the unbound function + instance as `self`
job = thread(Particle.step, p, 0.016)
await job
# p.x / p.y may be written back after join depending on pack/writeback
```

**Invalid**

```python
# bound method with no explicit self arg — missing `self` in bind_args
thread(p.step)

# __init__ on Threadable — not allowed by API rules
@Threadable
class Bad:
    def __init__(self, x: float):
        self.x = x
```

---

## 5. Running work: `thread`, `spawn`, `Job`

### `thread(fn, *args, force=False, **kwargs) -> Job`
High-level entry:

| Situation | Behavior |
|-----------|----------|
| Kernels already loaded, `force=False` | **Spawn only** (safe under concurrency) |
| Nothing loaded | Cache-checked `prepare` + `load_kernels`, then spawn |
| `force=True` while loaded | **Raises** — call `unload_kernels()` first |

```python
job = thread(add, 1, 2)
print(await job)                 # auto-start + wait off the event loop

job = thread(add, a=1, b=2)      # kwargs by parameter name
job.start()
job.join()
print(job.result())
```

### `spawn(fn, *args, **kwargs) -> Job`
Low-level: bind + spawn only. Kernels must already be loaded (`prepare` + `load_kernels` or a prior `thread()`).

### `Job` API

| Method | Meaning |
|--------|---------|
| `start()` | Start OS worker; returns `self` |
| `await job` | Auto-start if needed; wait without blocking the asyncio loop; return `result()` |
| `join()` | Block this OS thread until done (GIL released in C++) |
| `wait()` | Condition wait until done |
| `done()` | Non-blocking poll |
| `result()` | Return value after completion |

```python
# FastAPI / asyncio
@app.post("/add")
async def endpoint():
    return {"sum": await thread(add, 2, 3)}
```

---

## 6. Prepare / load / unload

```python
path = prepare(force=False)       # codegen + link (hash-cached); does NOT unload
load_kernels(str(path))           # optional warm load at startup

# ... many concurrent thread() calls ...

unload_kernels()                  # manual — e.g. process shutdown or before force rebuild
```

**Force rebuild (Windows):** unload first, then rebuild, then load:

```python
unload_kernels()
path = prepare(force=True)
load_kernels(str(path))
```

Also available: `compile()`, `build()` (lower-level pieces used by `prepare`).

---

## 7. Language subset inside `@Thread` bodies

### Statements
- Annotated assign: `x: int = 1`
- Assign to known names / attributes / subscripts (after annotated declare)
- AugAssign: `+=`, `-=`, `*=`, …
- `return`, `pass`, `if` / `else`, `while`, `break`, `continue`
- `for x in xs:` when `xs` is a `list[...]`
- `for i in range(n)` / `range(a, b)` / `range(a, b, s)` (positive step assumed)

### Expressions
- Literals, names, `obj.attr`, `xs[i]` (no slices)
- Arithmetic / bitwise ops in `pyOps.BINOPS` (including `%`, `**` → `std::pow`)
- Unary `+`, `-`, `not`, `~`
- Comparisons `== != < <= > >=` (including chains)
- Boolean `and` / `or`

### Calls (whitelist only)

**Builtins** (bare name):

```python
n = len(xs)           # → (xs).size()
for i in range(n):    # → C++ index for-loop
    ...
```

**Stdlib `math`** (resolved via globals — `import math` or `from math import sqrt`):

```python
import math

@Thread
def f(x: float) -> float:
    return math.sqrt(x) + math.pi
```

Mapped to `std::*` / `<cmath>` / `std::numbers` for supported names (`sqrt`, `sin`, `log`, `pi`, … — filtered by `hasattr(math, name)` on the running interpreter).

**`cthreads.math`** (module marked internal):

```python
from cthreads import math as cm

@Thread
def g(x: float) -> float:
    return cm.clamp(cm.abs(x), 0.0, 1.0)
```

| Op | Arity |
|----|-------|
| `abs` | 1 |
| `min`, `max`, `uniform`, `randint` | 2 |
| `clamp` | 3 |
| `random` | 0 |
| `seed` | 1 |

Unsupported calls raise at compile/translate time.

### Locals

```python
@Thread
def ok(n: int) -> float:
    s: float = 0.0      # required style
    i: int = 0
    while i < n:
        s += 1.0
        i += 1
    return s

@Thread
def bad(n: int) -> float:
    s = 0.0             # ERROR: declare with annotated assignment first
    ...
```

---

## 8. Sync primitives

```python
from cthreads import Thread, Threadable, sync

@Threadable
class Counter:
    value: int

    @Thread
    def bump(self, lock: sync.Lock) -> None:
        lock.acquire()
        self.value += 1
        lock.release()
```

Available on `cthreads.sync`: `Lock`, `Event`, `RWLock` (GIL released on blocking waits in the Python bindings). Pass instances into kernels like other allowed types.

---

## 9. End-to-end examples

### Free function (sync)

```python
from cthreads import Thread, thread

@Thread
def mul(a: float, b: float) -> float:
    return a * b

job = thread(mul, 1.5, 2.0)
job.start()
job.join()
assert job.result() == 3.0
```

### Free function (async)

```python
import asyncio
from cthreads import Thread, thread

@Thread
def mul(a: float, b: float) -> float:
    return a * b

async def main():
    return await thread(mul, 1.5, 2.0)

assert asyncio.run(main()) == 3.0
```

### Threadable method

```python
from cthreads import Thread, Threadable, thread

@Threadable
class Vec2:
    x: float
    y: float

    @Thread
    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y

v = Vec2()
v.x = 3.0
v.y = 4.0
assert await thread(Vec2.length_sq, v) == 25.0
```

### Startup warm-load (servers)

```python
from contextlib import asynccontextmanager
import cthreads
import kernels  # registers @Thread functions on import

@asynccontextmanager
async def lifespan(app):
    path = cthreads.prepare(force=False)
    cthreads.load_kernels(str(path))
    yield
    cthreads.unload_kernels()
```

Then per request: `await cthreads.thread(kernels.burn_ct, n)` with no unload races.

---

## 10. Quick reference — do / don’t

| Do | Don’t |
|----|--------|
| Annotate fields, params, returns, locals | Use bare `x = 0` for new locals |
| `thread(Cls.method, instance, ...)` | `thread(instance.method)` alone |
| Field annotations on `@Threadable` | `__init__` on `@Threadable` |
| `await job` or `join()` after work | Assume unbound concurrent `thread(force=True)` while loaded |
| `unload_kernels()` only when you mean it | Expect `prepare`/`thread` to unload for you |
| Whitelisted calls (`math`, `cthreads.math`, `len`, `range`) | Arbitrary Python calls inside `@Thread` |
