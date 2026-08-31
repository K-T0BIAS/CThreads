
![LOGO](./docs/__ressources/CTHREADS_03_1.svg)

----

**cthreads** compiles a typed Python subset into C++ so work can run on real OS threads **without the GIL** allowing true concurrency without `multiprocessing`’s process boundaries and pickling tax.

Use `@Thread` on functions/methods and `@Threadable` on classes. The whitelist covers the usual scalars and containers, plus your own Threadable types. Code runs at native speed while you keep a Python-shaped control flow (jobs, pools, sync).

----

### Docs

- [Install](./docs/install.md)
- [Guides](./docs/index.md)
- [Release (GitHub / PyPI)](./docs/release.md)
- [Math & linalg](./docs/guide/math_and_linalg.md)
- [Compiler notes](./docs/COMPILER.md)
- [Sync / state writeback](./docs/sync_state_docs.md)
- [API reference](./docs/API.md)

----

# Install

**Python ≥ 3.10**, a **C++17** compiler, and **CMake ≥ 3.18**. Full toolchain notes (MSVC / gcc / clang, venv CMake): [docs/install.md](./docs/install.md).

```bash
python -m venv .venv
# activate, then:
pip install cmake ninja    # CMake/Ninja in the venv; compiler is still system/MSVC
pip install -e ".[test]"   # or: pip install -e .
```

First `cthreads.thread(...)` auto-runs cache-checked `prepare` + `load_kernels`. Call `unload_kernels()` before a force rebuild (`thread(..., force=True)` or `prepare(force=True)`).

----

# Introduction

Annotate what should become a native kernel:

- **`@Thread`** - functions / methods compiled to C++
- **`@Threadable`** - classes compiled to C++ structs (shared state across kernels)

## Supported types

Allowed in annotations (arguments, returns, locals, Threadable fields):

* `int`, `float`, `bool`, `str`
* `list[...]` of allowed types
* `dict[...]` of allowed types (typically `dict[str, ...]`)
* nested combinations of the above
* [`@Threadable`](#threadable) classes
* any internal types imported by `cthreads`

This is a **whitelist**, not full Python. No arbitrary objects, no untyped values in kernels.

## `@Thread`

Marks a function or method for compilation. Pass it to `cthreads.thread(...)` to run off the GIL.

```python
from cthreads import Thread

@Thread
def my_example_function() -> None:
    return None
```

### Rules

1. **Typed parameters and a return type** (use `-> None` when there is no value).
2. **No `*args` / `**kwargs`.**
3. **Locals must be annotated** with an allowed type (`x: int = 0`).
4. Inside the body, only call other **`@Thread`** functions/methods, plus **`python math (import math)`**, **`cthreads.modules`**, not arbitrary Python.
5. Return values must match the declared return type.

```python
from cthreads import Thread

@Thread
def example_function(val1: int, val2: list[float], val3: ExampleClass) -> ExampleClass:
    var4: str = "hello there"
    var5: int = 42

    val3.some_string_attr = var4
    val3.some_int_attr = var5
    return val3
```

## `@Threadable`

Python’s open object model does not map cleanly to C++. `@Threadable` marks a class so the compiler can emit a fixed C++ struct and marshal it safely.

```python
from cthreads import Threadable

@Threadable
class MyExample:
    x: float
    y: float
```

### Rules

1. **All fields are typed** at class scope (dataclass-style annotations).
2. **Do not define / override `__init__`.** The decorator injects a dataclass-style constructor (`ExampleClass(1, "x")` or `ExampleClass(attr1=1)`; omitted fields zero / empty, matching C++ `T{}`).
3. **Kernel methods must use `@Thread`** and take **`self`** like normal methods.
4. Method argument / return annotations must be allowed types (or `-> None`).

```python
from cthreads import Threadable, Thread

@Threadable
class ExampleClass:
    attr1: int
    attr2: str
    attr3: list[float]

    @Thread
    def method1(self) -> None:
        self.attr1 += 1

    @Thread
    def method2(self, string: str) -> str:
        return self.attr2 + string


obj = ExampleClass(0, "1", [2.0, 3.0])

obj.method1()
print(obj.attr1, obj.method2(" 1"))  # 1  1 1
```

**Why Threadables?**

- shared state for worker threads
- typed containers / domain objects
- grouping related kernel methods

----

# Run a `@Thread`

```python
import cthreads
from cthreads import Thread

@Thread
def example_function(lhs: float, rhs: float, count: int) -> float:
    for i in range(count):
        lhs += rhs
    return lhs

# Sync: Job -> join -> result
job = cthreads.thread(example_function, 1.5, 2.0, 200)
job.join() # starts if needed; blocks this thread (GIL released in C++)
result = job.result()

# Async: await auto-starts and returns the result (event loop stays free)
job = cthreads.thread(example_function, 1.5, 2.0, 200)
result = await job
```

Signature: `cthreads.thread(fn, *args, force: bool = False, **kwargs) -> Job`.

----

# What’s next

**TODO LINK DOCS WHEN COMPLETE**
