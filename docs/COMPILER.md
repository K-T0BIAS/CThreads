# End-to-end: how cthreads compiles and runs kernels

This document walks through **one program** from Python source to generated C++, then through packing, running the kernel, mid-run sync, and writing results back.

Everything below uses the **same** example. Later sections only add detail; they do not change the story.

**Audience**

- Want to *use* the library? Read §0–§11.
- Want to *change the compiler*? Read §0–§4 so the example is in your head, then jump to **§12 (compile)** and **§13 (translation)**. Those name every moving part of “Python function in, C++ files out.”

**Layout note (V2 root)**

Public package root is `cthreads/` (former V2). Important paths:

```text
src/cthreads/python/cthreads/
  frontend/          # @Thread, @Threadable, REGISTRY
  compiler/
    orchestrator/    # CompileSession, ThreadableUnit, ThreadUnit
    translation/     # Signature, Syntax, Typeof, plugins
  types/             # PyType / hint_to_pytype
  sync/              # TBuffer host API (+ Lock/Event from _ext)
  kernel_meta.py     # KERNELS, trampolines, tbuffer runtime emit
  marshal.py         # pack / writeback / unpack
  prepare.py         # compile / prepare / thread
  build.py
```

There is **no** `STORE` / `CONFIG`. Paths live on `REGISTRY.threadable_units` / `thread_units`. Kernel schemas live in `KERNELS` / `fn.__kernel_meta__`.

---

## 0. The Python program

Assume this file lives at:

```text
my_project/demo.py
```

```python
from __future__ import annotations

import math

from cthreads import (
    Thread,
    Threadable,
    thread,
    sync,
    TBuffer,
    create_tbuffer,
    __sync_state,
)
from cthreads import linalg  # from native _ext when installed

Lock = sync.Lock
ArrayF32 = linalg.ArrayF32
Shape = linalg.Shape


@Threadable
class Vec2:
    x: float
    y: float


@Threadable
class Particle:
    pos: Vec2
    velocity: float
    tags: list[str]
    scores: dict[str, float]
    lock: Lock

    @Thread
    def kick(self, dv: float) -> None:
        self.lock.acquire()
        self.velocity = self.velocity + dv
        self.lock.release()


@Thread
def simulate_step(
    p: Particle,
    forces: list[float],
    dt: float,
    buf: TBuffer[Particle],
    slot: int,
    vel_field: ArrayF32,
) -> bool:
    """
    Mutate one particle, touch lists/dicts, use math + linalg + TBuffer,
    and optionally publish a mid-run snapshot via __sync_state().
    """
    # annotated locals only
    i: int = 0
    n: int = len(forces)
    ax: float = 0.0

    p.lock.acquire()
    while i < n:
        ax = ax + forces[i]
        i = i + 1

    # stdlib math → std::…
    scale: float = math.sqrt(ax * ax + 1.0)
    p.velocity = p.velocity + scale * dt
    p.pos.x = p.pos.x + p.velocity * dt
    p.pos.y = p.pos.y + 0.5 * dt

    # containers (list/dict methods go through ContainerMethodPlugin)
    p.tags.append("stepped")
    p.scores.clear()
    # d.get(k, default) / d.pop(k, default) are also supported; bare `k in d` is not

    # linalg: reshape a 1-D view of the velocity field
    sh: Shape = Shape([n])
    view: ArrayF32 = vel_field.reshape(sh)
    if view.numel() > 0:
        # property → C++ method
        _ndim: int = view.ndim

    # triple-buffer: write this particle into a slot, publish
    # (exact host/kernel TBuffer API: publish / read_at / generation)
    buf.publish()

    p.lock.release()

    # mid-run writeback barrier (host can observe p / lists so far)
    __sync_state()

    fast: bool = p.velocity > 10.0
    return fast


# ---------- host setup ----------
p = Particle(Vec2(1.0, 2.0), 3.0, ["spawn"], {"energy": 1.0}, Lock())

forces = [0.5, 0.25]
vel_field = ArrayF32(Shape([2]))  # host-built; passed into the kernel
buf = create_tbuffer(Particle, capacity=8)

job = thread(simulate_step, p, forces, 0.1, buf, 0, vel_field)
job.start()
job.join()

print(p.pos.x, p.velocity, p.tags)
print(job.result())  # True or False
```

What this program *means* in plain language:

1. `Vec2` / `Particle` are data bags (structs). `Particle` also owns a `Lock` and nested/container fields.
2. `Particle.kick` is a **method kernel** (C++ member + trampoline).
3. `simulate_step` is a **free kernel** that mixes arithmetic, `math`, containers, linalg, `TBuffer`, and `__sync_state`.
4. `thread(...)` does **not** run the Python body. It runs the compiled C++ copy on an OS thread, then copies structured mutations back into the Python objects.

The rest of this document shows every intermediate step that makes that true. For trampoline / pack detail we zoom in on **`simulate_step`’s** simpler cousin from the classic docs where needed — the machinery is identical; only schemas grow with nested fields / lists / TBuffer.

---

## 1. What happens when Python loads `demo.py`

Importing and decorating does **not** compile C++ yet. It only **marks and registers** things.

### 1.1 `@Threadable` on `Vec2` / `Particle`

The decorator `Threadable` (`frontend/Threadable/wrapper.py`) runs on the class right after the class body is defined.

It does these things, in order:

1. Sets `Cls.__threadable = True`.
2. Sets `Cls.__threadable_version` to `REGISTRY.VERSION`.
3. Resolves field annotations with `get_type_hints` (so `list[str]`, nested `Vec2`, `Lock`, etc. work) and checks each type via `is_threadable`.
4. Requires every method on the class to be `@Thread`-decorated (no plain Python methods).
5. Rejects a user-defined `__init__`.
6. Registers the class:

   ```text
   REGISTRY.threadables["Vec2"] = Vec2
   REGISTRY.threadables["Particle"] = Particle
   ```

7. Injects a dataclass-style `__init__` (host `Cls(1.0, 2.0)` / keywords; omitted fields zero / empty). Compile later emits `Name() = default` and `field{}` on the C++ struct.

So after decoration, these are still normal Python classes. You create instances with the generated constructor. The decorator’s job is: “remember this class so compile can turn it into a C++ `struct` later.”

### 1.2 `@Thread` on `kick` and `simulate_step`

The decorator `Thread` (`frontend/Thread/wrapper.py`) runs on each function.

It does these things, in order:

1. Sets `fn.__threaded = True`.
2. Sets `fn.__thread_version` to `REGISTRY.VERSION`.
3. Checks type hints (all params annotated; return annotated or `-> None`; no `*args` / `**kwargs`).
4. Registers the function:

   ```text
   REGISTRY.threads["Particle.kick"] = kick      # __qualname__
   REGISTRY.threads["simulate_step"] = simulate_step
   ```

5. Returns the **same** function. Calling `simulate_step(...)` as normal Python still runs the Python body. Only `thread(simulate_step, ...)` uses the compiled path.

### 1.3 State after import (before any compile)

| Name | Extra flags / registry entry |
|------|------------------------------|
| `Vec2`, `Particle` | `__threadable=True`, in `REGISTRY.threadables` |
| `kick`, `simulate_step` | `__threaded=True`, in `REGISTRY.threads` |
| `REGISTRY.threadable_units` / `thread_units` | **empty** until compile |
| C++ files | **none yet** |
| Kernel DLL | **none yet** |
| `fn.__kernel_meta__` | **missing until emit** |

---

## 2. When compile runs

Compile usually happens on the first `thread(...)` call if no kernel library is loaded yet:

```text
thread(simulate_step, ...)
  → prepare()
      → compile()   # CompileSession: units → emit .hpp/.cpp
      → build()     # compile those into one shared library
  → load_kernels(path_to_shared_library)
  → spawn the job
```

You can also call `cthreads.compile(force=...)` / `prepare(force=...)` explicitly.

### 2.1 `CompileSession.compile()` drain order

`compiler/orchestrator/session/compileSession.py`:

1. Clear `REGISTRY.threadable_units`, `REGISTRY.thread_units`, and `KERNELS`.
2. Require a non-empty `REGISTRY.threadables` or `REGISTRY.threads`.
3. Pick **project root** = directory of the first registered class/function’s source file (`my_project/`).
4. Ensure `.gitignore` for generated trees; load `.cthreads_cache.json`.
5. **Build ThreadableUnits** for every `@Threadable` (fields via `hint_to_pytype`, attach method `ThreadUnit`s, claim those `__qualname__`s).
6. **Build free ThreadUnits** for remaining `@Thread` functions.
7. **Emit** every ThreadableUnit, then every ThreadUnit (`force` / `src_hash` short-circuit inside `emit`).
8. `write_tbuffer_runtime(root)` if any kernel uses `TBuffer[Threadable]`.
9. Save cache; return `{root, cache, rewritten}`.

For our example that means (conceptually):

```text
1. emit Vec2       →  my_project/__Threadable__/Vec2.{hpp,cpp}
2. emit Particle   →  my_project/__Threadable__/Particle.{hpp,cpp}
                      (includes kick member + trampolines)
3. emit simulate_step → my_project/__Thread__/simulate_step.{hpp,cpp}
4. emit tbuffer runtime → my_project/__Thread__/cthreads_tbuffer.{hpp,cpp}
```

Also:

```text
my_project/__Thread__/cthreads_export.hpp
```

Folder layout after compile:

```text
my_project/
  demo.py
  .cthreads_cache.json
  __Threadable__/
    Vec2.hpp
    Vec2.cpp
    Particle.hpp
    Particle.cpp
  __Thread__/
    cthreads_export.hpp
    cthreads_tbuffer.hpp
    cthreads_tbuffer.cpp
    simulate_step.hpp
    simulate_step.cpp
```

What `compile()` does internally (units, cache, fingerprint) is spelled out in **§12**. The next sections stay with the example: Threadables first, then `simulate_step`.

---

## 3. Compiling Threadables (`Vec2`, `Particle`)

### 3.1 Units, not STORE

After the session fills maps:

```text
REGISTRY.threadable_units["Vec2"] = ThreadableUnit(
  handle=...,
  fields={"x": PyFloat(), "y": PyFloat()},
  hpp_path=.../__Threadable__/Vec2.hpp,
  cpp_path=.../__Threadable__/Vec2.cpp,
  methods=[],
)

REGISTRY.threadable_units["Particle"] = ThreadableUnit(
  fields={
    "pos": PyThreadable("Vec2"),
    "velocity": PyFloat(),
    "tags": PyList(PyString()),
    "scores": PyDict(PyString(), PyFloat()),
    "lock": PyCThreadsInternalType("Lock", ...),
  },
  methods=[ThreadUnit for Particle.kick],
  hpp_path=.../Particle.hpp,
  ...
)
```

`build()` and `include_for` read **`unit.hpp_path`**, not a global path dict.

### 3.2 Type mapping for fields

| Python annotation | Internal mapping | C++ field |
|-------------------|------------------|-----------|
| `x: float` | `PyFloat` | `double x;` |
| `pos: Vec2` | `PyThreadable("Vec2")` | `Vec2 pos;` (+ `#include` of Vec2.hpp) |
| `tags: list[str]` | `PyList(PyString)` | `std::vector<std::string> tags;` |
| `scores: dict[str, float]` | `PyDict` | `std::unordered_map<std::string, double> scores;` |
| `lock: Lock` | `PyCThreadsInternalType` | `cthreads::sync::Lock lock;` (+ sync header) |

`Lock` / `ArrayF32` / … are **not** Threadables. They are marked `__cthreads_internal__` and looked up in `CTHREADS_INTERNAL_TYPES` (`types/pyType/internal/include_map.py`): link the shipped header, do not generate a struct.

### 3.3 Generated `Vec2.hpp` (shape)

```cpp
#pragma once

#include "../__Thread__/cthreads_export.hpp"

struct Vec2 {
    double x;
    double y;
};
```

### 3.4 Generated `Particle.hpp` (shape)

Includes relative to this file, export macro, nested Threadable, STL, sync:

```cpp
#pragma once

#include "../__Thread__/cthreads_export.hpp"
#include "Vec2.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include "sync/pyLock.hpp"

struct Particle {
    Vec2 pos;
    double velocity;
    std::vector<std::string> tags;
    std::unordered_map<std::string, double> scores;
    cthreads::sync::Lock lock;

    void kick(double dv);
};

// C trampoline surface for the method (lifecycle; accessors in .cpp)
CTHREADS_API void* Particle_kick__args_new();
CTHREADS_API void  Particle_kick__args_free(void* p);
CTHREADS_API void  Particle_kick__call(void* p);
```

(Exact export names follow `kernel_meta` for symbol `"Particle_kick"`.)

### 3.5 Method body: `self` → `this`

`kick` translates with `owner=Particle` unit:

| Python | C++ |
|--------|-----|
| `self.lock.acquire()` | `(this->lock).acquire();` |
| `self.velocity = self.velocity + dv` | `this->velocity = (this->velocity + dv);` |

`FieldAttrPlugin` lowers `self.<attr>` when `ctx.owner is not None`. Sync methods go through `SyncMethodPlugin` after `Typeof` resolves `this->lock` as `Lock`.

---

## 4. Compiling `simulate_step` (the free Thread function)

Three big pieces:

1. Parse the Python function’s AST.
2. Translate signature + body into a real C++ function.
3. Emit **trampoline** helpers so Python can fill an args pack and call through a stable C API.

Entry point: `translate_function(fn, this_file=..., owner=None)` in `compiler/translation/translate.py`.

### 4.1 Load source and get the function AST

`Source.parse_function(fn)` → `inspect.getsource` → dedent → `ast.parse` → `FunctionDef`.

After that, **nobody looks at the original source text again**. Later decisions are “what kind of node is this?”

### 4.2 Translate the signature (`Signature.translate`)

Hints for `simulate_step` (abbreviated):

| Hint | Python type | C++ parameter |
|------|-------------|---------------|
| `p` | `Particle` | `Particle& p` |
| `forces` | `list[float]` | `std::vector<double>& forces` |
| `dt` | `float` | `double dt` |
| `buf` | `TBuffer[Particle]` | `cthreads::sync::tripple_buffer<Particle>& buf` (pass_as `tbuffer` in meta; trampoline uses pointer slot) |
| `slot` | `int` | `int slot` |
| `vel_field` | `ArrayF32` | `cthreads::linalg::Array<float>& vel_field` (internal type) |
| return | `bool` | `bool` |

Threadable / list / dict / TBuffer / many internals are **references** (or tbuffer pointer in the pack) so the kernel can mutate the pack’s copy. Scalars are by value.

Signature includes (relative to `__Thread__/simulate_step.hpp`) pull in Particle, t_buffer, array headers, etc. via `include_for(py_type, this_file)`.

Full free-function signature string (shape):

```cpp
CTHREADS_API bool simulate_step(
    Particle& p,
    std::vector<double>& forces,
    double dt,
    cthreads::sync::tripple_buffer<Particle>& buf,
    int slot,
    cthreads::linalg::Array<float>& vel_field
)
```

`CTHREADS_API` comes from `cthreads_export.hpp` (`extern "C"` + dllexport on Windows).

### 4.3 Translate the body (`Syntax.stmt` / `Syntax.expr`)

Dispatch lives under `compiler/translation/syntax/`:

| Concern | Module |
|---------|--------|
| Constants / list displays | `Literal` |
| Names (`self` → `(*this)`) | `Name` |
| `+` `**` `and` compare | `Op` |
| Assign / AnnAssign / AugAssign | `Assign` |
| if / for / while / return | `Flow` |
| subscript | `Index` |
| Call / Attribute | `Syntax` → **plugins** |

#### Annotated locals

```python
i: int = 0
```

→ name table `symbols["i"] = PyInt()`, emit `int i = 0;`

Plain `i = 0` for a **new** name is rejected. That is deliberate: the name table has no other way to learn the type.

#### `len(forces)`

`LenPlugin`: `(forces).size()` (TBuffer → `.capacity()`).

#### `p.lock.acquire()`

1. `Typeof.of(p.lock)` → Threadable field `lock` → `PyCThreadsInternalType("Lock")`.
2. `SyncMethodPlugin` → `(p.lock).acquire()`.

#### `math.sqrt(...)`

`MathCallPlugin` resolves `math.sqrt` via `fn.__globals__`, arity check, emit `std::sqrt(...)`, `#include <cmath>`.

#### `p.tags.append("stepped")`

`ContainerMethodPlugin` list table: `(p.tags).push_back("stepped")`.

#### `Shape([n])` / `vel_field.reshape(sh)` / `view.ndim`

- List display `[n]` → `std::vector<int>{n}` (element type from name table / literals).
- `LinalgCtorPlugin` / `LinalgMethodPlugin` / `LinalgPropPlugin` (`ndim` property → `(view).ndim()`).

#### `buf.publish()`

`TBufferMethodPlugin` → `(buf).publish()`.

#### `__sync_state()`

`SyncStatePlugin` → `cthreads::detail::__sync_state()` + include `syncState.hpp`. At runtime this is the mid-run writeback barrier into the host Python objects (TLS in `_ext`).

#### `return fast`

→ `return fast;` with `fast` already in the name table as `bool`.

### 4.4 The real C++ kernel (abbreviated)

Putting signature + body together (illustrative; exact parentheses/includes match translators):

```cpp
CTHREADS_API bool simulate_step(
    Particle& p,
    std::vector<double>& forces,
    double dt,
    cthreads::sync::tripple_buffer<Particle>& buf,
    int slot,
    cthreads::linalg::Array<float>& vel_field
) {
    int i = 0;
    int n = (forces).size();
    double ax = 0.0;
    (p.lock).acquire();
    while ((i < n)) {
        ax = (ax + (forces[i]));
        i = (i + 1);
    }
    double scale = std::sqrt(((ax * ax) + 1.0));
    p.velocity = (p.velocity + (scale * dt));
    p.pos.x = (p.pos.x + (p.velocity * dt));
    p.pos.y = (p.pos.y + (0.5 * dt));
    (p.tags).push_back("stepped");
    (p.scores).clear();
    cthreads::linalg::Shape sh = cthreads::linalg::Shape(std::vector<int>{n});
    cthreads::linalg::Array<float> view = (vel_field).reshape(sh);
    if (((view).shape().numel() > 0)) {
        int _ndim = (view).ndim();
        (void)_ndim;
    }
    (buf).publish();
    (p.lock).release();
    cthreads::detail::__sync_state();
    bool fast = (p.velocity > 10.0);
    return fast;
}
```

This function never sees Python objects. It only knows C++ types.

That walk is the whole translation idea. **§13** names every plugin and syntax entry so you can extend it.

---

## 5. Kernel meta and trampolines

The kernel alone is not enough. Python still has Python objects. Something must:

1. Allocate a C++ **pack** (`simulate_step__args`).
2. Copy Python values into that pack (via set trampolines).
3. Call the real kernel from the pack.
4. Copy mutated structured fields back into Python.
5. Unpack the return value.
6. Free the pack.

Those helpers are generated by `kernel_meta.py` during unit `emit`.

### 5.1 Symbol table (pattern)

| Role | Symbol (example) |
|------|------------------|
| Real kernel | `simulate_step` |
| Allocate pack | `simulate_step__args_new` |
| Free pack | `simulate_step__args_free` |
| Call kernel from pack | `simulate_step__call` |
| Pack struct | `simulate_step__args` |
| Param `i` | prefix `a{i}` |
| Threadable field | `a0_pos_x`, `a0_tags_…`, … |
| TBuffer param | `a3` held as pointer; `…__set_a3_ptr` |
| Return | `ret` + `…__get_ret` |

Method kernels use symbols like `Particle_kick__call` with `a0` = `self` as a **pointer** (`pass_as: "ptr"`).

### 5.2 `__kernel_meta__` on the Python function

After emit, roughly:

```python
simulate_step.__kernel_symbol__ = "simulate_step"
simulate_step.__kernel_meta__ = {
  "symbol": "simulate_step",
  "call_symbol": "simulate_step__call",
  "args_new_symbol": "simulate_step__args_new",
  "args_free_symbol": "simulate_step__args_free",
  "is_method": False,
  "return_schema": {"kind": "bool", "cpp_type": "bool"},
  "params": [
    {"name": "p", "pass_as": "ref", "kind": "threadable", "schema": {… nested fields …}},
    {"name": "forces", "pass_as": "ref", "kind": "list", "schema": {…}},
    {"name": "dt", "pass_as": "value", "kind": "float", …},
    {"name": "buf", "pass_as": "tbuffer", "kind": "tbuffer",
     "schema": {"inner": {"kind": "threadable", "type_name": "Particle", …}}},
    …
  ],
  "types": {"Particle": Particle, "Vec2": Vec2, …},
  "schemas": {…},
}
```

Also stored in process-global `KERNELS["simulate_step"]` as a `KernelMeta` dataclass.

### 5.3 Declarations vs definitions

| Form | Where | What |
|------|-------|------|
| **Declaration** | `simulate_step.hpp` | Lifecycle trio (+ real kernel decl) |
| **Definition** | `simulate_step.cpp` | Kernel body, pack struct, lifecycle + **all accessors** |

Marshal finds accessors **by symbol name** via ctypes after `load_kernels` — they need not be listed in the header.

### 5.4 Pack layout (conceptual)

```cpp
struct simulate_step__args {
    Particle a0;
    std::vector<double> a1;
    double a2;
    cthreads::sync::tripple_buffer<Particle>* a3;  // tbuffer slot
    int a4;
    cthreads::linalg::Array<float> a5;             // or ref-shaped per emit
    bool ret;
};
```

Exact member types follow `emit_trampoline_cpp`. Important ideas:

- `pass_as: "ref"` Threadable → pack **owns** a value `Particle a0`; `__call` passes `a->a0` as `Particle&`.
- `pass_as: "tbuffer"` → pack holds a **pointer**; marshal sets it with `…__set_aN_ptr` from `TBufferHandle` / native ptr.
- Nested fields get accessors like `simulate_step__set_a0_pos_x`.
- Lists/dicts get resize / set_elem / insert helpers (see trampoline emission in `kernel_meta.py`).

### 5.5 Lifecycle definitions (pattern)

Same as the classic small example:

```cpp
CTHREADS_API void* simulate_step__args_new() { return new simulate_step__args(); }
CTHREADS_API void  simulate_step__args_free(void* p) {
    delete static_cast<simulate_step__args*>(p);
}
CTHREADS_API void  simulate_step__call(void* p) {
    auto* a = static_cast<simulate_step__args*>(p);
    a->ret = simulate_step(a->a0, a->a1, a->a2, *a->a3, a->a4, a->a5);
}
```

### 5.6 TBuffer runtime (extra files)

Because `TBuffer[Particle]` appears in a kernel, `write_tbuffer_runtime` emits allocator symbols used by `create_tbuffer` on the host:

```text
cthreads_create_tbuffer / destroy / generation / read_copy / free_read_copy
```

Switching on type name `"Particle"`, including `../__Threadable__/Particle.hpp` from `__Thread__/`.

---

## 6. Build and load the kernel library

### 6.1 `build()`

`build.py` walks **`REGISTRY.threadable_units` and `thread_units`** (hpp/cpp paths), plus tbuffer sources under `__Thread__/`, and compiles them into **one** shared library, e.g.:

```text
my_project/cthreads_kernels.dll
```

That library contains structs, free kernels, method wrappers, trampolines, and tbuffer helpers.

### 6.2 `load_kernels(path)`

`_ext` loads the shared library. From then on:

- C++ looks up `simulate_step__call`, …
- `cthreads.marshal` calls the same symbols through ctypes

`BINARY_PATH` / `kernel_path()` reflect the loaded library.

---

## 7. Runtime: `thread(simulate_step, …)` step by step

### 7.1 High-level Python entry (`prepare.py`)

```text
thread(fn, *args, force=False)
  loaded = kernel_path()
  if force and loaded: raise (unload first)
  if force or not loaded:
      binary = prepare(force=…)
      load_kernels(binary)
  return wrap_job(_ext.thread(fn, *args))
```

First call: prepare + load + spawn. Later calls: spawn only.

### 7.2 Native checks inside `_ext.thread`

1. `fn.__threaded` must be true.
2. `fn.__kernel_meta__` must exist.
3. Args bound against meta param names → ordered values.

### 7.3 `spawn_from_meta`

1. `pack = simulate_step__args_new()`
2. `marshal.pack_params(...)` fills the pack
3. Worker lambda: `__call` → `writeback_params` → `unpack_return` → `__args_free`
4. Return `Job` (not started until `start()` / `await`)

---

## 8. Packing: Python values → C++ pack

`cthreads.marshal.pack_params` walks each parameter schema and calls `…__set_…` trampolines.

| Index | prefix | Python value | kind |
|-------|--------|--------------|------|
| 0 | `a0` | `Particle` instance | threadable (nested pos, list, dict, lock, …) |
| 1 | `a1` | `forces` list | list[float] |
| 2 | `a2` | `0.1` | float |
| 3 | `a3` | `TBufferHandle` / buffer | tbuffer → set ptr |
| 4 | `a4` | `0` | int |
| 5 | `a5` | `ArrayF32` | internal array |

Threadable packing is **field by field** (and nested): `a0_pos_x`, `a0_velocity`, list resize/elems, dict inserts, etc. There is no single “memcpy the Python object.”

TBuffer packing:

```text
ptr = tbuffer_ptr(handle)   # sync.tbuffer_host
simulate_step__set_a3_ptr(pack, ptr)
```

After packing, the pack holds **copies** (or pointers into kernel-owned buffers). The Python objects are unchanged until writeback.

---

## 9. Running the kernel on an OS thread

### 9.1 Call

`job.start()` → OS thread → `simulate_step__call(pack)` → real `simulate_step(...)`.

Mutations hit the pack’s `Particle a0`, vector `a1`, etc.

### 9.2 Mid-run `__sync_state()`

When the kernel hits `cthreads::detail::__sync_state()`, the runtime (via `_ext` TLS) runs a writeback of the current pack into the **same** Python objects, then continues the kernel. That is how a host loop can observe `p.tags` grow before the job finishes (see integration test `test_kernel_sync_state_mid_run_writeback`).

### 9.3 Final writeback

After `__call` returns, worker (with GIL) runs `writeback_params` again for structured kinds: threadable / list / dict. Primitives and TBuffer pointers are not “copied back” as Python values the same way (TBuffer stays native; lists/dicts/threadables update in place).

### 9.4 Unpack return / free

```text
unpack_return → bool into Job result box
simulate_step__args_free(pack)
```

```python
job.join()
print(job.result())  # True/False
```

`await job` auto-starts and waits without blocking the asyncio event loop.

---

## 10. Full timeline (one page)

```text
demo.py import
  @Threadable Vec2, Particle  →  REGISTRY.threadables
  @Thread kick, simulate_step →  REGISTRY.threads

first thread(simulate_step, …)
  CompileSession.compile
    fill threadable_units / thread_units
    emit Vec2, Particle(+kick), simulate_step
    write_tbuffer_runtime (Particle)
    attach __kernel_meta__ / KERNELS
  build → cthreads_kernels shared library
  load_kernels

  spawn
  pack = simulate_step__args_new()
  marshal.pack_params (nested fields, list, tbuffer ptr, array, …)
  return Job

job.start()
  OS thread:
    simulate_step__call(pack)
      … may __sync_state() mid-run …
    writeback structured params
    result = ret bool
    args_free

job.join(); job.result()
Python Particle / lists updated in place
```

---

## 11. Mental model (keep this)

| Layer | What it is |
|-------|------------|
| Python `@Threadable` | Normal class; compile → C++ `struct` |
| Python `@Thread` | Normal function; compile → C++ kernel + pack API |
| `REGISTRY.threadables` / `threads` | Import-time registration |
| `REGISTRY.*_units` | Compile-time paths + field types + emit handles |
| `TranslationContext` | Scratch pad: symbols, includes, owner, `this_file` |
| `Syntax` + plugins | AST → C++ strings |
| `KernelMeta` / trampolines | Stable C ABI for marshal / `_ext` |
| `marshal` | Schema-driven ctypes calls to set/get symbols |
| `Job` | Native thread handle + result box |

One sentence:

**Compile drains the registry into units and C++ on disk; translation walks the AST with a name table and plugins; `thread()` packs values, runs the kernel off the GIL, writes structured results back, and returns a Job.**

---

## 12. Compile in depth (registry → units → files)

Code:

```text
frontend/Registry/registry.py
compiler/orchestrator/session/compileSession.py
compiler/orchestrator/units/{threadableUnit,threadUnit,baseUnit,handle}.py
cache.py
kernel_meta.py          # meta + trampolines + tbuffer runtime
prepare.py / build.py
```

### 12.1 Words

| Phrase | Meaning |
|--------|---------|
| **Registry** | Process-wide maps filled by decorators; units filled by `CompileSession` |
| **Unit** | `ThreadableUnit` or `ThreadUnit` — one emitable artifact |
| **Fingerprint / src_hash** | Hash of Python source; skip rewrite when unchanged |
| **Syntax tree** | Python `ast` nodes |

### 12.2 `compile()` is a drain, not a JIT of one call

`thread(simulate_step, …)` may *trigger* compile, but `compile()` emits **every** registered Threadable and Thread (methods claimed under their owner). One shared library contains every kernel.

Empty registry → `RuntimeError: Nothing registered to compile`.

### 12.3 Project root and folders

| Folder | Holds |
|--------|--------|
| `__Threadable__/` | One `.hpp`/`.cpp` per Threadable (methods included) |
| `__Thread__/` | Free Threads + `cthreads_export.hpp` + optional `cthreads_tbuffer.*` |

Method `@Thread`s are **not** separate free files; they live in the owner’s Threadable unit (`hpp_path=None` on the method `ThreadUnit`).

### 12.4 Order: structs before free functions

```text
1. create all ThreadableUnits (+ method ThreadUnits)
2. create free ThreadUnits
3. emit all Threadables
4. emit all Thread units (methods no-op or skip file emit; free functions write __Thread__)
5. tbuffer runtime
```

Threadables first so `include_for(PyThreadable("Particle"), …)` can resolve `hpp_path`.

### 12.5 One ThreadableUnit.emit

High level (`threadableUnit.py`):

1. Validate `@Threadable`.
2. Fingerprint class (+ methods). If unchanged and files exist and not `force` → skip rewrite; still refresh method kernel meta as needed.
3. Else: for each method, `translate_function(..., owner=self, this_file=hpp_path)`; build struct fields via `include_for`; `build_kernel_meta` + trampolines for methods; `write_if_changed` hpp/cpp.

### 12.6 One free ThreadUnit.emit

1. Paths under `__Thread__/`.
2. Ensure export header.
3. Fingerprint; maybe skip body rewrite.
4. `translate_function(fn, this_file=hpp_path, owner=None)`.
5. Write kernel + lifecycle decls to hpp; kernel + pack + trampoline defs to cpp.
6. Attach `__kernel_meta__`; register `KERNELS[symbol]`.

### 12.7 Cache

`.cthreads_cache.json` under project root: per-unit `src_hash` (+ link hash used by `build`).

| Situation | Behavior |
|-----------|----------|
| First compile | Write files, store hashes |
| Unchanged source | Skip rewrite; refresh meta |
| Edited body | Rewrite that unit |
| `force=True` | Rewrite everything |

### 12.8 What compile does **not** do

- Run the kernel
- Start an OS thread
- Pack Python values
- Load the DLL (`build` + `load_kernels`)

Contract: **registry → units + C++ text + metadata.**

---

## 13. Translation in depth (AST → C++ strings)

Translation does not know about DLLs or Jobs. Compile *calls* it; emit writes disks.

```text
compiler/translation/
  translate.py          translate_function
  Source.py             parse_function, resolve_annotation
  Signature.py
  Typeof.py
  context.py            TranslationContext
  result.py             TranslationResult
  include.py            add_include, include_for
  Cpp.py
  syntax/               Syntax.expr / Syntax.stmt
  plugins/              CallPlugin / AttrPlugin tables
    stdlib/             Math, Containers (len, list, dict)
    cthreads_math/
    sync/               Lock/Event/RWLock, TBuffer, __sync_state
    linalg/             methods, ctors, props
    fields/             FieldAttrPlugin (self.x → this->x)  # register last
```

### 13.1 `translate_function` pipeline

1. `Source.parse_function(fn)` → `FunctionDef`
2. `TranslationContext(fn, this_file, owner)`
3. `Signature.translate(func_def, ctx)` → fills `symbols`, `sig_includes`, `params_csv`, return type
4. For each stmt in body: `Syntax.stmt(stmt, ctx)`
5. Return `TranslationResult`

### 13.2 `TranslationContext` (scratch pad)

| Field | Role |
|-------|------|
| `fn` | Live function (`__globals__` for math/linalg lookups) |
| `this_file` | Generated `.hpp` path (relative includes) |
| `owner` | `ThreadableUnit` for methods; else `None` |
| `symbols` | Name → `PyType` |
| `sig_includes` / `body_includes` | `#include` lines |
| `seen_*` | Dedup includes |
| `func_name` | Property from `fn.__name__` |

Name table introductions:

| When | Inserted |
|------|----------|
| Signature params | Always |
| `x: int = …` | AnnAssign |
| `for i in range` / `for x in xs` | Loop var, removed after loop |
| Method | `self` → `PyThreadable(owner.name)` |

### 13.3 Python types → `PyType` → C++ (`types/`)

| You write | Internal | C++ |
|-----------|----------|-----|
| `int` / `float` / `bool` / `str` | PyInt / PyFloat / … | `int` / `double` / `bool` / `std::string` |
| `list[T]` / `dict[K,V]` | PyList / PyDict | `std::vector` / `std::unordered_map` |
| `@Threadable` | `PyThreadable("Name")` | `Name` (+ unit hpp include) |
| `Lock` / `ArrayF32` / … | `PyCThreadsInternalType` | catalog C++ type + header |
| `TBuffer[Particle]` | `PyTBuffer` | `tripple_buffer<Particle>` |

### 13.4 `Typeof.of` (receiver typing for methods)

```text
Name        → symbols[name]
Attribute   → if base is PyThreadable: unit.fields[attr]
Subscript   → if base is PyList: inner type
```

Field types come from **`REGISTRY.threadable_units[name].fields`** (filled at compile), not a second handwritten map.

Printing `this->a` is separate (`FieldAttrPlugin`). Common bug: path prints but type is unknown → call plugins miss → `unsupported call`.

### 13.5 Statement / expression catalog (supported)

Statements: AnnAssign, Assign, AugAssign, Return, Pass, If, While, For (`range` or list), Break, Continue, Expr (calls / docstring).

Expressions: Constant, Name, Attribute (plugins), BinOp/UnaryOp/BoolOp/Compare, Call (plugins), Subscript (no Python slice sugar for lists yet), List display.

Rejected by design: `*args`/`**kwargs`, unannotated new locals, `None` literal, arbitrary Python calls, `for`/`else`, `while`/`else`.

### 13.6 Call / attribute plugin order

`Syntax.expr` on `Call` / `Attribute` runs `plugins.lower_call` / `lower_attr`.

Registration order (`plugins/__init__.py`):

```text
stdlib (len, list/dict, math call + math const)
cthreads_math
sync (methods, TBuffer methods, __sync_state)
linalg (methods, ctors, props)
fields Attr  ← last (self.x / fallback (base).attr)
```

First plugin that returns a string wins. A plugin that recognizes the receiver but hates the method **raises** (does not fall through).

Examples:

| Python | C++ |
|--------|-----|
| `xs.append(v)` | `(xs).push_back(v)` |
| `d.get(k, default)` | find/ternary lambda |
| `lock.acquire()` | `(lock).acquire()` |
| `math.sqrt(x)` | `std::sqrt(x)` |
| `a.matmul(b)` | `(a).matmul(b)` |
| `a.shape` | `(a).shape()` |
| `ArrayF32(sh)` | `cthreads::linalg::Array<float>(sh)` |
| `__sync_state()` | `cthreads::detail::__sync_state()` |

### 13.7 Includes

`include_for(py_type, this_file)`:

- primitives → often empty / STL
- Threadable → `os.path.relpath(unit.hpp_path, this_file.parent)`
- internal → quoted `"linalg/array.hpp"` etc. + extra STL

Signature includes → `.hpp`; body-only → `.cpp` after `#include "simulate_step.hpp"`.

### 13.8 What translation returns

`TranslationResult`: `return_type`, `func_name`, `params_csv`, `sig_includes`, `body`, `body_includes`, plus helpers `method_decl()` / `method_def_signature(owner_name)` for Threadable emit.

Translation never writes to disk.

### 13.9 Intentional limits

- Only dispatched AST kinds + plugins
- Host-only Array helpers (`from_list`, …) are not kernel ops — build on host, pass in
- Method tables are the **kernel** API; do not mirror every Python host method blindly
- `Shape` C++ ctor takes `vector<size_t>` (and `vector<int>` for codegen); keep that in mind if you change list-literal lowering

---

## 14. How a contributor should extend this

### 14.1 New statement or expression syntax

1. Add logic under `syntax/` (keep core logic together; split only if shared).
2. Register on `Syntax` dispatch tables.
3. Unit tests: `tests/unit/test_Ast_*.py` with `helpers.make_ctx` / `parse_expr`.

### 14.2 New method on an existing typed object

Add a `MethodOp` row to the right `MethodTablePlugin.tables` (Containers / Sync / linalg / TBuffer).  
Do **not** teach `Typeof` about the method.

### 14.3 New native `cthreads.*` type

1. Mark pybind type `__cthreads_internal__ = True`.
2. Add `CTHREADS_INTERNAL_TYPES` row.
3. Add plugin (call/attr) and `register_*`.
4. Translation tests can use a dummy class with the flag.

### 14.4 New Threadable field type

If `is_threadable` + `hint_to_pytype` accept it, units’ `fields` and `Typeof` pick it up automatically.

### 14.5 Tests map

| Kind | File |
|------|------|
| AST expr/stmt | `test_Ast_exprs.py`, `test_Ast_stmts.py` |
| Signature | `test_signature_translate.py` |
| Containers / math / sync / linalg | `test_*_ops.py` |
| Typeof | `test_typeof.py` |
| Kernel meta / tbuffer runtime | `test_kernel_meta.py`, `test_tbuffer_runtime.py` |
| Marshal TBuffer | `test_marshal_tbuffer.py` |
| End-to-end | `tests/integration/test_compile_api.py` |

### 14.6 Mental picture

```text
@Thread / @Threadable
        │
        ▼
    REGISTRY.threadables / threads          (import time)
        │
        ▼
    CompileSession.compile()
        │
        ├── ThreadableUnit / ThreadUnit     (paths, fields, params)
        ├── unit.emit()
        │     ├── translate_function
        │     │     ├── Signature → symbols
        │     │     └── Syntax + plugins → body
        │     ├── kernel_meta trampolines
        │     └── write .hpp/.cpp
        └── write_tbuffer_runtime
        │
        ▼
    build() / load_kernels() / marshal / Job
```

One sentence for this half:

**Compile fills units on the registry and emits files; translation walks the AST with a name table and ordered plugins; `Typeof` types dotted receivers from unit fields; trampolines and `thread()` are a later layer — do not put packing logic inside a new expression plugin.**

---

## 15. Quick reference — public API

```python
from cthreads import (
    Thread, Threadable,
    compile, prepare, thread, build,
    Job, sync_state, __sync_state,
    load_kernels, unload_kernels, spawn,
    sync, TBuffer, create_tbuffer,
    REGISTRY, KERNELS, VERSION, BINARY_PATH,
)
# math / linalg from _ext when built:
from cthreads import math, linalg
```

Force rebuild safely:

```python
unload_kernels()
prepare(force=True)
# or: thread(fn, *args, force=True)  # only if nothing loaded
```
