# Mid-run state sync (`__sync_state` / `sync_state`)

This document explains how cthreads mirrors C++ pack state back into Python
**while a job is still running**. Read it when you have forgotten why a bridge,
TLS, and by-ref mutables exist.

Related APIs:

| API | Where | Role |
|-----|--------|------|
| `__sync_state()` | Inside `@Thread` bodies only | Kernel barrier -> writeback via `_ext` |
| `job.sync_state()` | Host Python | Same writeback, driven from outside the kernel |
| `cthreads.sync_state(job)` | Host Python | Alias for `job.sync_state()` |

There is no `sync_thread` API; the host entry point is **`sync_state`**.

---

## 1. Mental model (start here)

cthreads does **not** let the kernel poke live Python objects while it runs off the GIL.

```text
Python args  ──pack (copy)──►  C++ args struct ("the pack")
                                      │
                              kernel mutates pack
                                      │
                    __sync_state / sync_state / final join
                                      │
                                      ▼
                         writeback: pack ──copy──► Python args
```

Important consequences:

1. **Two worlds.** During the job, Python and C++ are separate. Sync copies
   C++ -> Python; it does not share memory with live Python objects mid-run.
2. **Writeback targets job args only.** Sync updates the Python objects that
   were passed into `spawn` / `thread`, not parent containers, slices’ parents,
   or other aliases.
3. **Full snapshot, not a diff.** Each writeback pushes the whole mutable arg
   (list / dict / Threadable) from that job’s pack. Last writer wins if two jobs
   share one Python list object.
4. **Mutables must bind the pack by reference** in the generated C++ signature.
   Otherwise the kernel edits a throwaway copy and sync copies stale pack data
   back (frozen UI). See §7.

---

## 2. Two DLLs (why a “bridge” exists)

| DLL | What it is |
|-----|------------|
| `cthreads._ext` (`.pyd`) | Installed extension: jobs, `LoadLibrary`, GIL, marshal writeback |
| `cthreads_kernels.dll` | User-built codegen from `@Thread` / `@Threadable` |

Kernels are **not** linked against `_ext`. `_ext` loads them with `LoadLibrary`
and looks up symbols with `GetProcAddress` / `dlsym`.

So a kernel cannot “just call” `_ext` C++ by normal linking. The sync bridge
installs a **function pointer** from `_ext` into the kernel DLL at
`load_kernels` time.

```text
┌──────────────────────────┐         ┌─────────────────────────────┐
│  _ext.pyd                │         │  cthreads_kernels.dll       │
│                          │         │                             │
│  g_job (thread_local)    │         │  g_ext_sync_state (fn ptr)  │
│  cthreads_ext_sync_state │◄──ptr───│  __sync_state() dials it    │
│  job_do_writeback        │         │  cthreads_bind_sync_state   │
│  marshal.writeback…      │         │                             │
└──────────────────────────┘         └─────────────────────────────┘
```

---

## 3. Public Python usage

### 3.1 Kernel barrier: `__sync_state()`

```python
from cthreads import Thread, __sync_state, spawn, sync_state

@Thread
def flock_continuous(boids: list[Boid], dt: float) -> None:
    # ... mutate boids in the pack ...
    __sync_state()   # copy pack -> Python args so the host can read mid-run
```

Line-by-line:

| Line | Meaning |
|------|---------|
| `from cthreads import __sync_state` | Importable stub so the name resolves at edit/compile time. |
| `@Thread` | Function is compiled to C++; body is not executed as normal Python. |
| `__sync_state()` | **Not** a real Python call at runtime inside the job. Codegen lowers it to `cthreads::detail::__sync_state();`. |

Calling `__sync_state()` from ordinary Python (outside `@Thread`) raises:

```python
# cthreads/__init__.py - stub only
def __sync_state() -> None:
    raise RuntimeError(
        "cthreads.__sync_state() is only valid inside @Thread bodies "
        "(it is compiled to cthreads::detail::__sync_state())"
    )
```

| Line | Meaning |
|------|---------|
| `def __sync_state()` | Exists so `from cthreads import __sync_state` and AST name lookup work. |
| `raise RuntimeError(...)` | Guards misuse: this function must never run as Python mid-job. |

### 3.2 Host API: `job.sync_state()` / `cthreads.sync_state(job)`

```python
job = spawn(flock_continuous, state.boids, dt)
job.start()
while not job.done():
    sync_state(job)          # or job.sync_state()
    # read state.boids - same list object passed to spawn
    draw(state.boids)
job.join()
```

| Line | Meaning |
|------|---------|
| `spawn(..., state.boids, dt)` | Packs a **copy** of `state.boids` into C++. Keeps the Python list as the writeback target. |
| `job.start()` | Worker runs; must start before host sync. |
| `sync_state(job)` | Steals the job mutex, takes GIL, writebacks pack -> args. Does **not** use kernel TLS. |
| `draw(state.boids)` | Sees updates only if pack was actually mutated (by-ref) and writeback ran. |
| `job.join()` | Final writeback also runs when the kernel finishes (even without mid-run sync). |

#### `Job.sync_state` (Python wrapper)

```python
# job.py
def sync_state(self) -> None:
    if not self._started:
        raise RuntimeError(
            "cthreads.Job.sync_state: job not started "
            "(call start() or await / join first)"
        )
    sync = getattr(self._raw, "sync_state", None)
    if sync is None:
        raise RuntimeError(
            "cthreads.Job.sync_state: native Job has no sync_state "
            "(rebuild the cthreads extension)"
        )
    sync()
```

| Line | Meaning |
|------|---------|
| `if not self._started` | Host sync needs a running/started job; pack must exist. |
| `getattr(self._raw, "sync_state", None)` | Native `_ext.Job.sync_state`; older wheels may lack it. |
| `sync()` | Enters C++ `SpawnedKernel::sync_state()` (see §6). |

#### Module-level helper

```python
def sync_state(job: Job) -> None:
    if not isinstance(job, Job):
        raise TypeError(...)
    job.sync_state()
```

| Line | Meaning |
|------|---------|
| `isinstance(job, Job)` | Only cthreads jobs; not raw threads or arbitrary objects. |
| `job.sync_state()` | Same path as the method. |

---

## 4. Codegen: Python call -> C++ call

Detection lives in `AstTranslators/call.py` (builtin, like `len`):

```python
if is_builtin_call(node, "__sync_state"):
    if node.keywords or node.args:
        raise TypeError("... __sync_state() takes no arguments")
    add_include(ctx.body_includes, ctx.seen_body, '#include "sync/syncState.hpp"\n')
    return "cthreads::detail::__sync_state()"
```

| Line | Meaning |
|------|---------|
| `is_builtin_call(..., "__sync_state")` | Match bare `__sync_state()` (also listed in `pyOps.BUILTINS`). |
| arity check | No args / kwargs. |
| `#include "sync/syncState.hpp"` | Declares `cthreads::detail::__sync_state()`. |
| return `"cthreads::detail::__sync_state()"` | Emit that expression as a statement in the kernel body. |

**Why not emit “full writeback” here?**  
Writeback needs the job’s pack, live Python arg lists, GIL, and `cthreads.marshal`. Those live in `_ext`, not in the generated kernel TU. Emitting full logic would either duplicate `_ext` into every kernel or just move the problem to binding many data handlers instead of one doorbell function. The translator correctly emits a **call**; the bridge + `_ext` implement the work.

---

## 5. Kernel bridge (function pointer)

File: `src/cthreads/cpp/runtime/sync_bridge.cpp`  
Always linked into `cthreads_kernels.dll` by `build.py` (`_collect_sources_and_includes`).

```cpp
namespace {
void (*g_ext_sync_state)() = nullptr;   // (1)
}

CTHREADS_BRIDGE_API void cthreads_bind_sync_state(void (*fn)()) {  // (2)
    g_ext_sync_state = fn;
}

namespace cthreads::detail {
void __sync_state() {                   // (3)
    if (g_ext_sync_state) {
        g_ext_sync_state();
    }
}
}
```

| Piece | Meaning |
|-------|---------|
| `(1) g_ext_sync_state` | Process-wide (per kernel DLL) pointer to `_ext`’s sync entry. Starts null. |
| `(2) cthreads_bind_sync_state` | `extern "C"` export so `_ext` can find it with `GetProcAddress` and install the pointer. |
| `(3) __sync_state` | What codegen calls: dial `_ext` if bound, else no-op. |

Concurrent jobs: every worker dials the **same** function address. Which job is current is decided by **TLS inside `_ext`**, not by this pointer.

Header `sync/syncState.hpp` only declares `__sync_state()` and defines `JobContext`. It must **not** contain `inline thread_local` - that was the dual-fridge bug (each DLL got its own TLS slot).

---

## 6. `_ext` side: bind, TLS, writeback

### 6.1 Bind on `load_kernels`

```cpp
void bind_sync_state_to_kernels() {
    if (!cthreads::kernels().loaded()) return;
    void* p = cthreads::kernels().sym("cthreads_bind_sync_state");
    if (!p) return;  // old kernel DLL without bridge
    using BindFn = void (*)(void (*)());
    reinterpret_cast<BindFn>(p)(&cthreads_ext_sync_state);
}

// pybind:
m.def("load_kernels", [](const std::string& path) {
    cthreads::load_kernels(path);
    bind_sync_state_to_kernels();
});
```

| Line | Meaning |
|------|---------|
| `sym("cthreads_bind_sync_state")` | Lookup export in the loaded kernel DLL. |
| `(&cthreads_ext_sync_state)` | Pass address of `_ext`’s entry into the kernel’s global. |
| call after `load_kernels` | Every fresh LoadLibrary re-binds. |

### 6.2 Per-thread job sticky note (TLS)

```cpp
thread_local cthreads::detail::JobContext* g_job = nullptr;

void set_job_context(JobContext* ctx) { g_job = ctx; }

void cthreads_ext_sync_state() {
    JobContext* ctx = g_job;
    if (!ctx || !ctx->do_writeback || !ctx->pack) return;
    ctx->do_writeback(ctx);
}
```

| Line | Meaning |
|------|---------|
| `thread_local g_job` | **Only in `_ext`.** Each OS worker thread has its own pointer. |
| `cthreads_ext_sync_state` | Target of the kernel’s function pointer. |
| `ctx->do_writeback(ctx)` | Usually `&job_do_writeback`. |

`JobContext` fields (opaque void* so kernels need no pybind):

| Field | Points at |
|-------|-----------|
| `state_mu` | Job mutex (host sync vs kernel sync) |
| `pack` | C++ args struct |
| `symbol` / `params` / `values` / `types` / `schemas` | Marshal inputs |
| `do_writeback` | C++ function that performs locked GIL writeback |

### 6.3 Worker installs context around the kernel call

```cpp
auto job = [call_fn, self, params_keep_for_job]() mutable {
    JobContext ctx{};
    ctx.state_mu = &self->state_mu;
    ctx.pack = self->pack;
    ctx.symbol = &self->symbol;
    ctx.params = params_keep_for_job.get();
    ctx.values = self->values_keep.get();
    ctx.types = self->types_keep.get();
    ctx.schemas = self->schemas_keep.get();
    ctx.do_writeback = &job_do_writeback;

    set_job_context(&ctx);
    try {
        call_fn(self->pack);
    } catch (...) {
        set_job_context(nullptr);
        throw;
    }
    set_job_context(nullptr);
    // then final writeback + free pack ...
};
```

| Line | Meaning |
|------|---------|
| Fill `ctx` | Sticky note contents for this job. |
| `set_job_context(&ctx)` | Pin note on **this** worker thread before entering the kernel DLL. |
| `call_fn(self->pack)` | Runs generated code; may call `__sync_state()`. |
| clear context | After return, mid-run sync must not see a dangling stack `ctx`. |

### 6.4 `job_do_writeback` (shared core)

```cpp
void job_do_writeback(JobContext* ctx) {
    std::lock_guard<std::mutex> g(*ctx->state_mu);  // 1) mutex first
    py::gil_scoped_acquire gil;                     // 2) then GIL
    writeback_params(
        *symbol, *params, *values, ctx->pack, *types, *schemas
    );
}
```

| Step | Meaning |
|------|---------|
| Lock `state_mu` | Serialize with host `sync_state` and final writeback. |
| Acquire GIL | Need Python for marshal. |
| `writeback_params` | Imports `cthreads.marshal` and copies pack -> Python args. |

Lock order is always **mutex then GIL** (never reverse) to avoid deadlock.

### 6.5 Host `SpawnedKernel::sync_state`

```cpp
void SpawnedKernel::sync_state() {
    if (finished || !pack) return;
    std::lock_guard<std::mutex> g(state_mu);
    py::gil_scoped_acquire gil;
    if (finished || !pack) return;
    py::list params = (*meta_keep)["params"].cast<py::list>();
    writeback_params(symbol, params, *values_keep, pack, *types_keep, *schemas_keep);
}
```

| Line | Meaning |
|------|---------|
| `finished \|\| !pack` | No-op after final writeback freed the pack. |
| Same mutex + GIL | Compatible with kernel mid-run sync. |
| Uses Job fields directly | **Does not** read `g_job` / bridge. |

Same end effect as kernel `__sync_state()`; different entry door.

### 6.6 Full call chain (kernel path)

```text
[kernel]  cthreads::detail::__sync_state()
              │
              │  g_ext_sync_state()          // fn ptr -> _ext
              ▼
[_ext]    cthreads_ext_sync_state()
              │  g_job (TLS) -> do_writeback
              ▼
[_ext]    job_do_writeback(ctx)
              │  mutex + GIL
              ▼
[_ext]    writeback_params(...)
              │
              ▼
[Python]  cthreads.marshal.writeback_params(symbol, params, values, pack, ...)
```

---

## 7. Prerequisite: mutables by reference (pack must actually change)

Sync only copies **whatever is in the pack**. If the kernel received lists/dicts **by value**, it mutated a temporary and the pack stayed stale - sync correctly copied garbage/old data.

Codegen now emits references for mutables:

**Signature** (`signature.py`):

```python
if isinstance(py_type, (PyThreadable, PyList, PyDict)):
    params.append(f"{py_type.cpp_name}& {arg.arg}")  # e.g. std::vector<Boid>& boids
else:
    params.append(f"{py_type.cpp_name} {arg.arg}")   # int, double, ...
```

**Kernel meta** (`kernel_meta.py`): `pass_as="ref"` for the same types so the trampoline binds pack slots correctly.

| Type | Pass mode |
|------|-----------|
| `list` / `dict` / `@Threadable` | by ref (`&`) into the pack |
| `int` / `float` / `bool` / `str` | by value |

This is separate from sync, but **without it mid-run sync looks broken**.

---

## 8. Semantics cheat sheet (do not “fix” these)

| Situation | Expected behavior |
|-----------|-------------------|
| `spawn(fn, state.boids)` + sync | Updates `state.boids` (the arg object). |
| `owned = state.boids[a:b]` then `spawn(fn, owned)` + sync | Updates **`owned` only**, not `state.boids`. |
| Two jobs, same Python list, each mutates a range in its **private pack** | Each writeback replaces the **whole** list from that pack; last finish wins - not a merge. |
| Host holds `b = state.boids[0]` across sync | List writeback may rebuild elements; `b` can be a stale instance. Prefer reading through the arg list after sync. |

Design intent: **arg-rooted full writeback** is simple and predictable. Shared-memory / ranged merge is a future feature, not what sync does today.

---

## 9. Timeline (one job)

```text
prepare / build
  └─ compile sync_bridge.cpp into cthreads_kernels.dll

load_kernels(path)
  └─ LoadLibrary
  └─ cthreads_bind_sync_state(&cthreads_ext_sync_state)

spawn / thread(fn, args...)
  └─ pack Python -> C++ pack
  └─ keep Python arg objects on the Job

worker start
  └─ JobContext on stack; set_job_context(&ctx)
  └─ call_fn(pack)
        └─ ... kernel work ...
        └─ __sync_state()  -> bridge -> _ext TLS -> writeback
        └─ ... more work ...
  └─ clear TLS
  └─ final writeback + free pack + finished = true

host (any time while running)
  └─ job.sync_state() -> mutex + GIL + writeback (no TLS)
```

---

## 10. Why not “just link `_ext`” or “emit full sync in the translator”?

**Link kernels to `_ext`:** possible in theory, painful in practice (`.pyd` as import lib, venv paths, toolchains). TLS still must live in one DLL only.

**Translator emits full writeback:** needs Python/GIL/job metadata inside every kernel build -> mini-`_ext` per project, or binding many data handlers instead of one function pointer.

**Best fit for this architecture:** thin codegen call + `sync_bridge` doorbell + `_ext` owns TLS and writeback.

---

## 11. Debugging checklist

| Symptom | Likely cause |
|---------|----------------|
| UI / Python never moves; final join also stale | Pack not mutated -> check by-ref signatures / rebuild kernels |
| Host `sync_state` updates, in-kernel `__sync_state` does nothing | Bridge not bound (old DLL) or `load_kernels` skipped |
| `_debug_ext_sync_invocations` stays 0 | Kernel never entered `cthreads_ext_sync_state` (bind / emit / call path) |
| `LoadLibrary` error 4551 on Windows | Smart App Control blocking unsigned DLL (often under `%TEMP%`) |
| Sync updates slice list but not `state.boids` | You passed the slice; writeback is arg-rooted (expected) |

Test helper (extension): `_ext._debug_ext_sync_invocations` / `_debug_reset_ext_sync_invocations` count entries into `cthreads_ext_sync_state`.

---

## 12. File map

| Path | Role |
|------|------|
| `src/cthreads/python/cthreads/__init__.py` | `__sync_state` stub; re-exports `sync_state` |
| `src/cthreads/python/cthreads/job.py` | `Job.sync_state`, `sync_state(job)` |
| `src/cthreads/python/cthreads/pyOps.py` | `__sync_state` in `BUILTINS` |
| `.../AstTranslators/call.py` | Lowers `__sync_state()` -> C++ call |
| `.../AstTranslators/signature.py` | Mutable params as `T&` |
| `.../kernel_meta.py` | `pass_as="ref"` for mutables |
| `.../build.py` | Always compiles `sync_bridge.cpp` into kernel DLL |
| `src/cthreads/cpp/headers/sync/syncState.hpp` | `JobContext` + `__sync_state` declaration |
| `src/cthreads/cpp/runtime/sync_bridge.cpp` | Bind export + kernel `__sync_state` |
| `src/cthreads/cpp/bindings/module.cpp` | TLS, bind, `job_do_writeback`, host `Job.sync_state` |
| `src/cthreads/python/cthreads/marshal.py` | Actual pack ↔ Python field copy |

---

## 13. Minimal examples

### Kernel-only barrier (host polls Python args)

```python
from cthreads import Thread, __sync_state, spawn

@Thread
def pulse(xs: list[int]) -> None:
    xs.append(1)
    __sync_state()          # Python xs should become [1] here
    # ... more work ...
    xs.append(2)            # final join writeback -> [1, 2]

xs: list[int] = []
job = spawn(pulse, xs)
job.start()
# optionally poll xs while running; do not need sync_state(job) if kernel barriers
job.join()
assert xs == [1, 2]
```

### Host-driven sync (no in-kernel barrier required)

```python
from cthreads import Thread, spawn, sync_state

@Thread
def long_run(xs: list[int]) -> None:
    i: int = 0
    while i < 1000000:
        xs.append(i)
        i = i + 1

xs: list[int] = []
job = spawn(long_run, xs).start()
while not job.done():
    sync_state(job)
    print(len(xs))
job.join()
```

Both paths require list/dict/Threadable **by-ref** into the pack so mutations survive until writeback.
