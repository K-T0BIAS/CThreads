# End-to-end example: one `@Threadable` and one `@Thread`

This document walks through **one small program** from Python source to generated C++, then through packing, running the kernel, and writing results back.

Everything below uses the **same** example. Later sections only add detail; they do not change the story.

**If you want to help on the compiler itself**, start with §0–§4 so the example is in your head, then jump to **§12 (compile)** and **§13 (translation)**. Those two sections name every moving part of “Python function in, C++ files out.”

---

## 0. The Python program

Assume this file lives at:

```text
my_project/demo.py
```

```python
from cthreads import Thread, Threadable, thread

@Threadable
class Particle:
    x: float
    y: float

@Thread
def move(p: Particle, dt: float) -> None:
    p.x = p.x + dt
    p.y = p.y + dt

# build a Python Particle and run move off the GIL
p = Particle(1.0, 2.0)

job = thread(move, p, 0.1)
job.start()
job.join()

print(p.x, p.y)   # 1.1  2.1
print(job.result())  # None  (return type is -> None)
```

What this program *means* in plain language:

1. `Particle` is a small data bag with two floats.
2. `move` updates those floats by adding `dt`.
3. `thread(move, p, 0.1)` does **not** run the Python body of `move`.
4. It runs a **compiled C++ copy** of `move` on a real OS thread, then copies the updated `x`/`y` back into the Python object `p`.

The rest of this document shows every intermediate step that makes that true.

---

## 1. What happens when Python loads `demo.py`

Importing and decorating does **not** compile C++ yet. It only **marks and registers** things.

### 1.1 `@Threadable` on `Particle`

The decorator `Threadable` runs on the class right after the class body is defined.

It does these things, in order:

1. Sets `Particle.__threadable = True`.
2. Sets `Particle.__threadable_version` to the current cthreads version string.
3. Reads the field annotations (`x: float`, `y: float`) and checks that each type is allowed (`float` is allowed).
4. Registers the class in a global registry:

   ```text
   REGISTRY.threadables["Particle"] = Particle
   ```

5. Returns the **same** class object (no new `__init__` is invented).

So after decoration, `Particle` is still a normal Python class. You create instances with the generated dataclass-style constructor (`Particle(1.0, 2.0)`). The decorator’s job is: “remember this class so compile can turn it into a C++ `struct` later.”

### 1.2 `@Thread` on `move`

The decorator `Thread` runs on the function.

It does these things, in order:

1. Sets `move.__threaded = True`.
2. Sets `move.__thread_version` to the current cthreads version string.
3. Reads type hints (`p: Particle`, `dt: float`, `-> None`) and checks that each argument type is allowed. `None` as a return type is allowed and means “void” in C++.
4. Registers the function:

   ```text
   REGISTRY.threads["move"] = move
   ```

   (The real key is the function’s `__qualname__`, which for a free function is `"move"`.)

5. Returns the **same** function. Calling `move(p, 0.1)` as normal Python still runs the Python body. Only `thread(move, ...)` uses the compiled path.

### 1.3 State after import (before any compile)

| Name | Extra flags / registry entry |
|------|------------------------------|
| `Particle` | `__threadable=True`, in `REGISTRY.threadables` |
| `move` | `__threaded=True`, in `REGISTRY.threads` |
| C++ files | **none yet** |
| Kernel DLL | **none yet** |
| `move.__kernel_meta__` | **missing until compile** |

---

## 2. When compile runs

Compile usually happens on the first `thread(...)` call if no kernel library is loaded yet:

```text
thread(move, p, 0.1)
  -> prepare()
      -> compile()   # write .hpp / .cpp next to demo.py
      -> build()     # compile those into one shared library
  -> load_kernels(path_to_shared_library)
  -> spawn the job
```

`compile()` drains the registry in a fixed order:

1. **All** `@Threadable` classes first (so structs exist before functions that use them).
2. Then **free** `@Thread` functions that are not methods of a Threadable.

For our example that means:

```text
1. compile Particle  ->  my_project/__Threadable__/Particle.hpp
                         my_project/__Threadable__/Particle.cpp

2. compile move      ->  my_project/__Thread__/move.hpp
                         my_project/__Thread__/move.cpp
```

It also writes a small shared header used for DLL export macros:

```text
my_project/__Thread__/cthreads_export.hpp
```

Folder layout after compile:

```text
my_project/
  demo.py
  __Threadable__/
    Particle.hpp
    Particle.cpp
  __Thread__/
    cthreads_export.hpp
    move.hpp
    move.cpp
```

What `compile()` does internally (registry order, cache, which Python files run) is spelled out in **§12**. The next two sections stay with the example: first `Particle`, then `move`.

---

## 3. Compiling `Particle` (the Threadable)

### 3.1 Inputs

- Class name: `Particle`
- Fields from annotations:
  - `x` -> Python `float`
  - `y` -> Python `float`
- Methods marked `@Thread`: **none** in this example

### 3.2 Type mapping for fields

Each field annotation is turned into a small internal type object, then into a C++ declaration.

| Python annotation | Internal mapping | C++ field declaration |
|-------------------|------------------|------------------------|
| `x: float` | float -> C++ `double` | `double x;` |
| `y: float` | float -> C++ `double` | `double y;` |

No extra `#include` is needed for `double`.

### 3.3 Generated `Particle.hpp` (exact shape for this example)

```cpp
#pragma once

#include "../__Thread__/cthreads_export.hpp"

struct Particle {
    double x;
    double y;
};
```

Notes:

- The C++ type is named **`Particle`**, same as the Python class.
- Fields are public struct members.
- Because there are no `@Thread` methods on the class, the header has **no** method declarations and **no** trampoline declarations.

### 3.4 Generated `Particle.cpp`

```cpp
#include "Particle.hpp"
```

With no methods, the `.cpp` only includes the header.

### 3.5 What compile records

A global store remembers where the header lives, roughly:

```text
STORE["Particle"] = ".../my_project/__Threadable__/Particle.hpp"
```

Later, when `move` is compiled, that path (or the fixed relative form) is how `move.hpp` knows to `#include` the Particle header.

---

## 4. Compiling `move` (the free Thread function)

This is the larger step. It has three big pieces:

1. Read the Python function’s AST (syntax tree).
2. Translate signature + body into a real C++ function `move(...)`.
3. Emit **trampoline** helpers (`move__args_new`, setters, `move__call`, …) so Python can fill an args pack and call the kernel through a stable C API.

### 4.1 Load source and get the function AST

Compile asks Python for the source of `move`, dedents it, and parses it.

Roughly, the AST for the function is:

```text
FunctionDef(
  name='move',
  args=arguments(
    args=[
      arg(arg='p',   annotation=Name(id='Particle')),
      arg(arg='dt',  annotation=Name(id='float')),
    ]
  ),
  returns=Constant(value=None),   # from -> None
  body=[
    Assign(
      targets=[Attribute(value=Name(id='p'), attr='x')],
      value=BinOp(
        left=Attribute(value=Name(id='p'), attr='x'),
        op=Add(),
        right=Name(id='dt'),
      ),
    ),
    Assign(
      targets=[Attribute(value=Name(id='p'), attr='y')],
      value=BinOp(
        left=Attribute(value=Name(id='p'), attr='y'),
        op=Add(),
        right=Name(id='dt'),
      ),
    ),
  ],
)
```

That tree is what translators walk. Nothing “magical” is left in the source text after this; only nodes.

### 4.2 Translate the signature

Hints for `move`:

| Hint name | Python type | Role |
|-----------|-------------|------|
| `p` | `Particle` (Threadable class) | first parameter |
| `dt` | `float` | second parameter |
| `return` | `None` | means C++ `void` |

Rule used here:

- A `@Threadable` parameter becomes a **C++ reference**: `Particle& p`.
- A plain `float` becomes a **by-value** `double`: `double dt`.
- `-> None` becomes return type `void`.

Also, because `Particle` is a Threadable, the signature translator adds an include for its header:

```cpp
#include "../__Threadable__/Particle.hpp"
```

Intermediate signature parts:

| Part | Value |
|------|--------|
| return type | `void` |
| function name | `move` |
| params inside `(...)` | `Particle& p, double dt` |

Full free-function signature string:

```cpp
CTHREADS_API void move(Particle& p, double dt)
```

`CTHREADS_API` expands (via `cthreads_export.hpp`) to something like `extern "C" __declspec(dllexport)` on Windows, or `extern "C"` elsewhere, so the symbol can be found in the shared library.

### 4.3 Translate the body, statement by statement

#### Statement 1: `p.x = p.x + dt`

AST: `Assign(target=p.x, value=(p.x + dt))`

Walk of the right-hand side (`BinOp`):

1. Left: `Attribute(p, x)`
   - base name `p` -> `"p"`
   - attribute -> `"p.x"`
2. Right: name `dt` -> `"dt"`
3. Operator `+` -> wrap as `"(p.x + dt)"`

Walk of the left-hand side:

1. `Attribute(p, x)` -> `"p.x"`

Assign joins them:

```cpp
    p.x = (p.x + dt);
```

(The extra parentheses around every binary expression are intentional in the translator.)

#### Statement 2: `p.y = p.y + dt`

Same pattern:

```cpp
    p.y = (p.y + dt);
```

#### Full kernel body

```cpp
    p.x = (p.x + dt);
    p.y = (p.y + dt);
```

### 4.4 The real C++ kernel function

Putting signature + body together:

```cpp
CTHREADS_API void move(Particle& p, double dt) {
    p.x = (p.x + dt);
    p.y = (p.y + dt);
}
```

This is the function that will eventually run on the OS thread. It only knows about C++ `Particle` and `double`. It never sees Python objects.

That walk is the whole translation idea in miniature. **§13** repeats it slowly: every kind of Python syntax the compiler accepts, how types are remembered, how `p.x` is different from `xs.append(v)`, and where a new contributor would plug in.

---

## 5. Kernel meta and trampolines for `move`

The kernel above alone is not enough. Python still has a Python `Particle` instance and a Python float `0.1`. Something must:

1. Allocate a C++ “bag of arguments” (the **pack**).
2. Copy Python values into that pack.
3. Call `move(...)` using values from the pack.
4. Copy mutated fields back into the Python `Particle`.
5. Free the pack.

Those helpers are generated next to the kernel. Their names are fixed from the kernel **symbol**, which for a free function is the function name: `"move"`.

### 5.1 Symbol table for this example

| Role | Symbol name |
|------|-------------|
| Real kernel | `move` |
| Allocate pack | `move__args_new` |
| Free pack | `move__args_free` |
| Call kernel from pack | `move__call` |
| Pack struct type | `move__args` |
| Param 0 slot in pack | `a0` (holds a `Particle`) |
| Param 1 slot in pack | `a1` (holds a `double`) |
| Return slot | **none** (return is void) |

Setter / getter names for fields:

| Purpose | Symbol |
|---------|--------|
| set `a0.x` | `move__set_a0_x` |
| get `a0.x` | `move__get_a0_x` |
| set `a0.y` | `move__set_a0_y` |
| get `a0.y` | `move__get_a0_y` |
| set `a1` | `move__set_a1` |
| get `a1` | `move__get_a1` |

Naming pattern:

- Parameter index `i` -> prefix `a{i}`
- Threadable field -> `{prefix}_{field_name}` -> `a0_x`, `a0_y`
- Primitive parameter -> just `a1`

### 5.2 `__kernel_meta__` attached to the Python function

After compile, `move` gets metadata Python and C++ both understand. Simplified but accurate for this example:

```python
move.__kernel_symbol__ = "move"

move.__kernel_meta__ = {
  "symbol": "move",
  "call_symbol": "move__call",
  "args_new_symbol": "move__args_new",
  "args_free_symbol": "move__args_free",
  "is_method": False,
  "return_schema": None,
  "return_kind": "void",
  "params": [
    {
      "name": "p",
      "pass_as": "ref",          # C++ takes Particle&
      "kind": "threadable",
      "cpp_type": "Particle",
      "schema": {
        "kind": "threadable",
        "cpp_type": "Particle",
        "type_name": "Particle",
        "fields": [
          {"name": "x", "schema": {"kind": "float", "cpp_type": "double"}},
          {"name": "y", "schema": {"kind": "float", "cpp_type": "double"}},
        ],
      },
    },
    {
      "name": "dt",
      "pass_as": "value",
      "kind": "float",
      "cpp_type": "double",
      "schema": {"kind": "float", "cpp_type": "double"},
    },
  ],
  "types": {"Particle": Particle},   # real Python class object
  "schemas": {
    "Particle": { /* full Particle field layout, same as above */ }
  },
}
```

This dict is what `thread(move, ...)` reads later. It does not contain the C++ source; it only names symbols and describes shapes.

### 5.3 Two kinds of trampoline output: declarations vs definitions

Codegen produces trampolines in two forms:

| Form | Where it lives | What it is |
|------|----------------|------------|
| **Declaration** | `move.hpp` | Name + argument types only; ends with `;` - “this symbol exists” |
| **Definition** | `move.cpp` | Full function body in `{ ... }` - “here is the code” |

Think of declarations as the **menu** of exported entry points, and definitions as the **kitchen**.

There are also two *families* of trampoline symbols:

| Family | Symbols in this example | Job |
|--------|-------------------------|-----|
| **Lifecycle** | `move__args_new`, `move__args_free`, `move__call` | Allocate pack, free pack, run kernel from pack |
| **Accessors** | `move__set_a0_x`, `move__get_a0_x`, … | Read/write one field (or one primitive arg) inside the pack |

How the current codegen splits them:

- **Lifecycle** -> declared in `move.hpp` **and** defined in `move.cpp`
- **Accessors** -> defined in `move.cpp` with `CTHREADS_API` (exported from the DLL). They are **not** listed in `move.hpp` today; marshal finds them by **symbol name** via ctypes after `load_kernels`

The subsections below show both: what actually lands in the header, then the full accessor API as declarations (so you can see every signature), then the matching definitions.

### 5.4 How trampoline declarations are derived (step by step)

Start from `__kernel_meta__` for `move`. Walk parameters in order. For each parameter, walk its schema. Emit one set of accessor names per leaf (or per nested field).

#### Step A - lifecycle declarations (always three)

From meta fields:

| Meta field | Value | Declaration |
|------------|-------|-------------|
| `args_new_symbol` | `"move__args_new"` | `CTHREADS_API void* move__args_new();` |
| `args_free_symbol` | `"move__args_free"` | `CTHREADS_API void move__args_free(void* p);` |
| `call_symbol` | `"move__call"` | `CTHREADS_API void move__call(void* p);` |

These three are exactly what `emit_trampoline_decls(meta)` writes into `move.hpp`.

Meaning of each:

1. **`move__args_new`** - no inputs; returns an opaque `void*` pointing at a fresh `move__args`.
2. **`move__args_free`** - takes that pointer; deletes the pack.
3. **`move__call`** - takes that pointer; casts it back to `move__args*` and calls the real `move(...)`.

#### Step B - pack layout (needed to understand accessors)

Before accessors make sense, the pack struct must exist (defined in the `.cpp`, not declared as a public type in the header):

```cpp
struct move__args {
    Particle a0;   // param index 0 -> always named a0
    double   a1;   // param index 1 -> always named a1
};
```

No `ret` member, because return kind is `void`.

#### Step C - accessor declarations for param 0 (`p: Particle`)

Param index `0` -> prefix `a0`.  
Schema kind is `threadable`, so there is **no** single `move__set_a0(Particle)`.  
Instead, recurse into fields:

| Field | Child prefix | Leaf kind | Declarations produced |
|-------|--------------|-----------|------------------------|
| `x` | `a0_x` | `float` -> C++ `double` | set + get below |
| `y` | `a0_y` | `float` -> C++ `double` | set + get below |

For a non-string primitive leaf, the declaration pair is always:

```cpp
CTHREADS_API void move__set_<prefix>(void* p, double v);
CTHREADS_API void move__get_<prefix>(void* p, double* v);
```

So for `x` and `y`:

```cpp
// write / read pack->a0.x
CTHREADS_API void move__set_a0_x(void* p, double v);
CTHREADS_API void move__get_a0_x(void* p, double* v);

// write / read pack->a0.y
CTHREADS_API void move__set_a0_y(void* p, double v);
CTHREADS_API void move__get_a0_y(void* p, double* v);
```

Argument shapes (same for every float/int/bool accessor in this example):

| Parameter | Role |
|-----------|------|
| `void* p` | Opaque pack pointer (`move__args*` underneath) |
| `double v` (set) | Value to store |
| `double* v` (get) | Out-pointer; callee writes `*v = ...` |

#### Step D - accessor declarations for param 1 (`dt: float`)

Param index `1` -> prefix `a1`.  
Schema kind is already a primitive (`float`), so no field walk:

```cpp
CTHREADS_API void move__set_a1(void* p, double v);
CTHREADS_API void move__get_a1(void* p, double* v);
```

#### Step E - full trampoline declaration list for this example

Putting lifecycle + accessors together (the complete exported trampoline API for `move`):

```cpp
// --- lifecycle (also written into move.hpp) ---
CTHREADS_API void* move__args_new();
CTHREADS_API void  move__args_free(void* p);
CTHREADS_API void  move__call(void* p);

// --- accessors for a0 (Particle p) ---
CTHREADS_API void move__set_a0_x(void* p, double v);
CTHREADS_API void move__get_a0_x(void* p, double* v);
CTHREADS_API void move__set_a0_y(void* p, double v);
CTHREADS_API void move__get_a0_y(void* p, double* v);

// --- accessors for a1 (double dt) ---
CTHREADS_API void move__set_a1(void* p, double v);
CTHREADS_API void move__get_a1(void* p, double* v);
```

That is the whole trampoline surface for this example: **3 lifecycle + 6 accessors = 9 symbols**, plus the real kernel `move` itself.

### 5.5 Generated `move.hpp` (what the header actually contains)

Compile writes the real kernel declaration, then appends **only** the lifecycle trampoline declarations:

```cpp
#pragma once

#include "cthreads_export.hpp"

#include "../__Threadable__/Particle.hpp"

// real kernel
CTHREADS_API void move(Particle& p, double dt);

// trampoline declarations (lifecycle only)
CTHREADS_API void* move__args_new();
CTHREADS_API void move__args_free(void* p);
CTHREADS_API void move__call(void* p);
```

Who uses which declaration later:

| Declaration | Called by |
|-------------|-----------|
| `move(...)` | `move__call` (from C++ inside the DLL) |
| `move__args_new` | `_ext.spawn_from_meta` (native binding) |
| `move__args_free` | worker lambda after writeback |
| `move__call` | worker lambda on the OS thread |

Accessor symbols are still required at runtime (marshal calls them by name); they simply are not repeated as lines in this header.

### 5.6 Generated pack struct and trampoline definitions (`move.cpp`)

Inside `move.cpp`, after the real kernel body, codegen emits the pack type, then **definitions** for every trampoline (lifecycle + accessors).

```cpp
#include "move.hpp"

#include <cstddef>

CTHREADS_API void move(Particle& p, double dt) {
    p.x = (p.x + dt);
    p.y = (p.y + dt);
}

// ---- pack type (private to this translation unit’s trampolines) ----
struct move__args {
    Particle a0;
    double a1;
};

// ---- lifecycle definitions (match the three declarations in move.hpp) ----
CTHREADS_API void* move__args_new() {
    return new move__args();
}

CTHREADS_API void move__args_free(void* p) {
    delete static_cast<move__args*>(p);
}

// ---- accessor definitions (match the declaration list in §5.4 Step E) ----
CTHREADS_API void move__set_a0_x(void* p, double v) {
    static_cast<move__args*>(p)->a0.x = v;
}
CTHREADS_API void move__get_a0_x(void* p, double* v) {
    *v = static_cast<move__args*>(p)->a0.x;
}

CTHREADS_API void move__set_a0_y(void* p, double v) {
    static_cast<move__args*>(p)->a0.y = v;
}
CTHREADS_API void move__get_a0_y(void* p, double* v) {
    *v = static_cast<move__args*>(p)->a0.y;
}

CTHREADS_API void move__set_a1(void* p, double v) {
    static_cast<move__args*>(p)->a1 = v;
}
CTHREADS_API void move__get_a1(void* p, double* v) {
    *v = static_cast<move__args*>(p)->a1;
}

// ---- call definition ----
CTHREADS_API void move__call(void* p) {
    auto* a = static_cast<move__args*>(p);
    move(a->a0, a->a1);   // a0 binds to Particle&; a1 is copied as double
}
```

Line up declaration -> definition for one accessor:

```text
Declaration (logical API):
  CTHREADS_API void move__set_a0_x(void* p, double v);

Definition (move.cpp):
  CTHREADS_API void move__set_a0_x(void* p, double v) {
      static_cast<move__args*>(p)->a0.x = v;
  }
```

What the body is doing, in words:

1. Treat opaque `p` as a `move__args*`.
2. Assign into member `a0.x` (the Particle’s `x` field inside the pack).

Same pattern for every other set/get: cast -> touch one field or `a1`.

Important detail for `pass_as: "ref"`:

- The pack **owns** a full `Particle a0` by value.
- `move__call` passes `a->a0` into `move(Particle& p, ...)`.
- So the reference aliases the pack’s `a0`. Mutations inside `move` change `a0` inside the pack.

### 5.7 Declaration ↔ runtime call cheat sheet

| When | Declaration / symbol used | Example call with our numbers |
|------|---------------------------|-------------------------------|
| Create pack | `move__args_new` | `pack = move__args_new()` |
| Pack `p.x` | `move__set_a0_x` | `move__set_a0_x(pack, 1.0)` |
| Pack `p.y` | `move__set_a0_y` | `move__set_a0_y(pack, 2.0)` |
| Pack `dt` | `move__set_a1` | `move__set_a1(pack, 0.1)` |
| Run kernel | `move__call` | `move__call(pack)` |
| Writeback `x` | `move__get_a0_x` | `move__get_a0_x(pack, &tmp)` -> `p.x = tmp` |
| Writeback `y` | `move__get_a0_y` | `move__get_a0_y(pack, &tmp)` -> `p.y = tmp` |
| Destroy pack | `move__args_free` | `move__args_free(pack)` |

---

## 6. Build and load the kernel library

### 6.1 `build()`

After `.hpp` / `.cpp` exist, `build()` compiles every unit recorded in `STORE` into **one** shared library, for example:

```text
my_project/cthreads_kernels.dll     # Windows
# or .so / .dylib on other OSes
```

That library contains at least:

- `Particle` (as a C++ type used by other symbols)
- `move`
- `move__args_new` / `move__args_free` / `move__call`
- `move__set_a0_x`, `move__set_a0_y`, `move__set_a1`, and the matching getters

### 6.2 `load_kernels(path)`

The native extension `_ext` loads that shared library into the process.

From this point on, C++ can look up symbols by name (`move__call`, …), and Python’s `cthreads.marshal` module can call the same symbols through ctypes.

---

## 7. Runtime: `thread(move, p, 0.1)` step by step

Assume:

```python
p = Particle(1.0, 2.0)
job = thread(move, p, 0.1)
```

### 7.1 High-level Python entry

`thread(...)` (in `prepare.py`) checks whether kernels are already loaded.

- First call: `prepare()` then `load_kernels(...)`, then spawn.
- Later calls: spawn only (no rebuild).

Then it calls into the native extension:

```text
_ext.thread(move, p, 0.1)
```

and wraps the returned native handle in a Python `Job`.

### 7.2 Native checks inside `_ext.thread`

1. `move.__threaded` must be true.
2. `move.__kernel_meta__` must exist (from compile).
3. Args/kwargs are bound against meta param names.

### 7.3 `bind_args` -> ordered values

Meta params are named `"p"` and `"dt"`.

Positional call `thread(move, p, 0.1)` becomes:

```text
ordered_values = [p, 0.1]
```

Same length as `params` (2). Order matches schema order: index 0 is Particle, index 1 is float.

### 7.4 `spawn_from_meta` creates a `SpawnedKernel` / Job

Still before the OS thread starts:

1. Read symbols from meta: `move`, `move__call`, `move__args_new`, `move__args_free`.
2. Allocate pack:

   ```text
   pack = move__args_new()
   ```

   Memory now holds something like:

   ```text
   move__args {
     a0: Particle { x: <uninitialized or 0>, y: <uninitialized or 0> },
     a1: <uninitialized or 0.0>
   }
   ```

3. Fill the pack from Python values (next section).
4. Build a worker lambda that will later: call -> writeback -> unpack return -> free.
5. Wrap that lambda in a `CThread` inside a `SpawnedKernel`.
6. Return it to Python as a `Job` (**not started yet** unless you `start()` / `await`).

`SpawnedKernel` is just:

- `thr`: the native thread object
- `result`: a shared box for the Python return value (starts as `None`)

---

## 8. Packing: Python values -> C++ pack

Packing is done by `fill_pack_from_values` in C++, which immediately calls Python:

```text
cthreads.marshal.pack_params(
    symbol="move",
    params=meta["params"],
    values=[p, 0.1],
    pack_ptr=<integer address of pack>,
    types=...,
    schemas=...,
)
```

### 8.1 What `marshal` is

`cthreads.marshal` is the bridge that:

- opens the loaded kernel library via ctypes,
- walks each parameter’s **schema**,
- calls the matching `move__set_...` trampolines.

It does not run the kernel. It only fills memory.

### 8.2 Loop over parameters

`pack_params` does, conceptually:

```text
for i, (param_meta, value) in enumerate(zip(params, values)):
    pack_value(..., prefix=f"a{i}", schema=param_meta.schema, value=value, pack=...)
```

So:

| `i` | prefix | Python value | schema kind |
|-----|--------|--------------|-------------|
| 0 | `a0` | `p` (`Particle` instance) | `threadable` |
| 1 | `a1` | `0.1` | `float` |

### 8.3 Pack parameter 0 (`Particle`) field by field

For a threadable schema, marshal does **not** call one big “set whole object” function. It walks fields:

```text
for field in schema["fields"]:
    child_prefix = f"{prefix}_{field.name}"   # a0_x, a0_y
    child_value  = getattr(p, field.name)     # p.x, p.y
    pack_value(..., child_prefix, child_schema, child_value, ...)
```

Concrete calls for our numbers:

1. Read `p.x` -> `1.0`  
   Call `move__set_a0_x(pack, 1.0)`  
   -> pack memory: `a0.x = 1.0`

2. Read `p.y` -> `2.0`  
   Call `move__set_a0_y(pack, 2.0)`  
   -> pack memory: `a0.y = 2.0`

### 8.4 Pack parameter 1 (`dt`)

Schema kind is `float` (primitive):

```text
move__set_a1(pack, 0.1)
```

-> pack memory: `a1 = 0.1`

### 8.5 Pack after packing completes

```text
move__args {
  a0: Particle { x: 1.0, y: 2.0 },
  a1: 0.1
}
```

Python object `p` is unchanged so far. The pack holds a **copy** of the field values.

---

## 9. Running the kernel on an OS thread

When you do:

```python
job.start()
```

the native `CThread` starts an OS thread that runs the worker lambda.

### 9.1 Call

```text
move__call(pack)
```

which is:

```cpp
auto* a = static_cast<move__args*>(p);
move(a->a0, a->a1);
```

Inside `move`:

```cpp
p.x = (p.x + dt);   // 1.0 + 0.1 -> 1.1
p.y = (p.y + dt);   // 2.0 + 0.1 -> 2.1
```

Because `p` is a reference to `a->a0`, the pack now holds:

```text
move__args {
  a0: Particle { x: 1.1, y: 2.1 },
  a1: 0.1
}
```

This kernel body runs **without needing the Python GIL** for the arithmetic itself.

### 9.2 Writeback (C++ pack -> Python object)

After `move__call` returns, the worker acquires the GIL and calls:

```text
cthreads.marshal.writeback_params(...)
```

Writeback only cares about mutable structured kinds: `threadable`, `list`, `dict`.  
Primitives like `dt` are not written back.

For `a0` / `p`, it uses getters and writes into the **same** Python object that was passed in:

1. `move__get_a0_x(pack, &tmp)` -> `1.1` -> set `p.x = 1.1`
2. `move__get_a0_y(pack, &tmp)` -> `2.1` -> set `p.y = 2.1`

So the mutation model is:

```text
copy fields in  ->  mutate C++ copy in pack  ->  copy fields out into original Python object
```

Python never shared a live pointer to the C++ `Particle` with the Python instance. It shared **values**, twice.

### 9.3 Unpack return

```text
cthreads.marshal.unpack_return(meta, pack_ptr)
```

Because `return_schema` is `None` / kind `void`, this returns Python `None`.

That value is stored in the Job’s result box.

### 9.4 Free pack

```text
move__args_free(pack)
```

Pack memory is deleted. The Python object `p` still has the updated attributes.

### 9.5 Join / result

```python
job.join()
print(p.x, p.y)      # 1.1 2.1
print(job.result())  # None
```

`join` waits for the OS thread to finish (and re-raises if the worker threw).  
`result()` reads the result box filled during writeback/unpack.

`await job` is the same idea: auto-start, wait without blocking the asyncio event loop, then return `result()`.

---

## 10. Full timeline (one page)

```text
demo.py import
  @Threadable Particle  ->  mark + REGISTRY.threadables["Particle"]
  @Thread move          ->  mark + REGISTRY.threads["move"]

first thread(move, p, 0.1)
  prepare/compile
    Particle -> struct Particle { double x; double y; }
                 in __Threadable__/Particle.{hpp,cpp}
    move     -> void move(Particle& p, double dt) { ... }
                 + struct move__args { Particle a0; double a1; }
                 + hpp decls: move__args_new / free / call
                 + cpp defs: those three + set/get a0_x, a0_y, a1
                 + move.__kernel_meta__
                 in __Thread__/move.{hpp,cpp}
  build -> cthreads_kernels shared library
  load_kernels(that library)

spawn
  bind args -> [p, 0.1]
  pack = move__args_new()
  marshal.pack_params:
      move__set_a0_x(pack, 1.0)
      move__set_a0_y(pack, 2.0)
      move__set_a1(pack, 0.1)
  return Job (thread not started)

job.start()
  OS thread:
    move__call(pack)           # a0 becomes {1.1, 2.1}
    writeback into p           # p.x=1.1, p.y=2.1
    result = None
    move__args_free(pack)

job.join(); job.result() -> None
p is updated in place
```

---

## 11. Mental model (keep this)

| Layer | What it is in this example |
|-------|----------------------------|
| Python `Particle` | Normal class instance with attributes `x`, `y` |
| C++ `Particle` | `struct` with `double x`, `double y` |
| Python `move` | Still exists; used for metadata and optional pure-Python calls |
| C++ `move` | The real off-GIL kernel |
| `move__args` | Temporary native bag holding copies of args for one Job |
| Trampoline **declarations** | Lifecycle trio in `move.hpp` (`args_new` / `free` / `call`) |
| Trampoline **definitions** | That trio plus `set`/`get` accessors in `move.cpp` |
| `marshal` | Schema-driven caller of those trampoline symbols by name |
| `Job` / `SpawnedKernel` | Handle: native thread + result box |

If you remember only one sentence:

**Compile turns your annotated Python into a C++ kernel plus a pack API; `thread()` copies values into a pack, runs the kernel, copies structured results back, and gives you a Job.**

---

## 12. Compile in depth (registry -> C++ files)

§2 said `compile()` writes folders. This section is the same step, slower, aimed at someone who will change that code.

Words used here:

| Phrase in this doc | What it means |
|--------------------|----------------|
| **Registry** | A process-wide dictionary filled by `@Threadable` / `@Thread` at import time (`REGISTRY` in `CONFIG.py`) |
| **Unit** | One thing compile can emit: a Threadable class, or a free Thread function |
| **Fingerprint** | A hash of the Python source of that unit, used to skip rewrite when nothing changed |
| **Store** | `STORE`: unit name -> path of the generated `.hpp` |
| **Syntax tree** | Python’s parsed form of the function (an **AST**: Abstract Syntax Tree). Nested objects, not text |

Code lives mainly in:

```text
src/cthreads/python/cthreads/compile.py              # drain the registry
src/cthreads/python/cthreads/Threadable/compile.py   # one @Threadable class
src/cthreads/python/cthreads/Thread/compile/compile.py  # one free @Thread function
```

### 12.1 `compile()` is a drain, not a JIT of one call

`thread(move, p, 0.1)` may *trigger* compile, but `compile()` does **not** mean “compile only `move`.” It means:

1. Find every class in `REGISTRY.threadables`.
2. Find every function in `REGISTRY.threads`.
3. Emit C++ for all of them (unless the cache says the source is unchanged).

So if `demo.py` also defined `@Threadable class Box` and `@Thread def other(...)`, the first `thread(move, ...)` still compiles **Particle, Box, move, and other**. One shared library later contains every kernel.

If the registry is empty, compile raises: nothing was decorated.

### 12.2 Project root and output folders

Compile picks a **project root** from `inspect.getfile` of the first registered class or function. For `demo.py` that is `my_project/`.

Next to that file it creates (if needed):

| Folder | Holds |
|--------|--------|
| `__Threadable__/` | One `.hpp` + `.cpp` per Threadable class |
| `__Thread__/` | One `.hpp` + `.cpp` per **free** Thread function, plus `cthreads_export.hpp` |

A `@Thread` **method** on a Threadable is **not** a free function. It is written into that class’s `Particle.hpp` / `Particle.cpp` as a C++ member, plus a C wrapper. `move` in our example is free, so it gets its own pair under `__Thread__/`.

Compile also ensures a `.gitignore` so those generated trees are not committed by accident.

### 12.3 Order: structs before functions that mention them

Fixed order:

```text
1. every @Threadable class
     - fields -> C++ struct members
     - @Thread methods on that class -> C++ member functions + C wrappers
2. every remaining @Thread function
     - skip methods already emitted in step 1 (matched by __qualname__)
     - emit free functions like move
```

Why Particle first: `move(p: Particle, ...)` must `#include` `Particle.hpp`. If `move` were compiled first, that include would point at a file that does not exist yet.

### 12.4 One Threadable class (`Particle`)

`compile_threadable(Particle, methods=[])` in our example:

1. Resolve the output paths: `__Threadable__/Particle.hpp` and `Particle.cpp`.
2. Hash the class source (fingerprint). If files exist and the hash matches the cache, **do not rewrite**. Still refresh kernel metadata if needed, still record `STORE["Particle"]`.
3. Otherwise:
   - For each annotated field, map the Python type to a C++ field (see §13.3).
   - For each `@Thread` method, call the **same** translator used for free functions, with `owner_name="Particle"` so `self` becomes C++ `this`.
   - Write the struct, method declarations, method definitions, and any C wrappers / trampolines the methods need.

Our `Particle` has no methods, so the `.cpp` is only `#include "Particle.hpp"`.

### 12.5 One free Thread function (`move`)

`compile_free_thread(move)`:

1. Output paths: `__Thread__/move.hpp`, `move.cpp`. Always refresh `cthreads_export.hpp` (the `CTHREADS_API` macro).
2. Fingerprint the function source. Unchanged + files present -> skip rewrite, but still rebuild `move.__kernel_meta__` so Python packing still works.
3. Otherwise call `translate_thread(move)` (the next section), then:
   - Write the kernel declaration into the `.hpp`, plus trampoline **lifecycle** declarations (`args_new` / `free` / `call`).
   - Write includes, the kernel **definition**, pack struct, trampoline **definitions** (lifecycle + set/get) into the `.cpp`.
   - Attach `__kernel_meta__` on the Python function object.

`STORE["move"]` is set to the `.hpp` path either way, so `build()` knows which files to feed the C++ compiler.

### 12.6 Cache (why the second `thread()` is fast)

Under the project root, compile keeps a small cache of `{ unit_name: { src_hash, hpp, cpp } }`.

| Situation | What happens |
|-----------|----------------|
| First compile | Write files, store hashes |
| Source of `move` unchanged | Do not rewrite `move.hpp` / `move.cpp`; still refresh metadata |
| You edited `move`’s body | Hash changes -> translate and rewrite those two files |
| `force=True` | Ignore hashes, rewrite everything |

`build()` then compiles whatever `STORE` points at into one shared library. Unchanged C++ is still a normal compiler incremental build if the toolchain supports it.

### 12.7 What compile does **not** do

- It does not run `move`.
- It does not start an OS thread.
- It does not pack `p` or `dt`.
- It does not load the DLL; that is `build()` then `load_kernels()`.

Compile’s contract: **Python units in the registry become C++ text on disk, plus metadata on the function objects.**

---

## 13. Translation in depth (syntax tree -> C++ strings)

Translation is the heart of a `@Thread` body. Compile *calls* it; translation does not know about DLLs or Jobs.

For `move`, translation is exactly §4. This section names the machinery so you can change it without guessing.

Code lives under:

```text
src/cthreads/python/cthreads/Thread/compile/
  compile.py                         # translate_thread() entry
  lib.py                             # parse source, C++ literals, includes
  AstTranslators/                    # one file per syntax-tree node kind
    translate.py                     # dispatch tables + walk the function
    context.py                       # TranslateContext (the scratch pad)
    signature.py                     # parameters + return type
    typeof.py                        # “what type is this expression?”
    name.py, attribute.py, binOp.py, call.py, ...
  mathLibTranslators/                # math.sqrt -> std::sqrt, cthreads.math.*
  pythonContainerLibTranslators/     # list.append -> push_back, dict.get, ...
  syncBindingTranslators/            # Lock.acquire, Event.wait, triple-buffer
  linalgTranslations/                # ArrayF32.matmul, Shape, ...
```

### 13.1 From function object to syntax tree

`translate_thread(move)`:

1. Check `move.__threaded` and version match.
2. `inspect.getsource(move)` -> text of the `def`.
3. Parse that text with Python’s `ast` module -> a `FunctionDef` node (the tree in §4.1).
4. `get_type_hints(move)` -> `{ "p": Particle, "dt": float, "return": None }`.
5. `translate_function(move, func_def, hints, owner_name=None)`.

After step 3, **nobody looks at the original source text again**. Every later decision is “what kind of node is this?”

Two families of nodes:

| Family | Examples in `move` | Translator returns |
|--------|--------------------|--------------------|
| **Expression** | `p`, `dt`, `p.x`, `p.x + dt` | One C++ expression string, e.g. `"(p.x + dt)"` |
| **Statement** | `p.x = p.x + dt` | A list of C++ lines, already indented, e.g. `["    p.x = (p.x + dt);"]` |

`translate.py` holds two dictionaries:

- `EXPR_TRANSLATORS`: syntax-tree class -> function that prints an expression
- `STMT_TRANSLATORS`: syntax-tree class -> function that prints statement lines

Unknown expression -> `TypeError` (“unsupported expression …”).  
Unknown statement -> a C++ comment `// unsupported statement: ...` (so a stray `import` does not crash the whole file; it also will not do what you wanted).

### 13.2 The scratch pad: `TranslateContext`

Every translator receives the same mutable object:

| Field | Role in `move` |
|-------|----------------|
| `fn` | The live Python function (`move`), used to read `__globals__` (for `math`, `ArrayF32`, …) |
| `func_name` | `"move"` (error messages) |
| `hints` | The type-hint dict from step 4 above |
| `owner_name` | `None` for a free function; `"Particle"` if this were a method |
| `symbols` | **Name table**: Python local/parameter name -> internal type object |
| `sig_includes` / `body_includes` | `#include` lines for the header vs the `.cpp` body |
| `seen_sig` / `seen_body` | Deduplicate those includes |

The **name table** is how the compiler knows that `p` is a Threadable and `dt` is a float. It is **not** inferred from the C++ it already printed. It is filled when names are *introduced*:

| When | What is inserted |
|------|------------------|
| Translating the signature | `p` -> Threadable `Particle`, `dt` -> float |
| `x: int = ...` (annotated assignment) | `x` -> int |
| `for i in range(n):` | `i` -> int, then **removed** after the loop |
| `for x in xs:` when `xs` is `list[float]` | `x` -> float, then removed after the loop |
| Method with `self` | `self` -> Threadable of `owner_name` |

Plain `x = 1` (no annotation) is **not** allowed for a new name. You must write `x: int = 1`. That is deliberate: the name table has no other way to learn `x`’s type.

### 13.3 Python types -> internal types -> C++ names

Annotations become small objects in `pyTypes.py` (`PyInt`, `PyFloat`, `PyList`, …). Then `.cpp_name` is what is printed.

| You write in Python | Internal object | C++ type printed |
|---------------------|-----------------|------------------|
| `int` | `PyInt` | `int` |
| `float` | `PyFloat` | `double` (Python `float` is IEEE double) |
| `bool` | `PyBool` | `bool` |
| `str` | `PyString` | `std::string` |
| `list[int]` | `PyList` wrapping `PyInt` | `std::vector<int>` |
| `dict[str, int]` | `PyDict` | `std::unordered_map<std::string, int>` |
| `Particle` (`@Threadable`) | `PyThreadable` | `Particle` (the struct) |
| `cthreads.sync.Lock` | `PyCthreadsInternal` | `cthreads::sync::Lock` |
| `cthreads.linalg.ArrayF32` | `PyCthreadsInternal` | `cthreads::linalg::Array<float>` |
| `TBuffer[Particle]` | `PyTBuffer` | `cthreads::sync::tripple_buffer<Particle>` |

Threadable and list/dict **parameters** are passed by **reference** (`Particle&`, `std::vector<int>&`) so the kernel can mutate the pack’s copy. Scalars like `double dt` are passed by value.

Native cthreads classes (Lock, Array, …) are recognized because the pybind type has `__cthreads_internal__ = True` and a matching row in `CTHREADS_INTERNAL_TYPES`. That flag is **not** a Threadable; it means “link the existing C++ header, do not generate a struct.”

### 13.4 Signature walk for `move` (same as §4.2, with the name table)

1. No `self` (not a method).
2. For `p`: hint `Particle` -> `PyThreadable` -> record `symbols["p"]`, add `#include "../__Threadable__/Particle.hpp"`, emit `Particle& p`.
3. For `dt`: hint `float` -> `PyFloat` -> `symbols["dt"]`, emit `double dt`.
4. Return hint `None` -> `void`.
5. Reject `*args`, `**kwargs`, keyword-only args.

Result string:

```cpp
CTHREADS_API void move(Particle& p, double dt)
```

If `move` were a method on `Particle`:

- `owner_name="Particle"`
- First parameter `self` is skipped in the C++ parameter list (it is `this`)
- `symbols["self"]` is `PyThreadable("Particle", ...)`
- `self.x` prints as `this->x` (see §13.6)

### 13.5 Statement walk for `move`

`translate_function` loops `func_def.body` and calls `translate_stmt` on each.

Our two statements are both `Assign`. Rules for `Assign`:

- Exactly one target.
- Target is a name already in the name table, **or** an attribute (`p.x`), **or** a subscript (`xs[i]`).
- New locals must use annotated assignment instead.

Right-hand side is `translate_expr`. Left-hand side is too. Then:

```text
    <left> = <right>;
```

That is how `p.x = p.x + dt` becomes `p.x = (p.x + dt);`.

Other statement kinds the compiler **does** handle (not used in `move`, same dispatch):

| Python | What the translator does |
|--------|---------------------------|
| `x: int = 1` | Put `x` in the name table; emit `int x = 1;` |
| `x += 1` | Emit `x += 1;` (`**=` becomes `x = std::pow(x, ...)` ) |
| `return` / `return x` | `return;` / `return x;` |
| `pass` | No lines |
| `if cond:` / `else:` | `if ((cond)) { ... } else { ... }` |
| `while cond:` | `while ((cond)) { ... }` (`while`/`else` rejected) |
| `for i in range(n):` | `for (int i = 0; i < n; i += 1) { ... }` |
| `for x in xs:` | `for (auto& x : xs) { ... }` only if `xs` is a list type |
| `break` / `continue` | `break;` / `continue;` |
| Bare expression `xs.append(v)` | Translate the call, add `;` |

`for`/`else` and `while`/`else` are rejected. The loop variable must be a plain name and must not already exist in the name table.

### 13.6 Expression walk: names, fields, arithmetic

`translate_expr` looks up `type(node)` in `EXPR_TRANSLATORS`.

#### Names

- `dt` -> must be in the name table -> print `dt`
- unknown name -> `TypeError`
- `self` on a method -> `(*this)` when used as a value; `self.x` is special-cased to `this->x`

#### Constants

| Python | C++ |
|--------|-----|
| `10` | `10` |
| `1.5` | `1.5` |
| `True` / `False` | `true` / `false` |
| `"hi"` | `"hi"` with escapes |
| `None` | **rejected** (no safe kernel `None`) |

#### Attributes (field access)

Two different jobs, easy to mix up:

1. **Print a path** - `p.x` -> `p.x`, `self.x` -> `this->x`. This does **not** look up whether `x` exists. The C++ compiler will fail later if the struct has no such member. Nested `p.a.b` is just string concatenation: `p.a.b`.
2. **Special attributes** - `math.pi` -> `std::numbers::pi`. Array properties like `a.shape` -> `(a).shape()` (C++ uses a method, Python uses a property).

#### Binary / unary / compare / and-or

| Python | C++ |
|--------|-----|
| `a + b` | `(a + b)` (always parenthesized) |
| `a ** b` | `std::pow(a, b)` and `#include <cmath>` |
| `-a` | `(-a)` |
| `not a` | `(!a)` |
| `a < b < c` | `((a < b) && (b < c))` (Python chained-compare meaning) |
| `a and b` | `(a && b)` |

`+` `-` `*` `/` on two Arrays can print as `(a + b)` and then rely on C++ `operator+` on `Array`. Mask combinators (`keep & occ`) need the dedicated C++ helpers (`mask_and`, …); they are **not** the same as integer `&`. If you add that, it belongs in the binary-operator translator, not in the Array method table.

#### Subscript

`xs[i]` -> `(xs[i])`. Python slice syntax `xs[1:3]` is **rejected** at the moment. Array slicing in the kernel should use `Slice` / methods until that translator exists.

#### List displays

`[1, 2, 3]` is a list display (syntax-tree node `List`). It becomes `std::vector<int>{1, 2, 3}`. Empty `[]` becomes `{}` so `xs: list[int] = []` can take the element type from the annotation.

Element types come from literals and from names already in the name table. Mixed `[1, 2.0]` is rejected. Starred `[*xs]` is rejected.

That is why `Shape([2, 3])` can compile: the list becomes a `std::vector<int>`, and C++ `Shape` can construct from that vector.

### 13.7 Calls: a pipeline, not “any Python call”

`foo(...)` is an `ast.Call`. The call translator tries **matchers in a fixed order**. The first one that claims the call wins. If none match, `TypeError: unsupported call`.

Order today:

```text
1. len(...)
2. __sync_state()          # kernel barrier: copy Threadables back to Python mid-run
3. list / dict methods     # xs.append -> push_back, ...
4. triple-buffer methods   # publish, read_at, ...
5. Lock / Event / RWLock
6. Array / Shape methods   # matmul, transpose, ...
7. Array / Shape / Slice constructors
8. math.* and cthreads.math.*
```

Each matcher is allowed to say “not mine” (`None`) so the next can try. A matcher that *knows* the receiver (for example `xs` is a list) but sees a bad method (`xs.remove`) **raises** instead of falling through.

#### Why `append` cannot be printed as `.append`

Python `list.append` is C++ `std::vector::push_back`. The method **tables** (`LIST_METHODS`, `LOCK_METHODS`, `LINALG_METHODS`, …) store:

- Python name
- How many arguments
- A tiny function that builds the C++ string from the already-translated receiver and args

So:

```text
xs.append(v)   ->  (xs).push_back(v)
lock.acquire() ->  (lock).acquire()     # same name, still goes through the table
a.matmul(b)    ->  (a).matmul(b)
a.count()      ->  (a).count_nonzero()  # Python name ≠ C++ name
math.sqrt(x)   ->  std::sqrt(x)
```

Math is resolved through `move.__globals__`: `math` must be the stdlib module, or `cthreads.math` must be the marked submodule. A random `fake.sqrt` is ignored.

Constructors like `ArrayF32(sh)` are the same idea: look up the name in the function’s globals, check `__cthreads_internal__`, emit `cthreads::linalg::Array<float>(sh)`.

### 13.8 Type of the receiver (`typeof`) - why `self.a.matmul(b)` works

The method tables need the type of the object **before the last dot**.

| Call | Receiver expression | How its type is known |
|------|---------------------|------------------------|
| `a.matmul(b)` | name `a` | Name table: `a` was a parameter `a: ArrayF32` |
| `self.a.matmul(b)` | attribute `self.a` | `self` is a Threadable; field `a` is read from that class’s annotations via `REGISTRY` |
| `bodies[i].a.matmul(b)` | `bodies[i].a` | `bodies` is `list[Particle]` -> element is `Particle` -> field `a` |

`typeof.py` implements that walk:

1. Name -> look up the name table.
2. Attribute -> if the base is a Threadable, `get_type_hints` on the **live class** in `REGISTRY.threadables` (the same annotations compile used to emit the struct). No handwritten field dictionary to keep in sync.
3. Subscript `xs[i]` -> if `xs` is a list type, the element type.

Printing the C++ path (`this->a`) is a separate walk (`attribute.py`). Mixing them up is the usual contributor bug: the path can print even when the type is unknown; the **call** will then fail with “unsupported call” because no method table ran.

`obj.a.matmul` where `obj` is not in the name table and is not a typed Threadable field still fails. Bind a local if you need to:

```python
arr: ArrayF32 = self.a
arr.matmul(b)
```

### 13.9 Includes

Whenever a translator needs a C++ header, it calls `add_include` on either `sig_includes` (types that appear in the function signature) or `body_includes` (types / functions used only in the body).

Examples:

- `Particle& p` -> Particle header on the signature
- `math.sqrt` -> `#include <cmath>` on the body
- `ArrayF32` local -> `linalg/array.hpp` on the body

`compile_free_thread` writes signature includes into `move.hpp` and any extra body includes into `move.cpp` under the `#include "move.hpp"`.

### 13.10 What translation returns

A `TranslateResult`:

| Field | For `move` |
|-------|------------|
| `return_type` | `void` |
| `func_name` | `move` |
| `params_csv` | `Particle& p, double dt` |
| `sig_includes` | Particle header |
| `body` | the two assignment lines plus a trailing newline |
| `body_includes` | empty here |

Compile then wraps that in `CTHREADS_API ... { ... }`, trampolines, and files. Translation never writes to disk.

### 13.11 Intentional limits (so you do not “fix” them by accident)

These are not oversights in `move`; they are policy for every Thread body:

- Only the syntax-tree node kinds in the two dispatch tables.
- No `*args` / `**kwargs`.
- No new unannotated locals.
- No Python `None` as a value.
- Calls must match a matcher in §13.7 (no arbitrary `print`, no `obj.unknown()`).
- Host-only Array helpers (`from_list`, `to_list`) are not kernel operations; build arrays in Python and pass them in.
- Method tables still list Python names that are not 1:1 with C++ (`append`, `count`, …). Do not replace those tables with “whatever exists on the Python class” - the Python class is the **host** API, the table is the **kernel** API.

---

## 14. How a contributor should extend this (without making a second compiler)

The existing shape is: **dispatch on syntax-tree node kind**, then **optional matchers** for calls. New work should fit that, not bypass it.

### 14.1 New statement or expression syntax

Example: you want dictionary displays `{ "a": 1 }` in a kernel.

1. Add `AstTranslators/dictLiteral.py` with a `translate(node, ctx) -> str` (expressions) or `-> list[str]` (statements).
2. Register it on `EXPR_TRANSLATORS` or `STMT_TRANSLATORS` in `translate.py`.
3. Add unit tests next to `tests/unit/test_Ast_exprs.py` / `test_Ast_stmts.py` using `make_ctx` / `parse_expr` (see `tests/helpers.py`).
4. Keep the emit logic in that one file. Split a helper only if a second translator needs the exact same function.

### 14.2 New method on an existing typed object

Example: `xs.insert` already exists; a new list method would go in `pythonContainerLibTranslators/containerOps.py` (`LIST_METHODS`). Same pattern for Lock, Array, triple-buffer.

Do **not** teach `typeof` about the method. `typeof` only answers “what is the receiver.” The table answers “what C++ to print.”

### 14.3 New native type (another `cthreads.*` class)

1. Mark the pybind class `__cthreads_internal__ = True` (like Lock / Array).
2. Add a row to `CTHREADS_INTERNAL_TYPES` in `pyTypes.py` (Python name -> C++ type + header).
3. If it has methods, add a resolver + table next to `syncBindingTranslators` / `linalgTranslations`, and one more `if` in `call.py` **before** the final “unsupported call” error.
4. Tests: dummy class with `__cthreads_internal__` is enough for translation tests; live pybind tests belong under `tests/unit/...` for the binding itself.

### 14.4 New Threadable field type

If `@Threadable` already allows the annotation (`is_threadable` in `Threadable/lib.py`) and `hint_to_pytype` can map it, **`typeof` picks the field up automatically** from `get_type_hints`. You should not add a field map by hand.

### 14.5 Tests that match how the project already tests this

| Kind | Example file |
|------|----------------|
| One syntax-tree node | `tests/unit/test_Ast_exprs.py`, `test_Ast_stmts.py` |
| Signature / params | `tests/unit/test_signature_translate.py` |
| List methods | `tests/unit/test_container_ops.py` |
| Math | `tests/unit/test_math_ops.py` |
| Sync methods | `tests/unit/test_sync_ops.py` |
| Linalg methods / constructors | `tests/unit/test_linalg_ops.py` |
| `typeof` chains | `tests/unit/test_typeof.py` |
| Full compile + run | `tests/integration/test_compile_api.py` |

Helpers: `make_ctx(symbols=..., owner_name=..., globals_extra=...)` builds a fake `TranslateContext` without writing files. `registered_threadable(cls)` puts a dummy class on `REGISTRY` so field lookup works in tests, then removes it.

### 14.6 Mental picture for the compiler folders

```text
@Thread / @Threadable
        │
        ▼
    REGISTRY          (import time)
        │
        ▼
    compile()         (first thread() or explicit api.compile)
        │
        ├── Threadable.compile  ->  struct + methods on disk
        └── translate_thread
                │
                ├── signature  ->  C++ params + name table
                └── body walk
                      ├── translate_stmt / translate_expr
                      ├── typeof(receiver) for methods
                      └── method/math tables
        │
        ▼
    .hpp / .cpp  +  __kernel_meta__
        │
        ▼
    build() / load_kernels() / marshal / Job     (unchanged by most translator work)
```

If you remember only one sentence for this half of the file:

**Compile drains the registry onto disk; translation walks the syntax tree with a name table and a short list of call matchers; `typeof` types dotted receivers from Threadable annotations; trampolines and `thread()` are a later layer and do not belong in a new expression translator.**

