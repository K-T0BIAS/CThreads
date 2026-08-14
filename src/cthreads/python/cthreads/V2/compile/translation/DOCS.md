# V2 Translation

Turns one `@Thread` function into C++ **strings**. It does **not** write `.hpp` / `.cpp` files — that is emit’s job later.

**Prerequisite:** `CompileSession.compile()` has already filled `REGISTRY.threadable_units` / `thread_units` (paths + field types). Translation uses those for includes and for [`Typeof`](#typeofpy).

**Body AST:** language nodes are documented in [`syntax/DOCS.md`](./syntax/DOCS.md). Call / Attribute are reserved for plugins and are not wired yet.

---

## Layout

```text
translation/
  DOCS.md              ← this file
  __init__.py          public exports
  context.py           TranslationContext
  result.py            SignatureResult, TranslationResult
  include.py           add_include, include_for
  Cpp.py               Cpp.literal, Cpp.CMATH
  Source.py            Source.parse_function, Source.resolve_annotation
  Typeof.py            Typeof.of, Typeof.src
  Signature.py         Signature.translate
  syntax/              ← see syntax/DOCS.md
    Syntax.py          Syntax.expr, Syntax.stmt
    Literal.py Name.py Op.py Assign.py Flow.py Index.py
```

---

## Full flow (example)

Python:

```python
@Threadable
class Particle:
    x: float

    @Thread
    def move(self, dt: float) -> None:
        scale: float = 1.0
        n: int = 2 + 3
        if n > 0:
            for i in range(3):
                n = n + i
        return
```

### 0. Units already exist

After `CompileSession.compile()`:

- `REGISTRY.threadable_units["Particle"]` has `hpp_path`, `fields`, `methods`
- Method unit has `params`, `return_type`, `owner`

### 1. Parse source — [`Source.parse_function`](#sourcepy)

```python
from cthreads.V2.compile.translation import (
    Source, TranslationContext, Signature, Syntax, TranslationResult,
)

fn = Particle.move
unit = REGISTRY.threadable_units["Particle"]
func_def = Source.parse_function(fn)   # ast.FunctionDef
```

### 2. Create context — [`TranslationContext`](#contextpy)

```python
ctx = TranslationContext(
    fn=fn,
    this_file=unit.hpp_path,   # includes are relative to this generated file
    owner=unit,                # method; None for free @Thread
)
```

### 3. Signature — [`Signature.translate`](#signaturepy)

```python
sig = Signature.translate(func_def, ctx)
```

Effects:

| Result | Value |
|--------|--------|
| `sig.func_name` | `"move"` |
| `sig.params_csv` | `"double dt"` (`self` dropped from C++) |
| `sig.return_type` | `None` → void |
| `ctx.symbols["self"]` | `PyThreadable("Particle")` |
| `ctx.symbols["dt"]` | `PyFloat()` |
| `ctx.sig_includes` | any `#include` needed by params / return |

Uses [`include_for`](#includepy) / [`add_include`](#includepy) and `hint_to_pytype`.

### 4. Body — [`Syntax.stmt`](./syntax/DOCS.md#syntaxexpr--syntaxstmt) / [`Syntax.expr`](./syntax/DOCS.md#syntaxexpr--syntaxstmt)

```python
body_lines: list[str] = []
for stmt in func_def.body:
    body_lines.extend(Syntax.stmt(stmt, ctx))
body = "\n".join(body_lines) + ("\n" if body_lines else "")
```

What each statement hits in this example:

| Python | Syntax entry | Detail |
|--------|--------------|--------|
| `scale: float = 1.0` | [`Assign.ann_assign`](./syntax/DOCS.md#assignann_assign) | annotation via [`Source.resolve_annotation`](#sourcepy); RHS via [`Literal.constant`](./syntax/DOCS.md#literalconstant) → [`Cpp.literal`](#cpppy) |
| `n: int = 2 + 3` | [`Assign.ann_assign`](./syntax/DOCS.md#assignann_assign) | RHS [`Op.bin_op`](./syntax/DOCS.md#opbin_op) |
| `if n > 0:` … | [`Flow.if_stmt`](./syntax/DOCS.md#flowif_stmt) | test [`Op.compare`](./syntax/DOCS.md#opcompare); body nested with [`Flow.nest`](./syntax/DOCS.md#flownest) |
| `for i in range(3):` | [`Flow.for_stmt`](./syntax/DOCS.md#flowfor_stmt) | `range` via [`Op.is_builtin_call`](./syntax/DOCS.md#opis_builtin_call); loop var in `ctx.symbols` then removed |
| `n = n + i` | [`Assign.assign`](./syntax/DOCS.md#assignassign) | names via [`Name.name`](./syntax/DOCS.md#namename); `+` via [`Op.bin_op`](./syntax/DOCS.md#opbin_op) |
| `return` | [`Flow.return_stmt`](./syntax/DOCS.md#flowreturn_stmt) | bare `return;` |

Emitted body (approx.):

```cpp
    double scale = 1.0;
    int n = (2 + 3);
    if ((n > 0)) {
        for (int i = 0; i < 3; i += 1) {
            n = (n + i);
        }
    }
    return;
```

**Not supported yet in this example’s “real” physics body:** `self.x = …` needs [`ast.Attribute`](./syntax/DOCS.md#not-wired-call--attribute) (plugins).

### 5. Pack result — [`TranslationResult`](#resultpy)

```python
result = TranslationResult(
    return_type=sig.return_type,
    func_name=sig.func_name,
    params_csv=sig.params_csv,
    sig_includes=list(ctx.sig_includes),
    body=body,
    body_includes=list(ctx.body_includes),
)

result.method_decl()
# '    void move(double dt);'

result.method_def_signature("Particle")
# 'void Particle::move(double dt)'
```

There is **no** `translate_function()` helper yet; the glue above is the intended API until one is added.

---

## Top-level modules

### `context.py`

#### `TranslationContext`

Mutable state for **one** function walk.

| Field | Type | Meaning |
|-------|------|---------|
| `fn` | `Callable` | Live `@Thread` function |
| `this_file` | `Path` | Generated `.hpp`/`.cpp` path; base for relative includes |
| `owner` | `ThreadableUnit \| None` | Method owner, or `None` if free function |
| `symbols` | `dict[str, PyType]` | Local / param name → type (`typeof` / name checks) |
| `sig_includes` | `list[str]` | `#include` lines from the signature |
| `body_includes` | `list[str]` | `#include` lines from the body |
| `seen_sig` / `seen_body` | `set[str]` | Dedup for [`add_include`](#includepy) |

| Property | Meaning |
|----------|---------|
| `func_name` | `fn.__name__` (used in error messages) |

---

### `result.py`

#### `SignatureResult`

Return value of [`Signature.translate`](#signaturepy). Symbols stay on the context.

| Field | Meaning |
|-------|---------|
| `return_type` | `PyType` or `None` (= void) |
| `func_name` | C++ function name (`func_def.name`) |
| `params_csv` | Contents inside `(...)`, e.g. `"double dt"` or `"Particle& p, int n"` |

#### `TranslationResult`

Emit contract for one translated function.

| Field | Meaning |
|-------|---------|
| `return_type` | Same as signature |
| `func_name` | Same |
| `params_csv` | Same |
| `sig_includes` | Snapshot of signature includes |
| `body` | Full body text (indented lines + newlines) |
| `body_includes` | Snapshot of body includes |

| Method | Example output |
|--------|----------------|
| `free_signature()` | `CTHREADS_API int add(int a, int b)` |
| `method_decl()` | `    void move(double dt);` |
| `method_def_signature("Particle")` | `void Particle::move(double dt)` |

`None` return type formats as `void`.

---

### `include.py`

#### `add_include(bucket, seen, text)`

Splits `text` into lines and appends each unique line to `bucket` (tracked by `seen`). Used for both signature and body include lists.

#### `include_for(py_type, this_file) -> str`

Builds `#include …` text needed to use `py_type` in the file at `this_file` (the **generated** file, not the `.py` source).

| `py_type` | Behavior |
|-----------|----------|
| `PyThreadable` | `REGISTRY.threadable_units[name].hpp_path` → `os.path.relpath` from `this_file`’s directory; skip if same file (no self-include); requires units filled |
| `PyList` / `PyDict` / `PyTBuffer` | Own header via `PyType.build_include` **plus** recursive `include_for` on inners (so `list[Particle]` still gets `Particle.hpp`) |
| Other | `py_type.build_include()` |

---

### `Cpp.py`

#### `Cpp.CMATH`

`"#include <cmath>\n"` — used when lowering `**` ([`Op.bin_op`](./syntax/DOCS.md#opbin_op), [`Assign.aug_assign`](./syntax/DOCS.md#assignaug_assign)).

#### `Cpp.literal(value) -> str`

| Python | C++ |
|--------|-----|
| `True` / `False` | `true` / `false` |
| `str` | escaped `"..."` |
| `None` | error |
| other | `str(value)` (ints, floats, …) |

Used by [`Literal.constant`](./syntax/DOCS.md#literalconstant).

---

### `Source.py`

#### `Source.parse_function(fn) -> ast.FunctionDef`

`inspect.getsource` → dedent → `ast.parse` → first `FunctionDef` in the module body.

#### `Source.resolve_annotation(node, globals_ns) -> Any`

`ast.unparse` + `eval` in `globals_ns`, then validates with `hint_to_pytype`. Used by [`Assign.ann_assign`](./syntax/DOCS.md#assignann_assign).

---

### `Typeof.py`

Receiver typing for method / attribute lowering (plugins). Not required for the language-only Syntax walk, but ready for Call/Attribute.

#### `Typeof.src(node) -> str`

Best-effort source for errors (`ast.unparse`, else type name).

#### `Typeof.of(node, ctx) -> PyType | None`

| Node | Rule |
|------|------|
| `Name` | `ctx.symbols.get(id)` |
| `Attribute` | `Typeof.of(value)` must be `PyThreadable` → `REGISTRY.threadable_units[name].fields[attr]` |
| `Subscript` (not slice) | base `PyList` → `inner_type` |
| else | `None` |

---

### `Signature.py`

#### `Signature.translate(func_def, ctx) -> SignatureResult`

1. `get_type_hints(ctx.fn)` (with owner class in `localns` if method).
2. If method and first arg is `self`: register `symbols["self"] = PyThreadable(owner.name)`, drop from C++ params.
3. For each remaining arg: require annotation → `hint_to_pytype` → `symbols` + [`include_for`](#includepy) into `sig_includes`.
4. Pass-by-ref (`Type& name`) for TBuffer / `PyThreadable` / `PyList` / `PyDict`; scalars by value.
5. Reject `*args` / `**kwargs` / kw-only.
6. Return annotation → `PyType` or `None` (void); include return type headers if needed.

---

### `syntax/` package

Documented in full in [`syntax/DOCS.md`](./syntax/DOCS.md).

Quick map of what the top-level flow calls:

| Call site | Syntax API |
|-----------|------------|
| Body loop | [`Syntax.stmt`](./syntax/DOCS.md#syntaxexpr--syntaxstmt) |
| Nested exprs | [`Syntax.expr`](./syntax/DOCS.md#syntaxexpr--syntaxstmt) |
| AnnAssign / Assign / AugAssign | [`Assign.*`](./syntax/DOCS.md#assign) |
| If / For / While / Return / … | [`Flow.*`](./syntax/DOCS.md#flow) |
| `+` `-` `*` `**` comparisons `and`/`or` | [`Op.*`](./syntax/DOCS.md#op) |
| Names / `self` | [`Name.name`](./syntax/DOCS.md#namename) |
| Literals / `[…]` | [`Literal.*`](./syntax/DOCS.md#literal) |
| `xs[i]` | [`Index.subscript`](./syntax/DOCS.md#indexsubscript) |

---

## What is not done yet

| Missing | Notes |
|---------|--------|
| `translate_function()` one-shot | Glue Signature + Syntax → `TranslationResult` |
| Call / Attribute plugins | [`Syntax.expr`](./syntax/DOCS.md#not-wired-call--attribute) raises today |
| Emit | Write files from `TranslationResult` + unit paths |
| Kernel meta / trampolines | Still v1 |

---

## Public exports (`__init__.py`)

`add_include`, `include_for`, `TranslationContext`, `SignatureResult`, `TranslationResult`, `Cpp`, `Source`, `Typeof`, `Signature`, `Syntax`, `Literal`, `Name`, `Op`, `Assign`, `Flow`, `Index`.
