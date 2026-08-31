# V2 Syntax

AST lowering for `@Thread` **bodies**: Python language constructs -> C++ expression strings or statement lines.

Parent overview and signature / context / includes: [`../DOCS.md`](../DOCS.md).

Entry points: **`Syntax.expr`** and **`Syntax.stmt`**. Everything else is an area static class registered in those dispatch tables.

---

## Layout

```text
syntax/
  DOCS.md       ← this file
  Syntax.py     dispatcher
  Literal.py    Constant, List
  Name.py       Name / self
  Op.py         BinOp, UnaryOp, Compare, BoolOp + tables
  Assign.py     AnnAssign, Assign, AugAssign
  Flow.py       If, For, While, Break, Continue, Pass, Return, Expr
  Index.py      Subscript
```

Helpers used from the parent package:

- [`Cpp.literal`](../DOCS.md#cpppy) / [`Cpp.CMATH`](../DOCS.md#cpppy)
- [`Source.resolve_annotation`](../DOCS.md#sourcepy)
- [`add_include`](../DOCS.md#includepy) / [`include_for`](../DOCS.md#includepy)
- [`TranslationContext`](../DOCS.md#contextpy)

---

## `Syntax` - dispatcher

### `Syntax.expr(node, ctx) -> str`

Looks up `type(node)` in `_EXPR` and runs the handler.

| AST type | Handler |
|----------|---------|
| `ast.Constant` | [`Literal.constant`](#literalconstant) |
| `ast.Name` | [`Name.name`](#namename) |
| `ast.Subscript` | [`Index.subscript`](#indexsubscript) |
| `ast.BinOp` | [`Op.bin_op`](#opbin_op) |
| `ast.UnaryOp` | [`Op.unary_op`](#opunary_op) |
| `ast.Compare` | [`Op.compare`](#opcompare) |
| `ast.BoolOp` | [`Op.bool_op`](#opbool_op) |
| `ast.List` | [`Literal.list_display`](#literallist_display) |

Unknown expression -> `TypeError`.  
`ast.Call` / `ast.Attribute` -> see [Not wired](#not-wired-call--attribute).

### `Syntax.stmt(node, ctx) -> list[str]`

Looks up `type(node)` in `_STMT`. Returns a **list of C++ lines** (usually already indented with four spaces).

| AST type | Handler |
|----------|---------|
| `ast.AnnAssign` | [`Assign.ann_assign`](#assignann_assign) |
| `ast.Assign` | [`Assign.assign`](#assignassign) |
| `ast.AugAssign` | [`Assign.aug_assign`](#assignaug_assign) |
| `ast.Pass` | [`Flow.pass_stmt`](#flowpass_stmt) |
| `ast.Return` | [`Flow.return_stmt`](#flowreturn_stmt) |
| `ast.Expr` | [`Flow.expr_stmt`](#flowexpr_stmt) |
| `ast.If` | [`Flow.if_stmt`](#flowif_stmt) |
| `ast.For` | [`Flow.for_stmt`](#flowfor_stmt) |
| `ast.While` | [`Flow.while_stmt`](#flowwhile_stmt) |
| `ast.Break` | [`Flow.break_stmt`](#flowbreak_stmt) |
| `ast.Continue` | [`Flow.continue_stmt`](#flowcontinue_stmt) |

Unknown statement -> `[f"    // unsupported statement: {TypeName}"]` (does not raise).

Handlers call back into `Syntax.expr` / `Syntax.stmt` for nested nodes (lazy import to avoid cycles).

---

## Not wired: Call / Attribute

`Syntax.expr` raises if it sees `ast.Call` or `ast.Attribute`:

```text
… Attribute is handled by plugins (not wired yet)
```

That covers:

- Field access: `self.x`, `p.velocity`
- Methods: `xs.append(v)`, `lock.acquire()`
- Free calls: `math.sin(x)`, `len(xs)` (except `range` inside [`Flow.for_stmt`](#flowfor_stmt), which special-cases the call without going through `Syntax.expr` on the `Call` node as a general call)

Plugins will use [`Typeof.of`](../DOCS.md#typeofpy) for the receiver type.

---

## `Literal`

### `Literal.constant`

**AST:** `ast.Constant`  
**Returns:** C++ literal via [`Cpp.literal`](../DOCS.md#cpppy).

| Python | C++ |
|--------|-----|
| `10` | `10` |
| `1.5` | `1.5` |
| `True` | `true` |
| `"hi"` | `"hi"` |

### `Literal.list_display`

**AST:** `ast.List`  
**Returns:** `std::vector<T>{…}` or `{}` for empty.

| Python | C++ |
|--------|-----|
| `[]` | `{}` (+ `#include <vector>`) |
| `[1, 2, 3]` | `std::vector<int>{1, 2, 3}` |
| `[x, y]` | uses `ctx.symbols` for element `cpp_name` |
| `[[1, 2], [3, 4]]` | nested `std::vector<…>` |

Rules:

- No starred elements (`[*xs]`).
- Element type inferred from constants, typed names, nested lists; mixed types error.
- Adds `#include <vector>` (and `<string>` if element type mentions `std::string`) to `ctx.body_includes`.
- Elements lowered with [`Syntax.expr`](#syntaxexpr--syntaxstmt).

### `Literal._elem_cpp_type` (internal)

Infers one element’s C++ type string for list displays; `None` if unknown (e.g. call / binop - those need an annotated assignment for empty/untyped lists).

---

## `Name`

### `Name.name`

**AST:** `ast.Name`  
**Returns:** C++ identifier expression.

| Python | C++ |
|--------|-----|
| `dt` (in `ctx.symbols`) | `dt` |
| `self` when `ctx.owner` is set | `(*this)` |
| unknown name | `TypeError` |

`self` must still be present in `ctx.symbols` after [`Signature.translate`](../DOCS.md#signaturepy) for typing ([`Typeof`](../DOCS.md#typeofpy)); the emitted text is `(*this)`, not the name `self`.

---

## `Op`

Operator tables + binary / unary / compare / bool expressions. Also `range` detection for for-loops.

### Tables

| Attr | Maps | C++ tokens |
|------|------|------------|
| `Op.BINOPS` | `Add` `Sub` `Mult` `Div` `FloorDiv` `Mod` shifts / bitwise | `+` `-` `*` `/` `%` `<<` … |
| `Op.UNARYOPS` | `UAdd` `USub` `Not` `Invert` | `+` `-` `!` `~` |
| `Op.CMPOPS` | `Eq` `NotEq` `Lt` `LtE` `Gt` `GtE` | `==` `!=` `<` … |
| `Op.BOOLOPS` | `And` `Or` | `&&` `\|\|` |
| `Op.BUILTINS` | frozenset | `range`, `len`, `__sync_state` |

`Pow` (`**`) is **not** in `BINOPS`; handled specially in [`bin_op`](#opbin_op) / [`Assign.aug_assign`](#assignaug_assign).

### `Op.is_builtin_call(node, name) -> bool`

True when `node` is `name(...)` and `name` is in `Op.BUILTINS`. Used by [`Flow.for_stmt`](#flowfor_stmt) for `range`.

### `Op.bin_op`

**AST:** `ast.BinOp`

| Python | C++ |
|--------|-----|
| `a + b` | `(a + b)` |
| `a ** b` | `std::pow(a, b)` + [`Cpp.CMATH`](../DOCS.md#cpppy) |

Recurses with [`Syntax.expr`](#syntaxexpr--syntaxstmt) on left/right.

### `Op.unary_op`

**AST:** `ast.UnaryOp`

| Python | C++ |
|--------|-----|
| `-x` | `(-x)` |
| `not flag` | `(!flag)` |

### `Op.compare`

**AST:** `ast.Compare`

| Python | C++ |
|--------|-----|
| `a < b` | `(a < b)` |
| `a < b < c` | `((a < b) && (b < c))` |

Chained comparisons follow Python pairwise semantics joined with `&&`.

### `Op.bool_op`

**AST:** `ast.BoolOp`

| Python | C++ |
|--------|-----|
| `a and b` | `(a && b)` |
| `a or b or c` | `(a \|\| b \|\| c)` |

Requires at least two values. Note: Python `and`/`or` return an operand; C++ yields bool - fine for conditions in this subset.

---

## `Assign`

### `Assign.ann_assign`

**AST:** `ast.AnnAssign`  
**Returns:** one declaration line.

| Python | C++ |
|--------|-----|
| `scale: float = 1.0` | `    double scale = 1.0;` |
| `xs: list[int]` (no value) | `    std::vector<int> xs;` |

Rules:

- Target must be a plain `Name` (not attribute / subscript).
- No redeclaration if name already in `ctx.symbols`.
- Annotation via [`Source.resolve_annotation`](../DOCS.md#sourcepy) -> `hint_to_pytype`.
- Registers type in `ctx.symbols`; [`include_for`](../DOCS.md#includepy) -> `body_includes`.
- RHS via [`Syntax.expr`](#syntaxexpr--syntaxstmt); uses `PyType.to_cpp`.

### `Assign.assign`

**AST:** `ast.Assign`  
**Returns:** `    <lhs> = <rhs>;`

| Python | Notes |
|--------|--------|
| `n = n + i` | name must already be in `symbols` |
| `xs[i] = v` | LHS via [`Index.subscript`](#indexsubscript) |
| `p.x = v` | needs Attribute plugin - fails today |
| `a = b = 0` | multi-target not supported |

New locals must use AnnAssign (`x: int = …`), not bare Assign.

### `Assign.aug_assign`

**AST:** `ast.AugAssign`

| Python | C++ |
|--------|-----|
| `n += i` | `    n += i;` |
| `x **= 2` | `    x = std::pow(x, 2);` + cmath |

Uses [`Op.BINOPS`](#tables) for the operator token. Target/value via [`Syntax.expr`](#syntaxexpr--syntaxstmt).

---

## `Flow`

Control flow and simple statements. Nested bodies get one extra indent via [`Flow.nest`](#flownest).

### `Flow.nest(lines) -> list[str]`

Prepends four spaces to each non-empty line (statements already have one indent level from handlers).

### `Flow.if_stmt`

**AST:** `ast.If`

```python
if cond:
    ...
else:
    ...
```

```cpp
    if (<cond>) {
        ...
    } else {
        ...
    }
```

- Test: [`Syntax.expr`](#syntaxexpr--syntaxstmt) (often [`Op.compare`](#opcompare)).
- `elif` is `orelse=[If(...)]` in the AST; nested `if` inside `else` is fine.
- Body stmts: [`Syntax.stmt`](#syntaxexpr--syntaxstmt) then [`nest`](#flownest).

### `Flow.for_stmt`

**AST:** `ast.For`  
**Not supported:** `for`/`else`, non-`Name` targets, rebinding an existing symbol as the loop var.

#### `range` loops

Detected with [`Op.is_builtin_call(it, "range")`](#opis_builtin_call) (does not go through general Call plugins).

| Python | C++ |
|--------|-----|
| `for i in range(n):` | `for (int i = 0; i < n; i += 1)` |
| `for i in range(a, b):` | `for (int i = a; i < b; i += 1)` |
| `for i in range(a, b, s):` | `for (int i = a; i < b; i += s)` |

- No keyword args to `range`.
- Loop var typed as `PyInt` in `ctx.symbols` for the body, then **deleted**.
- Step assumed positive (common case).

#### Container loops

| Python | C++ |
|--------|-----|
| `for x in xs:` | `for (auto& x : xs) { … }` |

- `xs` must be a `Name` whose `ctx.symbols` type is `PyList`.
- Loop var type = `inner_type` for the body, then deleted.

### `Flow.while_stmt`

**AST:** `ast.While`  
No `while`/`else`.

```cpp
    while (<test>) {
        ...
    }
```

### `Flow.break_stmt` / `Flow.continue_stmt`

| Python | C++ |
|--------|-----|
| `break` | `    break;` |
| `continue` | `    continue;` |

### `Flow.pass_stmt`

Returns `[]` (no C++).

### `Flow.return_stmt`

| Python | C++ |
|--------|-----|
| `return` | `    return;` |
| `return expr` | `    return <expr>;` |

Value via [`Syntax.expr`](#syntaxexpr--syntaxstmt).

### `Flow.expr_stmt`

**AST:** `ast.Expr` (expression used as a statement).

| Case | Behavior |
|------|----------|
| Docstring `Expr(Constant(str))` | ignored -> `[]` |
| `Expr(Call(...))` | `    <expr>;` via [`Syntax.expr`](#syntaxexpr--syntaxstmt) - **needs Call plugin** |
| other | comment line `// unsupported statement: Expr (…)` |

---

## `Index`

### `Index.subscript`

**AST:** `ast.Subscript`  
**Returns:** `(base[index])`.

| Python | C++ |
|--------|-----|
| `xs[i]` | `(xs[i])` |
| `d[k]` | `(d[k])` |
| `xs[1:3]` | error (slices not supported) |

Base and index via [`Syntax.expr`](#syntaxexpr--syntaxstmt).  
Typing of `xs[i]` as an *expression type* is [`Typeof.of`](../DOCS.md#typeofpy) (list -> `inner_type`); Index itself only emits text.

---

## Recursion cheat sheet

Typical tree for `n: int = 2 + 3`:

```text
Syntax.stmt(AnnAssign)
  └─ Assign.ann_assign
       ├─ Source.resolve_annotation(int)
       └─ Syntax.expr(BinOp)
            └─ Op.bin_op
                 ├─ Syntax.expr(Constant 2) -> Literal.constant -> Cpp.literal
                 └─ Syntax.expr(Constant 3) -> Literal.constant -> Cpp.literal
```

Typical tree for `if n > 0: n = n + i`:

```text
Syntax.stmt(If)
  └─ Flow.if_stmt
       ├─ Syntax.expr(Compare) -> Op.compare
       │    ├─ Syntax.expr(Name n) -> Name.name
       │    └─ Syntax.expr(Constant 0)
       └─ Syntax.stmt(Assign) -> Assign.assign
            ├─ Syntax.expr(Name n)
            └─ Syntax.expr(BinOp) -> Op.bin_op
                 ├─ Name.name("n")
                 └─ Name.name("i")
```

---

## Error style

Most handlers raise `TypeError` with:

```text
Thread function {ctx.func_name}: …
```

Unknown **statements** are soft-failed with a `// unsupported` comment line; unknown **expressions** hard-fail (except the explicit Call/Attribute plugin message).
