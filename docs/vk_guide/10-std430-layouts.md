# 10 — `std430` layouts (scalar structs that match C++)

## Why layout rules exist

The CPU uploads a blob of bytes for scalars.
The shader interprets those bytes as a struct.

If padding differs, `n` and `a` get scrambled. Silent corruption — nasty to debug.

GLSL storage buffers default to **`std430`** packing when you write:

```glsl
layout(..., std430) buffer Scalars { ... };
```

Your host C++ struct must match **std430**, not "whatever the C++ compiler naturally did" unless you carefully match.

## Rules of thumb (std430)

For members we use in v1:

| Type | Size | Alignment |
|------|------|-----------|
| `int` / `uint` / `float` | 4 | 4 |
| `bool` in GLSL | treat carefully; prefer `int` as 0/1 on host | 4 |
| `vec2` | 8 | 8 |
| `vec3` | 12 | **16** (align like vec4) |
| `vec4` | 16 | 16 |

Arrays of `float`/`int` in an SSBO are tightly packed (stride 4).

Structs get padding so each member starts at a multiple of its alignment.
The struct's overall alignment is the max of its members (roughly).

## Easy case (saxpy scalars)

GLSL:

```glsl
layout(set=0, binding=0, std430) buffer Scalars {
    int n;
    float a;
} scalars;
```

Host:

```cpp
struct ScalarPack {
    int32_t n;  // offset 0
    float a;    // offset 4
};
// sizeof == 8, no surprise padding
```

Use fixed-width types (`int32_t`) so Windows LLP64 and Linux agree.

## Hard case (why vec3 bites)

```glsl
float a;
vec3 v;
```

`v` must start at offset 16, not 4. Bytes 4..15 are padding.

Host must insert the same padding or use `alignas`.

**For the current cthreads GPU path:** prefer scalars that are int/float/bool-as-int. Avoid vec3 in the scalar blob until a layout emitter matches std430 automatically.

## Lists are not inside the scalar struct

Do not put `float x[];` inside the scalar block when you also need `y[]`.
Unsized arrays in std430 must be last, and you only get one.

That is another reason cthreads uses **separate list SSBOs**.

## How codegen will help later

The `@Gpu` emitter should emit:

1. GLSL `std430` scalar block from kernel meta field order.
2. Matching host mirror struct / byte offsets for marshal.

Until then, hand-written smoke tests use a tiny fixed struct both sides agree on.

## Checklist when adding a scalar field

1. Append field to GLSL block in the same order.
2. Append field to host struct with matching type width.
3. Recompute offsets; watch for padding.
4. Bump any cached shader hash (old SPIR-V would mismatch).
5. Upload `sizeof(host_struct)` bytes (include trailing padding if required by std430).
