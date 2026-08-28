"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

This module builds compile-time kernel metadata and emits C++ trampoline code.

For each compiled `@Thread` function it produces a `KernelMeta` record
(`TypeSchema` for every parameter and return) and generates the pack struct plus
trampoline accessors that `cthreads.marshal` calls at runtime. Parameters and
returns share one recursive schema shape so pack, writeback, and
`job.result()` all round-trip the same way.

`build_kernel_meta` runs at compile time and stores metadata on
`fn.__kernel_meta__`. `emit_trampoline_cpp` and `emit_trampoline_decls` write the
native pack lifecycle symbols and field accessors into each kernel translation
unit.

#### Technical terms:
- TypeSchema: recursive description of one parameter or return type (`kind`,
  `fields`, `inner`, and related keys) used by marshal and codegen.
- KernelMeta: full compile record for one kernel (`symbol`, params, return,
  layouts, and trampoline symbol names).
- pack: per-job C++ args struct named `{symbol}__args`; holds `a0`, `a1`, ...,
  optional `ret`, and `__shared_host`.
- symbol: export prefix of a compiled kernel (for example `move`); starts every
  generated accessor name.
- prefix: field segment in accessor names (`a0`, `a0_x`, `ret`, `a0_elem`).
- trampoline accessor: small exported C function that reads or writes one pack
  field (`move__set_a0_x`, `move__a0_resize`, and similar).
- pass_as: how a parameter is passed into the real kernel call (`value`, `ref`,
  `ptr`, `tbuffer`, `sync`, or `shared`).
- layouts: map of Threadable type names to their full `TypeSchema` for cycle-safe
  accessor emission.
- is_ref: when true on a Threadable schema, marshal looks up the full layout in
  `meta["schemas"]` instead of inline fields.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

from .types import (
    PyBool,
    PyCThreadsInternalType,
    PyDict,
    PyFloat,
    PyInt,
    PyList,
    PyString,
    PyShared,
    PyTBuffer,
    PyThreadable,
    SYNC_INTERNAL_NAMES,
    TBUFFER_INTERNAL_NAMES,
    hint_to_pytype,
    is_sync_pytype,
    is_shared_pytype,
    is_tbuffer_pytype,
)

# Populated by build_kernel_meta(); cleared when CompileSession.compile() runs.
KERNELS: dict = {}


@dataclass
class TypeSchema:
    """
    Recursive compile-time type layout for one marshal or codegen slot.

    Serialized through `to_dict()` into `fn.__kernel_meta__["schemas"]` and
    per-parameter `schema` entries for marshal.
    """

    kind: str  # int, float, bool, str, threadable, list, dict, tbuffer, or sync
    cpp_type: str
    # threadable-only fields
    type_name: str | None = None
    fields: list[tuple[str, TypeSchema]] = field(default_factory=list)
    # When true, marshal resolves the full layout from meta["schemas"][type_name].
    is_ref: bool = False
    # list-only field
    inner: TypeSchema | None = None
    # dict-only fields
    key: TypeSchema | None = None
    value: TypeSchema | None = None

    def to_dict(self, *, _seen: frozenset[str] | None = None) -> dict[str, Any]:
        """
        Serialize this schema to a JSON-friendly dict for `__kernel_meta__`.

        Threadable cycles are broken by emitting `is_ref: true` and empty
        `fields` when a type is seen again on the recursion stack.

        #### Args:
        - _seen: frozenset[str] = Threadable type names already being serialized

        #### Returns
        - dict = schema node understood by marshal and tests
        """
        seen = _seen or frozenset()
        d: dict[str, Any] = {
            "kind": self.kind,
            "cpp_type": self.cpp_type,
            "is_ref": self.is_ref,
        }
        if self.kind == "threadable":
            d["type_name"] = self.type_name
            if self.is_ref or (self.type_name and self.type_name in seen):
                d["fields"] = []
                d["is_ref"] = True
                return d
            next_seen = seen | ({self.type_name} if self.type_name else set())
            d["fields"] = [
                {"name": n, "schema": s.to_dict(_seen=next_seen)}
                for n, s in self.fields
            ]
        elif self.kind == "list":
            d["inner"] = (
                self.inner.to_dict(_seen=seen) if self.inner else None
            )
        elif self.kind == "dict":
            d["key"] = self.key.to_dict(_seen=seen) if self.key else None
            d["value"] = (
                self.value.to_dict(_seen=seen) if self.value else None
            )
        elif self.kind == "tbuffer":
            d["inner"] = (
                self.inner.to_dict(_seen=seen) if self.inner else None
            )
        elif self.kind == "sync":
            d["type_name"] = self.type_name
        return d


@dataclass
class ParamMeta:
    """
    One kernel parameter: Python name, pass mode, and recursive `TypeSchema`.
    """

    name: str
    pass_as: str  # value, ref, ptr, tbuffer, sync, or shared
    schema: TypeSchema

    @property
    def kind(self) -> str:
        return self.schema.kind

    @property
    def cpp_type(self) -> str:
        return self.schema.cpp_type

    @property
    def fields(self) -> list:
        """
        Flat field list for older tests and binder snippets.

        Only exposes primitive Threadable fields as simple name/kind objects.

        #### Returns
        - list = lightweight stand-ins with `name` and `kind` attributes
        """
        return [
            type("F", (), {"name": n, "kind": s.kind})()
            for n, s in self.schema.fields
        ]

    @property
    def list_inner(self) -> str | None:
        """Return the inner primitive kind when this param is a list, else None."""
        if self.schema.kind == "list" and self.schema.inner:
            return self.schema.inner.kind
        return None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this parameter for `fn.__kernel_meta__["params"]`.

        Includes both nested `schema` and legacy flat keys for compatibility.

        #### Returns
        - dict = parameter metadata consumed by marshal and module.cpp
        """
        return {
            "name": self.name,
            "pass_as": self.pass_as,
            "kind": self.schema.kind,
            "cpp_type": self.schema.cpp_type,
            "schema": self.schema.to_dict(),
            # Legacy flat keys kept for older binder snippets and tests.
            "fields": [
                {"name": n, "kind": s.kind, "schema": s.to_dict()}
                for n, s in self.schema.fields
            ],
            "list_inner": self.list_inner,
        }


@dataclass
class KernelMeta:
    """
    Compile-time record for one `@Thread` kernel.

    Holds trampoline symbol names, parameter metadata, return layout, and
    Threadable layouts used when emitting accessors into the kernel DLL.
    """

    symbol: str
    call_symbol: str
    args_new_symbol: str
    args_free_symbol: str
    params: list[ParamMeta]
    return_schema: TypeSchema | None = None
    return_pass_as: str | None = None
    is_method: bool = False
    layouts: dict[str, TypeSchema] = field(default_factory=dict)

    @property
    def return_kind(self) -> str:
        """Return schema kind, or `void` when the kernel has no return value."""
        return self.return_schema.kind if self.return_schema else "void"

    @property
    def return_cpp_type(self) -> str | None:
        """Return C++ type name from the return schema, if any."""
        return self.return_schema.cpp_type if self.return_schema else None

    @property
    def return_fields(self) -> list:
        """
        Flat Threadable return fields for legacy metadata consumers.

        #### Returns
        - list = empty when return is not a Threadable; else name/kind stand-ins
        """
        if not self.return_schema or self.return_schema.kind != "threadable":
            return []
        return [
            type("F", (), {"name": n, "kind": s.kind})()
            for n, s in self.return_schema.fields
        ]

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize kernel metadata for `fn.__kernel_meta__` and the native binder.

        #### Returns
        - dict = symbol names, params, return layout, and legacy flat return keys
        """
        return {
            "symbol": self.symbol,
            "call_symbol": self.call_symbol,
            "args_new_symbol": self.args_new_symbol,
            "args_free_symbol": self.args_free_symbol,
            "params": [p.to_dict() for p in self.params],
            "return_schema": (
                self.return_schema.to_dict() if self.return_schema else None
            ),
            "return_pass_as": self.return_pass_as,
            "return_kind": self.return_kind,
            "return_cpp_type": self.return_cpp_type,
            "return_fields": [
                {"name": n, "kind": s.kind, "schema": s.to_dict()}
                for n, s in (
                    self.return_schema.fields
                    if self.return_schema and self.return_schema.kind == "threadable"
                    else []
                )
            ],
            "is_method": self.is_method,
        }


def _primitive_cpp(kind: str) -> str:
    """
    Map a marshal primitive kind to its C++ type spelling.

    #### Args:
    - kind: str = one of `int`, `float`, `bool`, or `str`

    #### Returns
    - str = C++ type used in generated pack fields and accessors
    """
    return {
        "int": "int",
        "float": "double",
        "bool": "bool",
        "str": "std::string",
    }[kind]


def _tbuffer_inner_schema(py_type: PyCThreadsInternalType) -> TypeSchema:
    """
    Build the element `TypeSchema` for fixed `cthreads.sync.TBuffer*` classes.

    These are concrete buffer typedefs (not generic `TBuffer[T]`) with a known
    element layout at compile time.

    #### Args:
    - py_type: PyCThreadsInternalType = internal TBuffer class descriptor

    #### Returns
    - TypeSchema = element layout stored inside the tbuffer param schema

    #### Raises
    - TypeError = internal TBuffer name is not recognized
    """
    fixed: dict[str, TypeSchema] = {
        "TBufferF64": TypeSchema("float", "double"),
        "TBufferI64": TypeSchema("int", "int"),
        "TBufferBool": TypeSchema("bool", "bool"),
        "TBufferStr": TypeSchema("str", "std::string"),
        "TBufferListF64": TypeSchema(
            "list", "std::vector<double>", inner=TypeSchema("float", "double")
        ),
        "TBufferDictStrF64": TypeSchema(
            "dict",
            "std::unordered_map<std::string, double>",
            key=TypeSchema("str", "std::string"),
            value=TypeSchema("float", "double"),
        ),
        "TBufferObj": TypeSchema("str", "py::object"),
    }
    inner = fixed.get(py_type.name)
    if inner is None:
        raise TypeError(f"unknown fixed TBuffer type {py_type.name!r}")
    return inner


def hint_to_schema(
    hint: Any,
    types: dict[str, type],
    *,
    _completed: dict[str, TypeSchema] | None = None,
    _stack: frozenset[str] | None = None,
) -> TypeSchema:
    """
    Convert one Python type hint into a recursive `TypeSchema`.

    Registers each `@Threadable` class in `types` so marshal can reconstruct
    Python objects on unpack. Detects recursive Threadable graphs and marks
    repeated nodes with `is_ref` so serialization and accessor emission stay
    finite.

    #### Args:
    - hint: Any = annotated Python type from a kernel signature
    - types: dict[str, type] = output map of Threadable name to Python class
    - _completed: dict[str, TypeSchema] | None = layouts built so far in this walk
    - _stack: frozenset[str] | None = Threadable names currently being expanded

    #### Returns
    - TypeSchema = layout for marshal and trampoline emission

    #### Raises
    - TypeError = hint is unsupported or invalid for kernel dispatch

    #### Technical terms:
    - TypeSchema: recursive compile-time layout (see module docstring).
    - is_ref: marks a forward reference to a Threadable already on the stack.
    """
    completed = _completed if _completed is not None else {}
    stack = _stack or frozenset()
    py_type = hint_to_pytype(hint)

    if isinstance(py_type, PyInt):
        return TypeSchema("int", "int")
    if isinstance(py_type, PyFloat):
        return TypeSchema("float", "double")
    if isinstance(py_type, PyBool):
        return TypeSchema("bool", "bool")
    if isinstance(py_type, PyString):
        return TypeSchema("str", "std::string")

    if isinstance(py_type, PyList):
        from typing import get_args

        (inner_hint,) = get_args(hint)
        inner = hint_to_schema(
            inner_hint, types, _completed=completed, _stack=stack
        )
        return TypeSchema("list", f"std::vector<{inner.cpp_type}>", inner=inner)

    if isinstance(py_type, PyDict):
        from typing import get_args

        key_hint, val_hint = get_args(hint)
        key = hint_to_schema(
            key_hint, types, _completed=completed, _stack=stack
        )
        if key.kind not in ("str", "int"):
            raise TypeError(
                f"dict key type {key.kind!r} not supported for dispatch "
                "(use str or int)"
            )
        val = hint_to_schema(
            val_hint, types, _completed=completed, _stack=stack
        )
        return TypeSchema(
            "dict",
            f"std::unordered_map<{key.cpp_type}, {val.cpp_type}>",
            key=key,
            value=val,
        )

    if isinstance(py_type, PyThreadable):
        if not isinstance(hint, type) or not getattr(hint, "__threadable", False):
            raise TypeError(f"expected @Threadable class, got {hint!r}")
        name = hint.__name__
        types[name] = hint
        if name in completed:
            # Reuse the layout object; mark as ref when still on the recursion stack.
            return TypeSchema(
                "threadable",
                name,
                type_name=name,
                fields=completed[name].fields,
                is_ref=name in stack,
            )
        if name in stack:
            return TypeSchema(
                "threadable", name, type_name=name, fields=[], is_ref=True
            )

        nested_stack = stack | {name}
        schema = TypeSchema("threadable", name, type_name=name, fields=[])
        completed[name] = schema
        fields: list[tuple[str, TypeSchema]] = []
        for fname, fhint in get_type_hints(hint).items():
            fields.append(
                (
                    fname,
                    hint_to_schema(
                        fhint,
                        types,
                        _completed=completed,
                        _stack=nested_stack,
                    ),
                )
            )
        schema.fields = fields
        return schema

    if isinstance(py_type, PyShared):
        from typing import get_args

        inner_hint = getattr(hint, "__cthreads_shared_inner__", None)
        if inner_hint is None:
            args = get_args(hint)
            inner_hint = args[0] if args else None
        if inner_hint is None:
            raise TypeError("Shared[...] missing inner type for schema")
        return hint_to_schema(
            inner_hint, types, _completed=completed, _stack=stack
        )

    if isinstance(py_type, PyTBuffer):
        from typing import get_args

        inner_hint = getattr(hint, "__cthreads_tbuffer_inner__", None)
        if inner_hint is None:
            args = get_args(hint)
            inner_hint = args[0] if args else None
        if inner_hint is None:
            raise TypeError("TBuffer[...] missing inner type for schema")
        inner = hint_to_schema(
            inner_hint, types, _completed=completed, _stack=stack
        )
        return TypeSchema(
            "tbuffer",
            f"cthreads::sync::tripple_buffer<{inner.cpp_type}>",
            inner=inner,
        )

    if (
        isinstance(py_type, PyCThreadsInternalType)
        and py_type.name in SYNC_INTERNAL_NAMES
    ):
        return TypeSchema(
            "sync",
            py_type.cpp_name,
            type_name=py_type.name,
        )

    if isinstance(py_type, PyCThreadsInternalType) and py_type.name in TBUFFER_INTERNAL_NAMES:
        inner = _tbuffer_inner_schema(py_type)
        return TypeSchema(
            "tbuffer",
            py_type.cpp_name,
            inner=inner,
        )

    raise TypeError(f"unsupported dispatch type {hint!r}")


def _param_from_hint(
    name: str,
    hint: Any,
    *,
    pass_as: str,
    types: dict[str, type],
    completed: dict[str, TypeSchema] | None = None,
) -> ParamMeta:
    """
    Build one `ParamMeta` from a parameter name and type hint.

    #### Args:
    - name: str = Python parameter name
    - hint: Any = annotated type for the parameter
    - pass_as: str = marshal and call binding mode for this parameter
    - types: dict[str, type] = Threadable registry updated by `hint_to_schema`
    - completed: dict[str, TypeSchema] | None = shared layout cache for recursion

    #### Returns
    - ParamMeta = parameter record attached to `KernelMeta.params`
    """
    schema = hint_to_schema(hint, types, _completed=completed)
    return ParamMeta(name=name, pass_as=pass_as, schema=schema)


def build_kernel_meta(
    fn,
    *,
    symbol: str,
    owner_name: str | None = None,
    owner_cls: type | None = None,
) -> KernelMeta:
    """
    Build and attach compile-time metadata for one `@Thread` kernel function.

    Inspects annotations, chooses `pass_as` for each parameter, builds return
    layout, stores the result in `KERNELS`, and sets `fn.__kernel_meta__` for
    marshal and the native binder.

    #### Args:
    - fn: function = compiled kernel Python wrapper with type hints
    - symbol: str = export prefix for generated C symbols
    - owner_name: str | None = Threadable class name when `fn` is a method
    - owner_cls: type | None = explicit Threadable class for method kernels

    #### Returns
    - KernelMeta = full metadata record for this kernel

    #### Raises
    - TypeError = missing annotations, unknown owner class, or bad types

    #### Technical terms:
    - pass_as: controls pack storage and how `emit_trampoline_cpp` calls the kernel.
    - layouts: Threadable schemas collected in `completed` for accessor emission.
    - symbol: becomes `{symbol}__call`, `{symbol}__args_new`, and accessor prefixes.
    """
    hints = get_type_hints(fn)
    params: list[ParamMeta] = []
    types: dict[str, type] = {}
    completed: dict[str, TypeSchema] = {}

    names = list(fn.__code__.co_varnames[: fn.__code__.co_argcount])
    if owner_name and names and names[0] == "self":
        # Method kernels bind the Threadable instance as an opaque pointer parameter.
        cls = owner_cls
        if cls is None:
            from .frontend.Registry import REGISTRY

            cls = REGISTRY.threadables.get(owner_name)
        if cls is None:
            raise TypeError(
                f"Cannot resolve Threadable class {owner_name!r} for method {symbol}"
            )
        schema = hint_to_schema(cls, types, _completed=completed)
        params.append(ParamMeta(name="self", pass_as="ptr", schema=schema))
        names = names[1:]

    for name in names:
        if name not in hints:
            raise TypeError(
                f"kernel {symbol}: parameter {name!r} needs a type annotation"
            )
        hint = hints[name]
        py_type = hint_to_pytype(hint)
        # pass_as tells marshal and the call trampoline how to store and bind the arg.
        if is_tbuffer_pytype(py_type):
            pass_as = "tbuffer"
        elif is_sync_pytype(py_type):
            pass_as = "sync"
        elif is_shared_pytype(py_type):
            pass_as = "shared"
        elif isinstance(py_type, (PyThreadable, PyList, PyDict)):
            pass_as = "ref"
        else:
            pass_as = "value"
        params.append(
            _param_from_hint(
                name, hint, pass_as=pass_as, types=types, completed=completed
            )
        )

    ret_hint = hints.get("return", None)
    return_schema: TypeSchema | None = None
    return_pass_as: str | None = None
    if ret_hint is not None and ret_hint is not type(None):
        ret_py = hint_to_pytype(ret_hint)
        return_schema = hint_to_schema(
            ret_hint, types, _completed=completed
        )
        if is_shared_pytype(ret_py):
            return_pass_as = "shared"

    meta = KernelMeta(
        symbol=symbol,
        call_symbol=f"{symbol}__call",
        args_new_symbol=f"{symbol}__args_new",
        args_free_symbol=f"{symbol}__args_free",
        params=params,
        return_schema=return_schema,
        return_pass_as=return_pass_as,
        is_method=owner_name is not None,
        layouts=dict(completed),
    )
    KERNELS[symbol] = meta
    meta_dict = meta.to_dict()
    meta_dict["types"] = types
    meta_dict["schemas"] = {
        name: sch.to_dict() for name, sch in completed.items()
    }
    if return_schema and return_schema.kind == "threadable" and return_schema.type_name:
        meta_dict["return_cls"] = types.get(return_schema.type_name)
    if return_pass_as:
        meta_dict["return_pass_as"] = return_pass_as
        meta_dict["return_shared_name"] = "__return__"
    fn.__kernel_meta__ = meta_dict
    fn.__kernel_symbol__ = symbol
    return meta


# --- Trampoline emission: pack struct, accessors, and __call wrapper ---


def _cpp_prim(kind: str) -> str:
    """Alias for `_primitive_cpp`; used by accessor emission helpers."""
    return _primitive_cpp(kind)


def _emit_prim_accessors(
    lines: list[str],
    *,
    symbol: str,
    struct: str,  # Reserved; pack struct name is carried through other helpers.
    prefix: str,
    expr: str,
    kind: str,
    extra_params: str,
    extra_args_use: str,  # Reserved for future path suffix wiring.
) -> None:
    """
    Append C++ set/get trampoline accessors for one primitive pack field.

    Emits `{symbol}__set_{prefix}` and `{symbol}__get_{prefix}` (plus length
    helpers for strings). Nested list or dict slots add index and key parameters
    through `extra_params`.

    #### Args:
    - lines: list[str] = output C++ source lines mutated in place
    - symbol: str = kernel export prefix
    - struct: str = pack struct name (unused here; kept for uniform call shape)
    - prefix: str = field segment in accessor names (`a0`, `a0_x`, `ret`, ...)
    - expr: str = C++ lvalue path into the pack (for example `a->a0.x`)
    - kind: str = primitive kind
    - extra_params: str = extra formal parameters (for example `, size_t i0`)
    - extra_args_use: str = reserved; path args are already in `extra_params`

    #### Technical terms:
    - trampoline accessor: exported getter or setter called by marshal (see module docstring).
    - prefix: parameter or nested field segment in generated symbol names.
    """
    cpp_t = _cpp_prim(kind)
    if kind == "str":
        lines.append(
            f"CTHREADS_API void {symbol}__set_{prefix}(void* p{extra_params}, const char* v) {{"
        )
        lines.append(f"    {expr} = v;")
        lines.append("}")
        lines.append(
            f"CTHREADS_API size_t {symbol}__{prefix}_len(void* p{extra_params}) {{"
        )
        lines.append(f"    return {expr}.size();")
        lines.append("}")
        lines.append(
            f"CTHREADS_API void {symbol}__get_{prefix}(void* p{extra_params}, char* buf, size_t n) {{"
        )
        lines.append(
            f"    if (n == 0) return; "
            f"std::strncpy(buf, {expr}.c_str(), n - 1); buf[n - 1] = '\\0';"
        )
        lines.append("}")
    else:
        lines.append(
            f"CTHREADS_API void {symbol}__set_{prefix}(void* p{extra_params}, {cpp_t} v) {{"
        )
        lines.append(f"    {expr} = v;")
        lines.append("}")
        lines.append(
            f"CTHREADS_API void {symbol}__get_{prefix}(void* p{extra_params}, {cpp_t}* v) {{"
        )
        lines.append(f"    *v = {expr};")
        lines.append("}")


def _emit_schema_accessors_fixed(
    lines: list[str],
    *,
    symbol: str,
    struct: str,
    prefix: str,
    expr: str,
    schema: TypeSchema,
    index_params: list[str],
    key_params: list[tuple[str, str]],
    layouts: dict[str, TypeSchema],
    ancestors: frozenset[str] = frozenset(),
) -> None:
    """
    Recursively emit trampoline accessors for one `TypeSchema` subtree.

    Walks threadables, lists, dicts, tbuffers, and sync params and appends the
    matching C++ getter and setter symbols to `lines`. Skips Threadable branches
    that would recurse forever through `ancestors`.

    #### Args:
    - lines: list[str] = output C++ source lines mutated in place
    - symbol: str = kernel export prefix
    - struct: str = pack struct name (for example `move__args`)
    - prefix: str = accessor field segment (`a0`, `a0_elem`, `ret`, ...)
    - expr: str = C++ lvalue into the pack for this node
    - schema: TypeSchema = layout node to emit accessors for
    - index_params: list[str] = list index formal parameter names (`i0`, `i1`, ...)
    - key_params: list[tuple[str, str]] = dict key formals as `(name, cpp_type)` pairs
    - layouts: dict[str, TypeSchema] = full Threadable layouts for cycle-safe walks
    - ancestors: frozenset[str] = Threadable type names already open on this path

    #### Raises
    - TypeError = schema kind cannot be emitted

    #### Technical terms:
    - trampoline accessor: generated C API marshal uses to read or write the pack.
    - prefix: grows with nesting (`a0_x`, `a0_elem`, `a0_at`, `a0_ival`, ...).
    - layouts: supplies full Threadable fields when `schema.is_ref` is set.
    """
    extra = "".join(f", size_t {i}" for i in index_params)
    extra += "".join(f", {ty} {name}" for name, ty in key_params)

    if schema.kind in ("int", "float", "bool", "str"):
        _emit_prim_accessors(
            lines,
            symbol=symbol,
            struct=struct,
            prefix=prefix,
            expr=expr,
            kind=schema.kind,
            extra_params=extra,
            extra_args_use="",
        )
        return

    if schema.kind == "threadable":
        layout = layouts.get(schema.type_name or "", schema)
        fields = layout.fields or schema.fields
        next_anc = ancestors | ({schema.type_name} if schema.type_name else set())
        for fname, fschema in fields:
            # Skip list-of-self and direct self fields that would recurse accessors forever.
            if (
                fschema.kind == "list"
                and fschema.inner
                and fschema.inner.kind == "threadable"
                and fschema.inner.type_name in ancestors
            ):
                continue
            if (
                fschema.kind == "threadable"
                and fschema.type_name in ancestors
            ):
                continue
            _emit_schema_accessors_fixed(
                lines,
                symbol=symbol,
                struct=struct,
                prefix=f"{prefix}_{fname}",
                expr=f"{expr}.{fname}",
                schema=fschema,
                index_params=index_params,
                key_params=key_params,
                layouts=layouts,
                ancestors=next_anc,
            )
        return

    if schema.kind == "list":
        assert schema.inner is not None
        # List accessors: resize and size at this level, then emit element accessors.
        lines.append(
            f"CTHREADS_API void {symbol}__{prefix}_resize(void* p{extra}, size_t n) {{"
        )
        lines.append(f"    {expr}.resize(n);")
        lines.append("}")
        lines.append(
            f"CTHREADS_API void {symbol}__{prefix}_size(void* p{extra}, size_t* n) {{"
        )
        lines.append(f"    *n = {expr}.size();")
        lines.append("}")
        next_i = f"i{len(index_params)}"
        _emit_schema_accessors_fixed(
            lines,
            symbol=symbol,
            struct=struct,
            prefix=f"{prefix}_elem",
            expr=f"{expr}[{next_i}]",
            schema=schema.inner,
            index_params=index_params + [next_i],
            key_params=key_params,
            layouts=layouts,
            ancestors=ancestors,
        )
        return

    if schema.kind == "dict":
        assert schema.key is not None and schema.value is not None
        key_kind = schema.key.kind
        key_cpp = "const char*" if key_kind == "str" else "int"
        key_name = f"k{len(key_params)}"

        # Dict accessors: clear and size, then per-key insert or nested ensure_key path.
        lines.append(
            f"CTHREADS_API void {symbol}__{prefix}_clear(void* p{extra}) {{"
        )
        lines.append(f"    {expr}.clear();")
        lines.append("}")
        lines.append(
            f"CTHREADS_API void {symbol}__{prefix}_size(void* p{extra}, size_t* n) {{"
        )
        lines.append(f"    *n = {expr}.size();")
        lines.append("}")

        pack_keys = key_params + [(key_name, key_cpp)]
        pack_extra = "".join(f", size_t {i}" for i in index_params)
        pack_extra += "".join(f", {ty} {name}" for name, ty in pack_keys)

        if schema.value.kind in ("int", "float", "bool", "str"):
            if schema.value.kind == "str":
                lines.append(
                    f"CTHREADS_API void {symbol}__{prefix}_insert(void* p{pack_extra}, const char* v) {{"
                )
                lines.append(f"    {expr}[{key_name}] = v;")
                lines.append("}")
            else:
                vt = _cpp_prim(schema.value.kind)
                lines.append(
                    f"CTHREADS_API void {symbol}__{prefix}_insert(void* p{pack_extra}, {vt} v) {{"
                )
                lines.append(f"    {expr}[{key_name}] = v;")
                lines.append("}")
        else:
            lines.append(
                f"CTHREADS_API void {symbol}__{prefix}_ensure_key(void* p{pack_extra}) {{"
            )
            lines.append(f"    {expr}[{key_name}];")
            lines.append("}")
            _emit_schema_accessors_fixed(
                lines,
                symbol=symbol,
                struct=struct,
                prefix=f"{prefix}_at",
                expr=f"{expr}[{key_name}]",
                schema=schema.value,
                index_params=index_params,
                key_params=pack_keys,
                layouts=layouts,
                ancestors=ancestors,
            )

        next_i = f"i{len(index_params)}"
        # Unpack path: iterate native map by index and read key plus value accessors.
        unpack_indices = index_params + [next_i]
        unpack_extra = "".join(f", size_t {i}" for i in unpack_indices)
        unpack_extra += "".join(f", {ty} {name}" for name, ty in key_params)

        if key_kind == "str":
            lines.append(
                f"CTHREADS_API void {symbol}__{prefix}_key_at(void* p{unpack_extra}, char* buf, size_t n) {{"
            )
            lines.append(f"    auto it = {expr}.begin();")
            lines.append(f"    std::advance(it, {next_i});")
            lines.append(
                "    if (n == 0) return; "
                "std::strncpy(buf, it->first.c_str(), n - 1); buf[n - 1] = '\\0';"
            )
            lines.append("}")
        else:
            lines.append(
                f"CTHREADS_API void {symbol}__{prefix}_key_at(void* p{unpack_extra}, int* k) {{"
            )
            lines.append(f"    auto it = {expr}.begin();")
            lines.append(f"    std::advance(it, {next_i});")
            lines.append("    *k = it->first;")
            lines.append("}")

        ival_expr = (
            f"std::next(({expr}).begin(), "
            f"static_cast<std::ptrdiff_t>({next_i}))->second"
        )
        _emit_schema_accessors_fixed(
            lines,
            symbol=symbol,
            struct=struct,
            prefix=f"{prefix}_ival",
            expr=ival_expr,
            schema=schema.value,
            index_params=unpack_indices,
            key_params=key_params,
            layouts=layouts,
            ancestors=ancestors,
        )
        return

    if schema.kind == "tbuffer":
        # TBuffer params store a host pointer; marshal sets it through set_{prefix}_ptr.
        lines.append(
            f"CTHREADS_API void {symbol}__set_{prefix}_ptr(void* p{extra}, void* buf) {{"
        )
        lines.append(
            f"    static_cast<{struct}*>(p)->{prefix} = static_cast<{schema.cpp_type}*>(buf);"
        )
        lines.append("}")
        return

    if schema.kind == "sync":
        # Sync params store a host Lock or Event pointer the same way as TBuffer.
        lines.append(
            f"CTHREADS_API void {symbol}__set_{prefix}_ptr(void* p{extra}, void* buf) {{"
        )
        lines.append(
            f"    static_cast<{struct}*>(p)->{prefix} = static_cast<{schema.cpp_type}*>(buf);"
        )
        lines.append("}")
        return

    raise TypeError(f"cannot emit accessors for kind {schema.kind!r}")


def emit_trampoline_cpp(meta: KernelMeta, real_call: str) -> str:
    """
    Generate C++ pack struct, trampoline accessors, and `__call` for one kernel.

    Produces the body appended to a kernel `.cpp` file: `{symbol}__args`, lifecycle
    helpers, per-parameter accessors, optional shared promote/demote helpers, and
    the call wrapper that invokes the real compiled kernel function.

    #### Args:
    - meta: KernelMeta = compile record from `build_kernel_meta`
    - real_call: str = C++ name of the compiled kernel function to invoke

    #### Returns
    - str = C++ source fragment including required `#include` lines

    #### Technical terms:
    - pack: `{symbol}__args` struct emitted at the top of the fragment.
    - trampoline accessor: set/get/resize symbols marshal calls at runtime.
    - pass_as: controls pack member type, pointer storage, and `__call` arguments.
    """
    lines: list[str] = []
    struct = f"{meta.symbol}__args"
    needs_cstring = False
    needs_iterator = False
    needs_cstddef = True

    lines.append(f"struct {struct} {{")
    lines.append("    cthreads::SharedHost* __shared_host = nullptr;")
    for i, p in enumerate(meta.params):
        # TBuffer and sync params are pointer slots; everything else is stored by value.
        if p.schema.kind == "tbuffer" or p.schema.kind == "sync":
            lines.append(f"    {p.schema.cpp_type}* a{i};")
        else:
            lines.append(f"    {p.cpp_type} a{i};")
    if meta.return_schema is not None:
        lines.append(f"    {meta.return_schema.cpp_type} ret;")
    lines.append("};")
    lines.append("")

    lines.append(f"CTHREADS_API void* {meta.args_new_symbol}() {{")
    lines.append(f"    return new {struct}();")
    lines.append("}")
    lines.append("")
    lines.append(f"CTHREADS_API void {meta.args_free_symbol}(void* p) {{")
    lines.append(f"    delete static_cast<{struct}*>(p);")
    lines.append("}")
    lines.append("")
    lines.append(
        f"CTHREADS_API void {meta.symbol}__set_shared_host(void* p, void* host) {{"
    )
    lines.append(
        f"    static_cast<{struct}*>(p)->__shared_host = "
        "static_cast<cthreads::SharedHost*>(host);"
    )
    lines.append("}")
    lines.append("")

    def walk_needs(schema: TypeSchema | None) -> None:
        """Track which standard headers accessor bodies will require."""
        nonlocal needs_cstring, needs_iterator
        if schema is None:
            return
        if schema.kind == "str":
            needs_cstring = True
        if schema.kind == "dict":
            needs_iterator = True
            needs_cstring = needs_cstring or (
                schema.key is not None and schema.key.kind == "str"
            )
            walk_needs(schema.key)
            walk_needs(schema.value)
        elif schema.kind == "list":
            walk_needs(schema.inner)
        elif schema.kind == "tbuffer":
            walk_needs(schema.inner)
        elif schema.kind == "threadable":
            for _, fs in schema.fields:
                walk_needs(fs)

    for i, p in enumerate(meta.params):
        walk_needs(p.schema)
        _emit_schema_accessors_fixed(
            lines,
            symbol=meta.symbol,
            struct=struct,
            prefix=f"a{i}",
            expr=f"static_cast<{struct}*>(p)->a{i}",
            schema=p.schema,
            index_params=[],
            key_params=[],
            layouts=meta.layouts,
        )
        lines.append("")
        if p.pass_as == "shared":
            # Shared params: promote pack slot into SharedHost at spawn, demote before readback.
            lines.append(
                f"CTHREADS_API void {meta.symbol}__promote_a{i}_shared("
                f"void* p, cthreads::SharedHost* h) {{"
            )
            lines.append(f"    if (!h) return;")
            lines.append(
                f"    h->set(\"{p.name}\", "
                f"std::move(static_cast<{struct}*>(p)->a{i}));"
            )
            lines.append("}")
            lines.append("")
            lines.append(
                f"CTHREADS_API void {meta.symbol}__demote_a{i}_shared("
                f"void* p, cthreads::SharedHost* h) {{"
            )
            lines.append(f"    if (!h || !h->contains(\"{p.name}\")) return;")
            lines.append(
                f"    static_cast<{struct}*>(p)->a{i} = "
                f"h->get<{p.cpp_type}>(\"{p.name}\");"
            )
            lines.append("}")
            lines.append("")

    if meta.return_schema is not None:
        walk_needs(meta.return_schema)
        # Return slot uses prefix `ret`; getters support unpack_return after the kernel runs.
        _emit_schema_accessors_fixed(
            lines,
            symbol=meta.symbol,
            struct=struct,
            prefix="ret",
            expr=f"static_cast<{struct}*>(p)->ret",
            schema=meta.return_schema,
            index_params=[],
            key_params=[],
            layouts=meta.layouts,
        )
        lines.append("")

    if meta.return_pass_as == "shared" and meta.return_schema is not None:
        lines.append(
            f"CTHREADS_API void {meta.symbol}__demote_return_shared("
            f"void* p, cthreads::SharedHost* h) {{"
        )
        lines.append("    if (!h || !h->contains(\"__return__\")) return;")
        lines.append(
            f"    static_cast<{struct}*>(p)->ret = "
            f"h->get<{meta.return_schema.cpp_type}>(\"__return__\");"
        )
        lines.append("}")
        lines.append("")

    call_args: list[str] = []
    for i, p in enumerate(meta.params):
        slot = f"a->a{i}"
        # Map pass_as to the C++ expression passed into the real kernel call.
        if p.pass_as == "ptr":
            call_args.append(f"&{slot}")
        elif p.pass_as in ("tbuffer", "sync"):
            call_args.append(f"*{slot}")
        elif p.pass_as == "shared":
            call_args.append(
                f"a->__shared_host->get<{p.cpp_type}>(\"{p.name}\")"
            )
        elif p.pass_as == "ref":
            call_args.append(slot)
        else:
            call_args.append(slot)
    args_csv = ", ".join(call_args)
    lines.append(f"CTHREADS_API void {meta.call_symbol}(void* p) {{")
    lines.append(f"    auto* a = static_cast<{struct}*>(p);")
    if meta.return_schema is None:
        lines.append(f"    {real_call}({args_csv});")
    elif meta.return_pass_as == "shared":
        lines.append(f"    a->ret = {real_call}({args_csv});")
        lines.append(
            "    if (a->__shared_host) {"
        )
        lines.append(
            f"        a->__shared_host->replace(\"__return__\", std::move(a->ret));"
        )
        lines.append("    }")
    else:
        lines.append(f"    a->ret = {real_call}({args_csv});")
    lines.append("}")
    lines.append("")

    body = "\n".join(lines)
    headers = "#include \"shared_host.hpp\"\n"
    if needs_cstddef:
        headers += "#include <cstddef>\n"
    if needs_cstring:
        headers += "#include <cstring>\n"
    if needs_iterator:
        headers += "#include <iterator>\n"
    return headers + body


def emit_trampoline_decls(meta: KernelMeta) -> str:
    """
    Emit header declarations for pack lifecycle and call trampolines.

    Accessor declarations are omitted here; they are linked from the `.cpp`
    translation unit. This list is the stable surface `module.cpp` relies on
    for allocate, free, call, and SharedHost wiring.

    #### Args:
    - meta: KernelMeta = compile record from `build_kernel_meta`

    #### Returns
    - str = C declaration block for the kernel `.hpp` file
    """
    lines = [
        f"CTHREADS_API void* {meta.args_new_symbol}();",
        f"CTHREADS_API void {meta.args_free_symbol}(void* p);",
        f"CTHREADS_API void {meta.call_symbol}(void* p);",
        f"CTHREADS_API void {meta.symbol}__set_shared_host(void* p, void* host);",
    ]
    return "\n".join(lines) + "\n"


def collect_tbuffer_threadables() -> set[str]:
    """
    Collect Threadable type names used as `TBuffer[Threadable]` kernel parameters.

    Used when emitting the host-side TBuffer allocator runtime.

    #### Returns
    - set[str] = Threadable class names referenced by compiled kernels
    """
    names: set[str] = set()
    for meta in KERNELS.values():
        for p in meta.params:
            if p.schema.kind != "tbuffer" or p.schema.inner is None:
                continue
            if p.schema.inner.kind == "threadable" and p.schema.inner.type_name:
                names.add(p.schema.inner.type_name)
    return names


def emit_tbuffer_runtime_files(
    threadable_names: set[str],
    thread_dir: Path,
) -> tuple[str, str]:
    """
    Emit `cthreads_tbuffer.hpp` and `cthreads_tbuffer.cpp` host allocator sources.

    The native extension passes opaque `void*` handles; kernels cast them to
    `tripple_buffer<T>&`. One switch case is generated per Threadable name.

    #### Args:
    - threadable_names: set[str] = Threadable types referenced by kernel params
    - thread_dir: Path = `__Thread__` output directory for include paths

    #### Returns
    - tuple[str, str] = `(hpp_source, cpp_source)` written by `write_tbuffer_runtime`

    #### Raises
    - RuntimeError = a named Threadable has no compiled ThreadableUnit

    #### Technical terms:
    - TBuffer: host-side triple buffer wrapper passed into kernels by pointer.
    - tripple_buffer: C++ buffer type stored behind the opaque host pointer.
    """
    from .frontend.Registry import REGISTRY

    thread_dir = Path(thread_dir).resolve()
    sorted_names = sorted(threadable_names)

    includes: list[str] = []
    create_cases: list[str] = []
    destroy_cases: list[str] = []
    generation_cases: list[str] = []
    read_cases: list[str] = []
    free_read_cases: list[str] = []

    for name in sorted_names:
        unit = REGISTRY.threadable_units.get(name)
        if unit is None:
            raise RuntimeError(
                f"cthreads: TBuffer[{name}] used in a kernel but {name!r} "
                "has no ThreadableUnit (compile @Threadable first)"
            )
        hpp = unit.hpp_path.resolve()
        rel = os.path.relpath(hpp, thread_dir).replace("\\", "/")
        includes.append(f'#include "{rel}"')
        create_cases.append(
            f'    if (name == "{name}") {{\n'
            f"        return new cthreads::sync::tripple_buffer<{name}>(capacity);\n"
            f"    }}"
        )
        destroy_cases.append(
            f'    if (name == "{name}") {{\n'
            f"        delete static_cast<cthreads::sync::tripple_buffer<{name}>*>(ptr);\n"
            f"        return;\n"
            f"    }}"
        )
        generation_cases.append(
            f'    if (name == "{name}") {{\n'
            f"        return static_cast<cthreads::sync::tripple_buffer<{name}>*>(ptr)->generation();\n"
            f"    }}"
        )
        read_cases.append(
            f'    if (name == "{name}") {{\n'
            f"        return static_cast<cthreads::sync::tripple_buffer<{name}>*>(ptr)->get_read_cpy();\n"
            f"    }}"
        )
        free_read_cases.append(
            f'    if (name == "{name}") {{\n'
            f"        delete[] static_cast<{name}*>(copy);\n"
            f"        return;\n"
            f"    }}"
        )

    hpp_src = (
        "#pragma once\n\n"
        '#include "cthreads_export.hpp"\n\n'
        "CTHREADS_API void* cthreads_create_tbuffer(const char* type_name, int capacity);\n"
        "CTHREADS_API void cthreads_destroy_tbuffer(const char* type_name, void* ptr);\n"
        "CTHREADS_API int cthreads_tbuffer_generation(const char* type_name, void* ptr);\n"
        "CTHREADS_API void* cthreads_tbuffer_read_copy(const char* type_name, void* ptr);\n"
        "CTHREADS_API void cthreads_tbuffer_free_read_copy(const char* type_name, void* copy);\n"
    )

    cpp_src = (
        '#include "cthreads_tbuffer.hpp"\n\n'
        '#include "sync/t_buffer.hpp"\n\n'
        + "\n".join(includes)
        + "\n\n#include <string>\n\n"
        "CTHREADS_API void* cthreads_create_tbuffer(const char* type_name, int capacity) {\n"
        "    if (!type_name || capacity <= 0) {\n"
        "        return nullptr;\n"
        "    }\n"
        "    const std::string name(type_name);\n"
        + "\n".join(create_cases)
        + "\n    return nullptr;\n"
        "}\n\n"
        "CTHREADS_API void cthreads_destroy_tbuffer(const char* type_name, void* ptr) {\n"
        "    if (!type_name || !ptr) {\n"
        "        return;\n"
        "    }\n"
        "    const std::string name(type_name);\n"
        + "\n".join(destroy_cases)
        + "\n}\n\n"
        "CTHREADS_API int cthreads_tbuffer_generation(const char* type_name, void* ptr) {\n"
        "    if (!type_name || !ptr) {\n"
        "        return 0;\n"
        "    }\n"
        "    const std::string name(type_name);\n"
        + "\n".join(generation_cases)
        + "\n    return 0;\n"
        "}\n\n"
        "CTHREADS_API void* cthreads_tbuffer_read_copy(const char* type_name, void* ptr) {\n"
        "    if (!type_name || !ptr) {\n"
        "        return nullptr;\n"
        "    }\n"
        "    const std::string name(type_name);\n"
        + "\n".join(read_cases)
        + "\n    return nullptr;\n"
        "}\n\n"
        "CTHREADS_API void cthreads_tbuffer_free_read_copy(const char* type_name, void* copy) {\n"
        "    if (!type_name || !copy) {\n"
        "        return;\n"
        "    }\n"
        "    const std::string name(type_name);\n"
        + "\n".join(free_read_cases)
        + "\n}\n"
    )

    return hpp_src, cpp_src


def write_tbuffer_runtime(root: Path) -> bool:
    """
    Write or remove `__Thread__/cthreads_tbuffer.*` for the current project.

    When no kernel uses `TBuffer[Threadable]`, existing allocator files are
    deleted so stale symbols are not linked.

    #### Args:
    - root: Path = project root containing the `__Thread__` codegen folder

    #### Returns
    - bool = True when allocator sources exist after this call, else False
    """
    from .io import write_if_changed

    root = Path(root).resolve()
    thread_dir = root / "__Thread__"
    thread_dir.mkdir(parents=True, exist_ok=True)
    hpp_path = thread_dir / "cthreads_tbuffer.hpp"
    cpp_path = thread_dir / "cthreads_tbuffer.cpp"

    names = collect_tbuffer_threadables()
    if not names:
        for path in (hpp_path, cpp_path):
            if path.is_file():
                path.unlink()
        return False

    hpp_src, cpp_src = emit_tbuffer_runtime_files(names, thread_dir)
    write_if_changed(hpp_path, hpp_src)
    write_if_changed(cpp_path, cpp_src)
    return True


# Back-compat for tests that still import FieldMeta.
@dataclass
class FieldMeta:
    """Legacy flat Threadable field descriptor (name and primitive kind only)."""

    name: str
    kind: str
