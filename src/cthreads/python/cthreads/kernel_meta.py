"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Kernel metadata + trampoline emission for dispatch.

Params and returns share one recursive TypeSchema so pack / writeback /
job.result() all use the same shape (primitives, Threadable, list, dict).
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
    PyTBuffer,
    PyThreadable,
    TBUFFER_INTERNAL_NAMES,
    hint_to_pytype,
    is_tbuffer_pytype,
)

# Filled by build_kernel_meta(); cleared by CompileSession.compile()
KERNELS: dict = {}


@dataclass
class TypeSchema:
    kind: str  # int|float|bool|str|threadable|list|dict|tbuffer
    cpp_type: str
    # threadable
    type_name: str | None = None
    fields: list[tuple[str, TypeSchema]] = field(default_factory=list)
    # True => look up full layout in meta["schemas"][type_name] (cycle break)
    is_ref: bool = False
    # list
    inner: TypeSchema | None = None
    # dict
    key: TypeSchema | None = None
    value: TypeSchema | None = None

    def to_dict(self, *, _seen: frozenset[str] | None = None) -> dict[str, Any]:
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
        return d


@dataclass
class ParamMeta:
    name: str
    pass_as: str  # value | ref | ptr | tbuffer
    schema: TypeSchema

    @property
    def kind(self) -> str:
        return self.schema.kind

    @property
    def cpp_type(self) -> str:
        return self.schema.cpp_type

    @property
    def fields(self) -> list:
        """Compat: flat FieldMeta-like objects for primitive threadable fields only."""
        return [
            type("F", (), {"name": n, "kind": s.kind})()
            for n, s in self.schema.fields
        ]

    @property
    def list_inner(self) -> str | None:
        if self.schema.kind == "list" and self.schema.inner:
            return self.schema.inner.kind
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pass_as": self.pass_as,
            "kind": self.schema.kind,
            "cpp_type": self.schema.cpp_type,
            "schema": self.schema.to_dict(),
            # compat keys used by older binder snippets / tests
            "fields": [
                {"name": n, "kind": s.kind, "schema": s.to_dict()}
                for n, s in self.schema.fields
            ],
            "list_inner": self.list_inner,
        }


@dataclass
class KernelMeta:
    symbol: str
    call_symbol: str
    args_new_symbol: str
    args_free_symbol: str
    params: list[ParamMeta]
    return_schema: TypeSchema | None = None
    is_method: bool = False
    layouts: dict[str, TypeSchema] = field(default_factory=dict)

    @property
    def return_kind(self) -> str:
        return self.return_schema.kind if self.return_schema else "void"

    @property
    def return_cpp_type(self) -> str | None:
        return self.return_schema.cpp_type if self.return_schema else None

    @property
    def return_fields(self) -> list:
        if not self.return_schema or self.return_schema.kind != "threadable":
            return []
        return [
            type("F", (), {"name": n, "kind": s.kind})()
            for n, s in self.return_schema.fields
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "call_symbol": self.call_symbol,
            "args_new_symbol": self.args_new_symbol,
            "args_free_symbol": self.args_free_symbol,
            "params": [p.to_dict() for p in self.params],
            "return_schema": (
                self.return_schema.to_dict() if self.return_schema else None
            ),
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
    return {
        "int": "int",
        "float": "double",
        "bool": "bool",
        "str": "std::string",
    }[kind]


def _tbuffer_inner_schema(py_type: PyCThreadsInternalType) -> TypeSchema:
    """Build element schema for fixed ``cthreads.sync.TBuffer*`` classes."""
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
    """Build a recursive TypeSchema; register Threadable classes in types."""
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
            # Same layout object (shared) — emit as ref in to_dict via is_ref copy
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
    schema = hint_to_schema(hint, types, _completed=completed)
    return ParamMeta(name=name, pass_as=pass_as, schema=schema)


def build_kernel_meta(
    fn,
    *,
    symbol: str,
    owner_name: str | None = None,
    owner_cls: type | None = None,
) -> KernelMeta:
    hints = get_type_hints(fn)
    params: list[ParamMeta] = []
    types: dict[str, type] = {}
    completed: dict[str, TypeSchema] = {}

    names = list(fn.__code__.co_varnames[: fn.__code__.co_argcount])
    if owner_name and names and names[0] == "self":
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
        if is_tbuffer_pytype(py_type):
            pass_as = "tbuffer"
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
    if ret_hint is not None and ret_hint is not type(None):
        return_schema = hint_to_schema(
            ret_hint, types, _completed=completed
        )

    meta = KernelMeta(
        symbol=symbol,
        call_symbol=f"{symbol}__call",
        args_new_symbol=f"{symbol}__args_new",
        args_free_symbol=f"{symbol}__args_free",
        params=params,
        return_schema=return_schema,
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
    fn.__kernel_meta__ = meta_dict
    fn.__kernel_symbol__ = symbol
    return meta


# --- trampoline emission -------------------------------------------------


def _cpp_prim(kind: str) -> str:
    return _primitive_cpp(kind)


def _emit_prim_accessors(
    lines: list[str],
    *,
    symbol: str,
    struct: str, # unused
    prefix: str,
    expr: str,
    kind: str,
    extra_params: str,
    extra_args_use: str, # unused
) -> None:
    """extra_params like ', size_t i0' or ', const char* k0'."""
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
    """Emit set/get accessors; skip Threadable fields that would recurse forever."""
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
            # One level of self-ref containers is enough (e.g. Boid.flock elems).
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
    lines: list[str] = []
    struct = f"{meta.symbol}__args"
    needs_cstring = False
    needs_iterator = False
    needs_cstddef = True

    lines.append(f"struct {struct} {{")
    for i, p in enumerate(meta.params):
        if p.schema.kind == "tbuffer":
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

    def walk_needs(schema: TypeSchema | None) -> None:
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

    if meta.return_schema is not None:
        walk_needs(meta.return_schema)
        # returns only need getters; still emit full accessors (set unused)
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

    call_args: list[str] = []
    for i, p in enumerate(meta.params):
        slot = f"a->a{i}"
        if p.pass_as == "ptr":
            call_args.append(f"&{slot}")
        elif p.pass_as == "tbuffer":
            call_args.append(f"*{slot}")
        elif p.pass_as == "ref":
            call_args.append(slot)
        else:
            call_args.append(slot)
    args_csv = ", ".join(call_args)
    lines.append(f"CTHREADS_API void {meta.call_symbol}(void* p) {{")
    lines.append(f"    auto* a = static_cast<{struct}*>(p);")
    if meta.return_schema is None:
        lines.append(f"    {real_call}({args_csv});")
    else:
        lines.append(f"    a->ret = {real_call}({args_csv});")
    lines.append("}")
    lines.append("")

    body = "\n".join(lines)
    headers = ""
    if needs_cstddef:
        headers += "#include <cstddef>\n"
    if needs_cstring:
        headers += "#include <cstring>\n"
    if needs_iterator:
        headers += "#include <iterator>\n"
    return headers + body


def emit_trampoline_decls(meta: KernelMeta) -> str:
    lines = [
        f"CTHREADS_API void* {meta.args_new_symbol}();",
        f"CTHREADS_API void {meta.args_free_symbol}(void* p);",
        f"CTHREADS_API void {meta.call_symbol}(void* p);",
    ]
    return "\n".join(lines) + "\n"


def collect_tbuffer_threadables() -> set[str]:
    """Threadable type names used in ``TBuffer[Threadable]`` kernel params."""
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
    Emit ``cthreads_tbuffer.{hpp,cpp}`` for opaque host allocation.

    ``_ext`` passes ``void*``; kernels cast to ``tripple_buffer<T>&``.
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
    Write or remove ``__Thread__/cthreads_tbuffer.*`` for the project.

    Returns True if allocator sources exist after this call.
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


# Back-compat for tests that still import FieldMeta
@dataclass
class FieldMeta:
    name: str
    kind: str
