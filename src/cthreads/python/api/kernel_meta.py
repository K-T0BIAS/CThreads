"""
Kernel metadata captured at compile time for dispatch.

The codegen already knows each @Thread parameter's C++ type; we store a
compact schema on the function / in KERNELS so runtime can bind args/kwargs
and call generated DLL trampolines without re-deriving types.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, get_type_hints

from .CONFIG import KERNELS
from .pyTypes import (
    PyBool,
    PyFloat,
    PyInt,
    PyList,
    PyString,
    PyThreadable,
    hint_to_pytype,
)


@dataclass
class FieldMeta:
    name: str
    kind: str  # "int" | "float" | "bool" | "str"


@dataclass
class ParamMeta:
    name: str
    kind: str  # "int" | "float" | "bool" | "str" | "threadable" | "list"
    cpp_type: str
    # How the trampoline passes this slot into the real kernel.
    pass_as: str  # "value" | "ref" | "ptr"
    fields: list[FieldMeta] = field(default_factory=list)
    list_inner: str | None = None  # kind of list element


@dataclass
class KernelMeta:
    symbol: str
    call_symbol: str
    args_new_symbol: str
    args_free_symbol: str
    params: list[ParamMeta]
    return_kind: str  # "void" | "int" | "float" | "bool" | "str"
    # For methods: first param is the Threadable receiver.
    is_method: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _kind_of_pytype(py_type) -> str:
    if isinstance(py_type, PyInt):
        return "int"
    if isinstance(py_type, PyFloat):
        return "float"
    if isinstance(py_type, PyBool):
        return "bool"
    if isinstance(py_type, PyString):
        return "str"
    if isinstance(py_type, PyThreadable):
        return "threadable"
    if isinstance(py_type, PyList):
        return "list"
    raise TypeError(f"unsupported kernel param type: {type(py_type).__name__}")


def _fields_for_threadable(cls: type) -> list[FieldMeta]:
    out: list[FieldMeta] = []
    for fname, hint in get_type_hints(cls).items():
        py_type = hint_to_pytype(hint)
        kind = _kind_of_pytype(py_type)
        if kind not in ("int", "float", "bool", "str"):
            raise TypeError(
                f"Threadable {cls.__name__}.{fname}: "
                f"only primitive fields are supported for dispatch packing"
            )
        out.append(FieldMeta(name=fname, kind=kind))
    return out


def _param_from_hint(name: str, hint: Any, *, pass_as: str) -> ParamMeta:
    py_type = hint_to_pytype(hint)
    kind = _kind_of_pytype(py_type)
    fields: list[FieldMeta] = []
    list_inner = None
    if isinstance(py_type, PyThreadable):
        # Resolve class from REGISTRY/STORE name via hint itself.
        fields = _fields_for_threadable(hint)
    elif isinstance(py_type, PyList):
        list_inner = _kind_of_pytype(py_type.inner_type)
        if list_inner not in ("int", "float", "bool"):
            raise TypeError(
                f"list element type {list_inner!r} not supported for dispatch yet"
            )
    return ParamMeta(
        name=name,
        kind=kind,
        cpp_type=py_type.cpp_name,
        pass_as=pass_as,
        fields=fields,
        list_inner=list_inner,
    )


def build_kernel_meta(
    fn,
    *,
    symbol: str,
    owner_name: str | None = None,
    owner_cls: type | None = None,
) -> KernelMeta:
    hints = get_type_hints(fn)
    params: list[ParamMeta] = []

    names = list(fn.__code__.co_varnames[: fn.__code__.co_argcount])
    if owner_name and names and names[0] == "self":
        cls = owner_cls
        if cls is None:
            from .CONFIG import REGISTRY

            cls = REGISTRY.threadables.get(owner_name)
        if cls is None:
            raise TypeError(
                f"Cannot resolve Threadable class {owner_name!r} for method {symbol}"
            )
        fields = _fields_for_threadable(cls)
        params.append(
            ParamMeta(
                name="self",
                kind="threadable",
                cpp_type=owner_name,
                pass_as="ptr",
                fields=fields,
            )
        )
        names = names[1:]

    for name in names:
        if name not in hints:
            raise TypeError(f"kernel {symbol}: parameter {name!r} needs a type annotation")
        hint = hints[name]
        py_type = hint_to_pytype(hint)
        if isinstance(py_type, PyThreadable):
            pass_as = "ref"
        else:
            pass_as = "value"
        params.append(_param_from_hint(name, hint, pass_as=pass_as))

    ret_hint = hints.get("return", None)
    if ret_hint is None or ret_hint is type(None):
        return_kind = "void"
    else:
        return_kind = _kind_of_pytype(hint_to_pytype(ret_hint))
        if return_kind not in ("int", "float", "bool", "void"):
            raise TypeError(f"kernel {symbol}: return type {return_kind!r} not supported yet")

    meta = KernelMeta(
        symbol=symbol,
        call_symbol=f"{symbol}__call",
        args_new_symbol=f"{symbol}__args_new",
        args_free_symbol=f"{symbol}__args_free",
        params=params,
        return_kind=return_kind,
        is_method=owner_name is not None,
    )
    KERNELS[symbol] = meta
    fn.__kernel_meta__ = meta.to_dict()
    fn.__kernel_symbol__ = symbol
    return meta


def emit_trampoline_cpp(meta: KernelMeta, real_call: str) -> str:
    """
    Emit args struct + alloc/free/setters/getters + __call for one kernel.

    real_call: expression like `move` or `Particle_step` (the exported C symbol).
    """
    lines: list[str] = []
    struct = f"{meta.symbol}__args"

    # --- struct ---
    lines.append(f"struct {struct} {{")
    for i, p in enumerate(meta.params):
        lines.append(f"    {p.cpp_type} a{i};")
    if meta.return_kind == "int":
        lines.append("    int ret;")
    elif meta.return_kind == "float":
        lines.append("    double ret;")
    elif meta.return_kind == "bool":
        lines.append("    bool ret;")
    lines.append("};")
    lines.append("")

    # --- new / free ---
    lines.append(f"CTHREADS_API void* {meta.args_new_symbol}() {{")
    lines.append(f"    return new {struct}();")
    lines.append("}")
    lines.append("")
    lines.append(f"CTHREADS_API void {meta.args_free_symbol}(void* p) {{")
    lines.append(f"    delete static_cast<{struct}*>(p);")
    lines.append("}")
    lines.append("")

    # --- setters / getters per param ---
    for i, p in enumerate(meta.params):
        if p.kind == "int":
            lines.append(
                f"CTHREADS_API void {meta.symbol}__set_a{i}(void* p, int v) {{"
            )
            lines.append(f"    static_cast<{struct}*>(p)->a{i} = v;")
            lines.append("}")
        elif p.kind == "float":
            lines.append(
                f"CTHREADS_API void {meta.symbol}__set_a{i}(void* p, double v) {{"
            )
            lines.append(f"    static_cast<{struct}*>(p)->a{i} = v;")
            lines.append("}")
        elif p.kind == "bool":
            lines.append(
                f"CTHREADS_API void {meta.symbol}__set_a{i}(void* p, bool v) {{"
            )
            lines.append(f"    static_cast<{struct}*>(p)->a{i} = v;")
            lines.append("}")
        elif p.kind == "threadable":
            field_args = ", ".join(
                ("int" if f.kind == "int" else "double" if f.kind == "float" else "bool" if f.kind == "bool" else "const char*")
                + f" {f.name}"
                for f in p.fields
            )
            lines.append(
                f"CTHREADS_API void {meta.symbol}__set_a{i}(void* p, {field_args}) {{"
            )
            lines.append(f"    auto& o = static_cast<{struct}*>(p)->a{i};")
            for f in p.fields:
                lines.append(f"    o.{f.name} = {f.name};")
            lines.append("}")
            # writeback getter
            out_args = ", ".join(
                ("int*" if f.kind == "int" else "double*" if f.kind == "float" else "bool*" if f.kind == "bool" else "char*")
                + f" {f.name}"
                for f in p.fields
            )
            lines.append(
                f"CTHREADS_API void {meta.symbol}__get_a{i}(void* p, {out_args}) {{"
            )
            lines.append(f"    auto& o = static_cast<{struct}*>(p)->a{i};")
            for f in p.fields:
                lines.append(f"    *{f.name} = o.{f.name};")
            lines.append("}")
        elif p.kind == "list":
            inner_cpp = {"int": "int", "float": "double", "bool": "bool"}[p.list_inner]
            lines.append(
                f"CTHREADS_API void {meta.symbol}__set_a{i}(void* p, const {inner_cpp}* data, size_t n) {{"
            )
            lines.append(
                f"    static_cast<{struct}*>(p)->a{i}.assign(data, data + n);"
            )
            lines.append("}")
        else:
            raise TypeError(f"cannot emit setter for kind {p.kind!r}")
        lines.append("")

    # --- return getter ---
    if meta.return_kind == "int":
        lines.append(f"CTHREADS_API int {meta.symbol}__get_ret(void* p) {{")
        lines.append(f"    return static_cast<{struct}*>(p)->ret;")
        lines.append("}")
        lines.append("")
    elif meta.return_kind == "float":
        lines.append(f"CTHREADS_API double {meta.symbol}__get_ret(void* p) {{")
        lines.append(f"    return static_cast<{struct}*>(p)->ret;")
        lines.append("}")
        lines.append("")
    elif meta.return_kind == "bool":
        lines.append(f"CTHREADS_API bool {meta.symbol}__get_ret(void* p) {{")
        lines.append(f"    return static_cast<{struct}*>(p)->ret;")
        lines.append("}")
        lines.append("")

    # --- call ---
    call_args: list[str] = []
    for i, p in enumerate(meta.params):
        slot = f"a->a{i}"
        if p.pass_as == "ptr":
            call_args.append(f"&{slot}")
        elif p.pass_as == "ref":
            call_args.append(slot)  # Particle& binds to lvalue
        else:
            call_args.append(slot)
    args_csv = ", ".join(call_args)
    lines.append(f"CTHREADS_API void {meta.call_symbol}(void* p) {{")
    lines.append(f"    auto* a = static_cast<{struct}*>(p);")
    if meta.return_kind == "void":
        lines.append(f"    {real_call}({args_csv});")
    else:
        lines.append(f"    a->ret = {real_call}({args_csv});")
    lines.append("}")
    lines.append("")

    body = "\n".join(lines)
    # list setters need size_t
    if any(p.kind == "list" for p in meta.params):
        body = "#include <cstddef>\n" + body
    return body


def emit_trampoline_decls(meta: KernelMeta) -> str:
    """Declarations for the hpp (optional; defs alone work with extern C linkage)."""
    lines = [
        f"CTHREADS_API void* {meta.args_new_symbol}();",
        f"CTHREADS_API void {meta.args_free_symbol}(void* p);",
        f"CTHREADS_API void {meta.call_symbol}(void* p);",
    ]
    return "\n".join(lines) + "\n"
