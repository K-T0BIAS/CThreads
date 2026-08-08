"""
Pack / unpack kernel args using the same TypeSchema as trampoline accessors.

Params and returns share this path so list[Threadable], nested Threadables,
list[base], and dict[str|int, T] all round-trip the same way.

Thread safety: every entry point takes an explicit pack pointer. There is no
process-global pack slot — concurrent Jobs may pack/writeback/unpack in
parallel (ctypes releases the GIL) without clobbering each other.

The kernel CDLL is cached once per path. Calls go through ``_call``, which
binds via CFUNCTYPE and never mutates shared ``fn.argtypes`` / ``fn.restype``
(those races + repeated LoadLibrary were a Windows hard-hang under Jobs).
"""

from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Path:
    indices: list[int] = field(default_factory=list)
    # pack-time dict keys: ("str", value) | ("int", value)
    keys: list[tuple[str, Any]] = field(default_factory=list)


_lib_lock = threading.Lock()
_cached_lib: ctypes.CDLL | None = None
_cached_path: str | None = None


def _lib():
    global _cached_lib, _cached_path
    from cthreads import _ext

    path = _ext.kernel_path()
    if not path:
        raise RuntimeError("cthreads.marshal: kernel library not loaded")
    with _lib_lock:
        if _cached_lib is None or _cached_path != path:
            # One LoadLibrary per kernel path — concurrent fresh CDLL() can
            # deadlock on the Windows loader lock while Jobs join.
            _cached_lib = ctypes.CDLL(path)
            _cached_path = path
        return _cached_lib


def _fn(lib, name: str):
    try:
        return getattr(lib, name)
    except AttributeError as e:
        raise RuntimeError(f"cthreads.marshal: missing symbol {name!r}") from e


def _call(fn, restype, argtypes: list, *args):
    """Call a CDLL symbol without mutating shared argtypes/restype."""
    addr = ctypes.cast(fn, ctypes.c_void_p).value
    if not addr:
        raise RuntimeError("cthreads.marshal: null function pointer")
    return ctypes.CFUNCTYPE(restype, *argtypes)(addr)(*args)


def _pack_c(pack: int | ctypes.c_void_p) -> ctypes.c_void_p:
    if isinstance(pack, ctypes.c_void_p):
        if not pack.value:
            raise RuntimeError("cthreads.marshal: null pack pointer")
        return pack
    if not pack:
        raise RuntimeError("cthreads.marshal: null pack pointer")
    return ctypes.c_void_p(int(pack))


def _extra(path: _Path) -> list:
    out: list = []
    for i in path.indices:
        out.append(ctypes.c_size_t(i))
    for kind, val in path.keys:
        if kind == "str":
            out.append(val.encode("utf-8") if isinstance(val, str) else val)
        else:
            out.append(ctypes.c_int(int(val)))
    return out


def _ctype_extras(path: _Path) -> list:
    types: list = []
    for _ in path.indices:
        types.append(ctypes.c_size_t)
    for kind, _ in path.keys:
        types.append(ctypes.c_char_p if kind == "str" else ctypes.c_int)
    return types


def _set_prim(
    lib,
    symbol: str,
    prefix: str,
    path: _Path,
    kind: str,
    value: Any,
    pack: ctypes.c_void_p,
) -> None:
    name = f"{symbol}__set_{prefix}"
    fn = _fn(lib, name)
    base = [ctypes.c_void_p, *_ctype_extras(path)]
    extras = _extra(path)
    if kind == "float":
        _call(fn, None, base + [ctypes.c_double], pack, *extras, float(value))
    elif kind == "int":
        _call(fn, None, base + [ctypes.c_int], pack, *extras, int(value))
    elif kind == "bool":
        _call(fn, None, base + [ctypes.c_bool], pack, *extras, bool(value))
    elif kind == "str":
        _call(
            fn,
            None,
            base + [ctypes.c_char_p],
            pack,
            *extras,
            str(value).encode("utf-8"),
        )
    else:
        raise TypeError(f"bad primitive kind {kind!r}")


def _get_prim(
    lib,
    symbol: str,
    prefix: str,
    path: _Path,
    kind: str,
    pack: ctypes.c_void_p,
) -> Any:
    extras = _extra(path)
    path_types = [ctypes.c_void_p, *_ctype_extras(path)]
    if kind == "str":
        len_fn = _fn(lib, f"{symbol}__{prefix}_len")
        n = int(_call(len_fn, ctypes.c_size_t, path_types, pack, *extras))
        buf = ctypes.create_string_buffer(n + 1)
        get_fn = _fn(lib, f"{symbol}__get_{prefix}")
        _call(
            get_fn,
            None,
            path_types + [ctypes.c_char_p, ctypes.c_size_t],
            pack,
            *extras,
            buf,
            n + 1,
        )
        return buf.value.decode("utf-8")

    get_fn = _fn(lib, f"{symbol}__get_{prefix}")
    if kind == "float":
        out = ctypes.c_double()
        _call(
            get_fn,
            None,
            path_types + [ctypes.POINTER(ctypes.c_double)],
            pack,
            *extras,
            ctypes.byref(out),
        )
        return float(out.value)
    if kind == "int":
        out = ctypes.c_int()
        _call(
            get_fn,
            None,
            path_types + [ctypes.POINTER(ctypes.c_int)],
            pack,
            *extras,
            ctypes.byref(out),
        )
        return int(out.value)
    if kind == "bool":
        out = ctypes.c_bool()
        _call(
            get_fn,
            None,
            path_types + [ctypes.POINTER(ctypes.c_bool)],
            pack,
            *extras,
            ctypes.byref(out),
        )
        return bool(out.value)
    raise TypeError(f"bad primitive kind {kind!r}")


def _attr_get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def _attr_set(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def _resolve_schema(
    schema: dict[str, Any], schemas: dict[str, Any]
) -> dict[str, Any]:
    """Follow is_ref / empty threadable layouts via meta['schemas']."""
    if schema.get("kind") != "threadable":
        return schema
    if schema.get("is_ref") or not schema.get("fields"):
        name = schema.get("type_name")
        if name and name in schemas:
            full = dict(schemas[name])
            full["is_ref"] = False
            return full
    return schema


def pack_value(
    lib,
    symbol: str,
    prefix: str,
    schema: dict[str, Any],
    value: Any,
    path: _Path,
    pack: ctypes.c_void_p,
    *,
    schemas: dict[str, Any] | None = None,
) -> None:
    schemas = schemas or {}
    schema = _resolve_schema(schema, schemas)
    kind = schema["kind"]
    if kind in ("int", "float", "bool", "str"):
        _set_prim(lib, symbol, prefix, path, kind, value, pack)
        return

    if kind == "threadable":
        for f in schema["fields"]:
            child_schema = f["schema"]
            try:
                pack_value(
                    lib,
                    symbol,
                    f"{prefix}_{f['name']}",
                    child_schema,
                    _attr_get(value, f["name"]),
                    path,
                    pack,
                    schemas=schemas,
                )
            except RuntimeError as e:
                if "missing symbol" in str(e):
                    continue
                raise
        return

    if kind == "list":
        inner = schema["inner"]
        n = len(value)
        resize = _fn(lib, f"{symbol}__{prefix}_resize")
        _call(
            resize,
            None,
            [ctypes.c_void_p, *_ctype_extras(path), ctypes.c_size_t],
            pack,
            *_extra(path),
            n,
        )
        for i, elem in enumerate(value):
            child = _Path(indices=path.indices + [i], keys=list(path.keys))
            pack_value(
                lib,
                symbol,
                f"{prefix}_elem",
                inner,
                elem,
                child,
                pack,
                schemas=schemas,
            )
        return

    if kind == "dict":
        key_schema = schema["key"]
        val_schema = schema["value"]
        clear = _fn(lib, f"{symbol}__{prefix}_clear")
        _call(
            clear,
            None,
            [ctypes.c_void_p, *_ctype_extras(path)],
            pack,
            *_extra(path),
        )

        key_kind = key_schema["kind"]
        for k, v in value.items():
            key_tag = ("str", str(k)) if key_kind == "str" else ("int", int(k))
            child = _Path(indices=list(path.indices), keys=path.keys + [key_tag])
            child_extras = _extra(child)
            if val_schema["kind"] in ("int", "float", "bool", "str"):
                insert = _fn(lib, f"{symbol}__{prefix}_insert")
                val_ctype = (
                    ctypes.c_char_p
                    if val_schema["kind"] == "str"
                    else ctypes.c_double
                    if val_schema["kind"] == "float"
                    else ctypes.c_int
                    if val_schema["kind"] == "int"
                    else ctypes.c_bool
                )
                atypes = [ctypes.c_void_p, *_ctype_extras(child), val_ctype]
                if val_schema["kind"] == "str":
                    _call(
                        insert,
                        None,
                        atypes,
                        pack,
                        *child_extras,
                        str(v).encode("utf-8"),
                    )
                elif val_schema["kind"] == "float":
                    _call(insert, None, atypes, pack, *child_extras, float(v))
                elif val_schema["kind"] == "int":
                    _call(insert, None, atypes, pack, *child_extras, int(v))
                else:
                    _call(insert, None, atypes, pack, *child_extras, bool(v))
            else:
                ensure = _fn(lib, f"{symbol}__{prefix}_ensure_key")
                _call(
                    ensure,
                    None,
                    [ctypes.c_void_p, *_ctype_extras(child)],
                    pack,
                    *child_extras,
                )
                pack_value(
                    lib,
                    symbol,
                    f"{prefix}_at",
                    val_schema,
                    v,
                    child,
                    pack,
                    schemas=schemas,
                )
        return

    raise TypeError(f"unsupported schema kind {kind!r}")


def unpack_value(
    lib,
    symbol: str,
    prefix: str,
    schema: dict[str, Any],
    path: _Path,
    pack: ctypes.c_void_p,
    *,
    types: dict[str, Any],
    schemas: dict[str, Any] | None = None,
    into: Any | None = None,
) -> Any:
    schemas = schemas or {}
    schema = _resolve_schema(schema, schemas)
    kind = schema["kind"]
    if kind in ("int", "float", "bool", "str"):
        return _get_prim(lib, symbol, prefix, path, kind, pack)

    if kind == "threadable":
        type_name = schema["type_name"]
        cls = types.get(type_name)
        if cls is None:
            raise RuntimeError(
                f"cthreads.marshal: no Python class registered for {type_name!r}"
            )
        obj = into if into is not None else cls()
        for f in schema["fields"]:
            try:
                child_val = unpack_value(
                    lib,
                    symbol,
                    f"{prefix}_{f['name']}",
                    f["schema"],
                    path,
                    pack,
                    types=types,
                    schemas=schemas,
                )
            except RuntimeError as e:
                if "missing symbol" in str(e):
                    continue
                raise
            _attr_set(obj, f["name"], child_val)
        return obj

    if kind == "list":
        inner = schema["inner"]
        size_fn = _fn(lib, f"{symbol}__{prefix}_size")
        n_out = ctypes.c_size_t()
        _call(
            size_fn,
            None,
            [
                ctypes.c_void_p,
                *_ctype_extras(path),
                ctypes.POINTER(ctypes.c_size_t),
            ],
            pack,
            *_extra(path),
            ctypes.byref(n_out),
        )
        n = int(n_out.value)
        out_list: list = into if isinstance(into, list) else []
        if isinstance(into, list):
            out_list.clear()
        for i in range(n):
            child = _Path(indices=path.indices + [i], keys=list(path.keys))
            out_list.append(
                unpack_value(
                    lib,
                    symbol,
                    f"{prefix}_elem",
                    inner,
                    child,
                    pack,
                    types=types,
                    schemas=schemas,
                )
            )
        return out_list

    if kind == "dict":
        key_schema = schema["key"]
        val_schema = schema["value"]
        size_fn = _fn(lib, f"{symbol}__{prefix}_size")
        n_out = ctypes.c_size_t()
        _call(
            size_fn,
            None,
            [
                ctypes.c_void_p,
                *_ctype_extras(path),
                ctypes.POINTER(ctypes.c_size_t),
            ],
            pack,
            *_extra(path),
            ctypes.byref(n_out),
        )
        n = int(n_out.value)
        out_dict: dict = into if isinstance(into, dict) else {}
        if isinstance(into, dict):
            out_dict.clear()
        for i in range(n):
            child = _Path(indices=path.indices + [i], keys=list(path.keys))
            child_extras = _extra(child)
            if key_schema["kind"] == "str":
                buf = ctypes.create_string_buffer(4096)
                key_at = _fn(lib, f"{symbol}__{prefix}_key_at")
                _call(
                    key_at,
                    None,
                    [
                        ctypes.c_void_p,
                        *_ctype_extras(child),
                        ctypes.c_char_p,
                        ctypes.c_size_t,
                    ],
                    pack,
                    *child_extras,
                    buf,
                    4096,
                )
                key = buf.value.decode("utf-8")
            else:
                k_out = ctypes.c_int()
                key_at = _fn(lib, f"{symbol}__{prefix}_key_at")
                _call(
                    key_at,
                    None,
                    [
                        ctypes.c_void_p,
                        *_ctype_extras(child),
                        ctypes.POINTER(ctypes.c_int),
                    ],
                    pack,
                    *child_extras,
                    ctypes.byref(k_out),
                )
                key = int(k_out.value)
            out_dict[key] = unpack_value(
                lib,
                symbol,
                f"{prefix}_ival",
                val_schema,
                child,
                pack,
                types=types,
                schemas=schemas,
            )
        return out_dict

    raise TypeError(f"unsupported schema kind {kind!r}")


def pack_params(
    symbol: str,
    params: list[dict],
    values: list[Any],
    pack_ptr: int,
    types: dict[str, Any] | None = None,
    schemas: dict[str, Any] | None = None,
) -> None:
    del types  # reserved for future use; schemas carry layouts
    lib = _lib()
    pack = _pack_c(pack_ptr)
    for i, (p, val) in enumerate(zip(params, values)):
        schema = p.get("schema") or _legacy_schema(p)
        pack_value(
            lib,
            symbol,
            f"a{i}",
            schema,
            val,
            _Path(),
            pack,
            schemas=schemas or {},
        )


def writeback_params(
    symbol: str,
    params: list[dict],
    values: list[Any],
    pack_ptr: int,
    types: dict[str, Any] | None = None,
    schemas: dict[str, Any] | None = None,
) -> None:
    lib = _lib()
    types = types or {}
    schemas = schemas or {}
    pack = _pack_c(pack_ptr)
    for i, (p, val) in enumerate(zip(params, values)):
        schema = p.get("schema") or _legacy_schema(p)
        if schema["kind"] not in ("threadable", "list", "dict"):
            continue
        if isinstance(val, dict) and schema["kind"] == "threadable":
            continue
        unpack_value(
            lib,
            symbol,
            f"a{i}",
            schema,
            _Path(),
            pack,
            types=types,
            schemas=schemas,
            into=val,
        )


def unpack_return(meta: dict[str, Any], pack_ptr: int) -> Any:
    schema = meta.get("return_schema")
    if not schema:
        kind = meta.get("return_kind", "void")
        if kind == "void":
            return None
        schema = {
            "kind": kind,
            "cpp_type": meta.get("return_cpp_type"),
            "type_name": None,
            "fields": meta.get("return_fields") or [],
        }
        if kind == "threadable":
            schema["type_name"] = meta.get("return_cpp_type")
            schema["fields"] = [
                {
                    "name": f["name"],
                    "schema": f.get("schema")
                    or {"kind": f["kind"], "cpp_type": "", "fields": []},
                }
                for f in (meta.get("return_fields") or [])
            ]

    if schema is None:
        return None

    lib = _lib()
    types = dict(meta.get("types") or {})
    schemas = dict(meta.get("schemas") or {})
    if meta.get("return_cls") is not None and schema.get("type_name"):
        types[schema["type_name"]] = meta["return_cls"]

    pack = _pack_c(pack_ptr)
    return unpack_value(
        lib,
        meta["symbol"],
        "ret",
        schema,
        _Path(),
        pack,
        types=types,
        schemas=schemas,
    )


def _legacy_schema(p: dict[str, Any]) -> dict[str, Any]:
    """Build a schema dict from older flat param meta."""
    kind = p["kind"]
    if kind in ("int", "float", "bool", "str"):
        return {"kind": kind, "cpp_type": p.get("cpp_type", "")}
    if kind == "list":
        inner_kind = p.get("list_inner") or "float"
        return {
            "kind": "list",
            "cpp_type": p.get("cpp_type", ""),
            "inner": {"kind": inner_kind, "cpp_type": ""},
        }
    if kind == "threadable":
        fields = []
        for f in p.get("fields") or []:
            if "schema" in f:
                fields.append({"name": f["name"], "schema": f["schema"]})
            else:
                fields.append(
                    {
                        "name": f["name"],
                        "schema": {"kind": f["kind"], "cpp_type": ""},
                    }
                )
        return {
            "kind": "threadable",
            "cpp_type": p.get("cpp_type", ""),
            "type_name": p.get("cpp_type"),
            "fields": fields,
        }
    raise TypeError(f"cannot legacy-upgrade kind {kind!r}")
