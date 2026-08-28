"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

This module moves data between Python values and the per-job C++ argument
struct (`pack`) that compiled `@Thread` kernels read and write. The native
`cthreads._ext` layer (`module.cpp`) calls into it at job spawn, after the
kernel finishes, and when the host requests a mid-run sync.

At spawn, `pack_params` walks each parameter's compile-time `TypeSchema` and
writes Python arguments into the pack through the kernel dynamic-link library
(DLL) generated trampoline accessors (`Fn__set_a0_*`, `Fn__a0_resize`, and so
on). If a parameter is `Shared[T]`, `promote_shared_to_host` copies the staged
pack slot into the cooperative `SharedHost` heap before the worker runs.

After the kernel mutates native memory, `writeback_job_state` pulls mutable
reference arguments back into the same Python objects the caller passed in.
It first runs `demote_shared_from_host` for any `Shared[T]` parameters (and
returns), then `writeback_params` for `Threadable`, `list`, and `dict`
arguments. `unpack_return` reads the `ret` field from the pack (or shared
host) and builds the Python return value.

Parameters and returns use the same recursive pack/unpack logic, so scalars,
`Threadable` instances, nested structures, `list[T]`, and `dict[str|int, T]`
all round-trip through one code path. Scalars are copied by value at spawn;
only reference-like types are written back.

Every public entry point takes an explicit pack pointer (and optionally a
`SharedHost` pointer). Marshal does not keep a process-global pack slot, so
concurrent jobs can pack, write back, and unpack in parallel while ctypes
releases the Global Interpreter Lock (GIL).

The kernel dynamic-link library (DLL) is loaded once per kernel path and
cached as a `CDLL` handle. Each native call goes through `_call`, which binds
the symbol with `CFUNCTYPE` for that invocation and never mutates shared
`fn.argtypes` or `fn.restype`, because doing so raced under concurrent jobs
and could deadlock the Windows loader when `LoadLibrary` was repeated.

#### Technical terms:
- pack: per-job C++ struct holding one copy of each kernel argument and the
  return value; Python fills it at spawn and reads it after the kernel runs.
- symbol: export prefix of a compiled `@Thread` function (for example `step`);
  every generated accessor name starts with it (`step__set_a0_x`).
- prefix: field segment inside accessor names (`a0` for the first parameter,
  `ret` for the return value, `a0_elem` for a list element field).
- trampoline accessor: C function emitted into the kernel DLL so Python can read
  or write one pack field without entering the kernel body. [more here: docs\guide\marshal_and_module.md]
- trampoline call: a single invocation of a trampoline accessor through `_call`.
- native accessor API: the full set of trampolines for one field (set, get,
  resize, clear, insert, and similar operations).
- TypeSchema: compile-time type description dict (`kind`, `fields`, `inner`,
  and so on) stored in kernel meta.
- key tag: in `_Path`, a tuple `("str", key)` or `("int", key)` that tells
  dict accessors which map entry to address.
- addressing trail: the `_Path` list indices and key tags passed as extra
  arguments after the pack pointer on nested accessor calls.
- SharedHost: cooperative heap used by `Shared[T]` values; separate from the
  per-job pack.
- staged pack slot: temporary pack field that holds a `Shared[T]` value before
  promote or after demote.
- promote: copy a `Shared[T]` value from its staged pack slot into SharedHost
  at spawn time.
- demote: copy a `Shared[T]` value from SharedHost back into its staged pack
  slot before read or writeback.
- writeback: update the caller's original Python objects from native pack memory
  after the kernel mutates them.
- kernel meta: compile output (`symbol`, `params`, schemas, return layout)
  passed from `module.cpp` into marshal.
"""
import ctypes
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Path:
    """
    Address one nested field inside a pack during recursive marshal.

    List elements append to `indices`. Dictionary entries append a key tag
    to `keys` so dict trampoline accessors know which map entry to use.

    #### Technical terms:
    - key tag: tuple `("str", value)` or `("int", value)` stored in `keys`;
      identifies one dictionary key when nested dict fields are addressed.
    - indices: list positions appended when addressing nested `list` elements
      inside the pack (index 0, then 1, and so on).
    """

    indices: list[int] = field(default_factory=list)
    keys: list[tuple[str, Any]] = field(default_factory=list)


_lib_lock = threading.Lock()  # Only one thread may load or refresh the cached kernel DLL.
_cached_lib: ctypes.CDLL | None = None
_cached_path: str | None = None


def _lib() -> ctypes.CDLL | None:
    """
    Return the cached kernel dynamic-link library (DLL) handle for marshal calls.

    The path comes from `cthreads._ext.kernel_path()`. The handle is loaded
    once per path and reused so parallel jobs do not call `LoadLibrary` again.

    #### Returns
    - ctypes.CDLL = loaded kernel library used to resolve trampoline accessor names

    #### Raises
    - RuntimeError = kernel path is missing or the library was never built

    #### Technical terms:
    - trampoline accessor: exported C function in the kernel DLL (see module docstring).
    """
    global _cached_lib, _cached_path
    import importlib

    _ext = importlib.import_module("cthreads._ext")
    path = _ext.kernel_path()
    if not path:
        raise RuntimeError("cthreads.marshal: kernel library not loaded")
    with _lib_lock:
        if _cached_lib is None or _cached_path != path:
            # Reuse one loaded DLL handle for all jobs. Loading it again from
            # many threads at once used to hang the process on Windows during
            # concurrent job join or sync.
            _cached_lib = ctypes.CDLL(path)
            _cached_path = path
        return _cached_lib


def _fn(lib, name: str):
    """
    Look up one exported trampoline accessor on the loaded kernel library.

    #### Args:
    - lib: CDLL = kernel library returned by `_lib`
    - name: str = full accessor export name (for example `step__set_a0_x`)

    #### Returns
    - CDLL function object = native accessor to invoke through `_call`

    #### Raises
    - RuntimeError = accessor name is not exported by the kernel build

    #### Technical terms:
    - trampoline accessor: C getter or setter emitted for one pack field (see
      module docstring).
    - symbol: the `step` part of `step__set_a0_x`; the compiled kernel name.
    """
    try:
        return getattr(lib, name)
    except AttributeError as e:
        raise RuntimeError(f"cthreads.marshal: missing symbol {name!r}") from e


def _call(fn, restype, argtypes: list, *args):
    """
    Perform one trampoline call into the kernel DLL.

    Every `pack_value` / `unpack_value` path reaches the native accessor API
    through this helper. Callers pass the exact `restype` and `argtypes` for
    that accessor on each invocation.

    Do not call `fn(*args)` after setting `fn.argtypes` / `fn.restype` on the
    shared `CDLL` object. Those attributes are process-wide on a loaded accessor,
    so concurrent jobs marshaling different kernels or signatures would overwrite each
    other's bindings and corrupt calls. Building a fresh `CFUNCTYPE` per call avoids
    that race (and was required to stop Windows hangs under parallel jobs).

    #### Args:
    - fn: CDLL symbol = accessor returned by `_fn`
    - restype: ctypes type or None = C return type (`None` means void)
    - argtypes: list = C argument types for this accessor, in order
    - *args: values passed through to the native call

    #### Returns
    - Any or None = native return value, or `None` when `restype` is void

    #### Raises
    - RuntimeError = accessor resolved to a null function pointer

    #### Technical terms:
    - trampoline call: one ctypes invocation of a generated pack accessor.
    - native accessor API: the set/get/resize family of trampolines for a field
      (see module docstring).
    """
    # Turn the CDLL wrapper into a raw address; we bind our own prototype below.
    addr = ctypes.cast(fn, ctypes.c_void_p).value
    if not addr:
        raise RuntimeError("cthreads.marshal: null function pointer")
    # Build a one-off callable with this call's restype and argtypes, then invoke it.
    # Nothing on the shared CDLL symbol is mutated, so parallel jobs stay safe.
    return ctypes.CFUNCTYPE(restype, *argtypes)(addr)(*args)


def _pack_c(pack: int | ctypes.c_void_p) -> ctypes.c_void_p:
    """
    Normalize a pack pointer passed from native code into a `c_void_p`.

    `module.cpp` passes the pack as an integer address. This helper accepts
    either that integer or an existing `c_void_p` and rejects null pointers.

    #### Args:
    - pack: int | c_void_p = native address of a per-job args struct

    #### Returns
    - c_void_p = non-null pointer suitable for trampoline calls

    #### Raises
    - RuntimeError = pointer is missing or zero

    #### Technical terms:
    - pack: per-job C++ args struct pointer (see module docstring).
    - trampoline call: native accessor invocation that reads or writes the pack.
    """
    if isinstance(pack, ctypes.c_void_p):
        if not pack.value:
            raise RuntimeError("cthreads.marshal: null pack pointer")
        return pack
    if not pack:
        raise RuntimeError("cthreads.marshal: null pack pointer")
    return ctypes.c_void_p(int(pack))


def _extra(path: _Path) -> list:
    """
    Build trailing path arguments for a trampoline call from a `_Path`.

    List indices become `c_size_t` values. Dictionary key tags become either a
    UTF-8 byte string or a signed integer, matching what the native accessor API
    expects after the pack pointer.

    #### Args:
    - path: _Path = addressing trail of list indices and dict key tags

    #### Returns
    - list = ctypes-compatible values appended after the pack pointer

    #### Technical terms:
    - trampoline call: native accessor invocation; extras follow the pack pointer.
    - native accessor API: generated getters and setters for nested list and dict
      fields (see module docstring).
    - key tag: `("str", key)` or `("int", key)` tuple from `_Path.keys`.
    - addressing trail: combined `indices` and `keys` that locate one nested slot.
    """
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
    """
    Build ctypes type objects that correspond to `_extra(path)`.

    Each list index adds `c_size_t`. Each dict key adds `c_char_p` for string
    keys or `c_int` for integer keys.

    #### Args:
    - path: _Path = nested list index and dict key trail

    #### Returns
    - list = ctypes types inserted into `_call` argtypes before value args

    #### Technical terms:
    - addressing trail: list indices and dict key tags from `_Path` (see
      module docstring).
    """
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
    """
    Write one scalar field into the pack through a generated set accessor.

    The accessor name is `{symbol}__set_{prefix}` with optional addressing
    trail arguments when `path` targets a nested list or dict slot.

    #### Args:
    - lib: CDLL = loaded kernel library
    - symbol: str = compiled kernel export prefix (for example `step`)
    - prefix: str = pack field segment in the accessor name (for example `a0`)
    - path: _Path = addressing trail for nested list or dict elements
    - kind: str = primitive kind (`int`, `float`, `bool`, or `str`)
    - value: Any = Python value to store
    - pack: c_void_p = per-job args struct pointer

    #### Raises
    - TypeError = `kind` is not a supported primitive
    - RuntimeError = accessor name is missing from the kernel build

    #### Technical terms:
    - symbol: compiled `@Thread` function name prefix (see module docstring).
    - prefix: parameter or return field segment such as `a0` or `a0_elem`.
    - addressing trail: `_Path` indices and key tags passed to the accessor.
    - trampoline accessor: generated `{symbol}__set_{prefix}` C function.
    """
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
    """
    Read one scalar field from the pack through a generated get accessor.

    String values are sized with `{symbol}__{prefix}_len` first, then copied
    into a buffer. Numeric and boolean values are read through an out-pointer.

    #### Args:
    - lib: CDLL = loaded kernel library
    - symbol: str = compiled kernel export prefix
    - prefix: str = pack field segment in the accessor name
    - path: _Path = addressing trail for nested list or dict elements
    - kind: str = primitive kind (`int`, `float`, `bool`, or `str`)
    - pack: c_void_p = per-job args struct pointer

    #### Returns
    - Any = decoded Python scalar

    #### Raises
    - TypeError = `kind` is not a supported primitive
    - RuntimeError = accessor name is missing from the kernel build

    #### Technical terms:
    - symbol: compiled `@Thread` function name prefix (see module docstring).
    - prefix: parameter or return field segment such as `a0` or `ret`.
    - addressing trail: `_Path` indices and key tags passed to the accessor.
    - trampoline accessor: generated `{symbol}__get_{prefix}` C function.
    """
    extras = _extra(path)
    path_types = [ctypes.c_void_p, *_ctype_extras(path)]
    if kind == "str":
        # Strings are variable-length in C++; query length, then copy into a buffer.
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
        # Fixed-size scalars are written through an out-pointer argument.
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
    """
    Read a field from a Threadable instance or a plain dict stand-in.

    #### Args:
    - obj: Any = object or dict passed as a Threadable argument
    - name: str = field name from compile-time schema

    #### Returns
    - Any = field value used for recursive packing

    #### Technical terms:
    - TypeSchema: compile-time field layout from kernel meta (see module docstring).
    """
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def _attr_set(obj: Any, name: str, value: Any) -> None:
    """
    Write a field back into a Threadable instance or a plain dict stand-in.

    #### Args:
    - obj: Any = live Python object receiving writeback
    - name: str = field name from compile-time schema
    - value: Any = unpacked native value to store

    #### Technical terms:
    - writeback: copy native pack memory into the caller's Python objects (see
      module docstring).
    """
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def _resolve_schema(
    schema: dict[str, Any], schemas: dict[str, Any]
) -> dict[str, Any]:
    """
    Expand a forward-reference Threadable schema using kernel meta schemas.

    Some compile-time layouts store only `type_name` and `is_ref`. This helper
    replaces those stubs with the full field list from `schemas`.

    #### Args:
    - schema: dict = schema node from param or return metadata
    - schemas: dict = `meta["schemas"]` lookup table from compile output

    #### Returns
    - dict = resolved schema ready for pack or unpack

    #### Technical terms:
    - TypeSchema: compile-time type description dict (see module docstring).
    - kernel meta: compile output containing the `schemas` lookup table.
    """
    if schema.get("kind") != "threadable":
        return schema
    if schema.get("is_ref") or not schema.get("fields"):
        name = schema.get("type_name")
        if name and name in schemas:
            full = dict(schemas[name])
            full["is_ref"] = False
            return full
    return schema


def _tbuffer_native_ptr(value: Any, inner_type_name: str | None = None) -> int:
    """
    Resolve a Python TBuffer host wrapper to its native pointer for packing.

    #### Args:
    - value: Any = TBuffer instance passed as a kernel argument
    - inner_type_name: str | None = Threadable type name when the buffer holds structs

    #### Returns
    - int = native address written into the pack pointer slot

    #### Technical terms:
    - pack: pointer slot field set by `{symbol}__set_{prefix}_ptr` trampoline.
    - TBuffer: host-side buffer wrapper whose C++ object address is stored in the pack.
    """
    from .sync.tbuffer_host import tbuffer_ptr

    return tbuffer_ptr(value, inner_type_name)


def _sync_native_ptr(value: Any) -> int:
    """
    Resolve a Python sync primitive to its native pointer for packing.

    Delegates to `cthreads._ext.sync_native_ptr`, which knows how each sync
    type exposes its underlying C++ object.

    #### Args:
    - value: Any = Lock, Event, or other supported sync host object

    #### Returns
    - int = native address written into the pack pointer slot

    #### Raises
    - RuntimeError = extension returned a null pointer

    #### Technical terms:
    - pack: pointer slot field set by `{symbol}__set_{prefix}_ptr` trampoline.
    - sync primitive: Lock, Event, or similar host object passed by pointer to the kernel.
    """
    import importlib

    _ext = importlib.import_module("cthreads._ext")
    ptr = int(_ext.sync_native_ptr(value))
    if not ptr:
        raise RuntimeError("cthreads.marshal: sync_native_ptr returned null")
    return ptr


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
    """
    Recursively write one Python value into the pack according to `schema`.

    Dispatches on `schema["kind"]` and issues the matching trampoline calls.
    This is the core pack path used by `pack_params` and by nested containers.

    #### Args:
    - lib: CDLL = loaded kernel library
    - symbol: str = compiled kernel export prefix
    - prefix: str = pack field segment (`a0`, `ret`, `a0_elem`, and similar)
    - schema: dict = compile-time TypeSchema for this value
    - value: Any = Python argument or field value to store
    - path: _Path = addressing trail for nested list or dict elements
    - pack: c_void_p = per-job args struct pointer
    - schemas: dict | None = forward-reference lookup from kernel meta

    #### Raises
    - TypeError = schema kind is not supported for packing
    - RuntimeError = required accessor name is missing

    #### Technical terms:
    - pack: per-job C++ args struct (see module docstring).
    - symbol: compiled kernel name prefix for accessor names.
    - prefix: field segment such as `a0` or `a0_elem` inside those names.
    - TypeSchema: type layout dict guiding recursive dispatch.
    - trampoline call: one native set, resize, clear, or insert accessor call.
    - addressing trail: `_Path` indices and key tags for nested containers.
    """
    schemas = schemas or {}
    schema = _resolve_schema(schema, schemas)
    kind = schema["kind"]
    if kind in ("int", "float", "bool", "str"):
        _set_prim(lib, symbol, prefix, path, kind, value, pack)
        return

    if kind == "threadable":
        # Each field gets prefix_{field_name} as its pack field segment in accessor names.
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
                # Older kernels may omit unused nested accessors; skip those field segments.
                if "missing symbol" in str(e):
                    continue
                raise
        return

    if kind == "list":
        inner = schema["inner"]
        n = len(value)
        # Resize the native list once, then pack each element at index i.
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
        # Replace the native map contents on each pack so Python dict order does not matter.
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
            # key tag: ("str", key) or ("int", key) - tells dict accessors which entry to use.
            key_tag = ("str", str(k)) if key_kind == "str" else ("int", int(k))
            child = _Path(indices=list(path.indices), keys=path.keys + [key_tag])
            child_extras = _extra(child)
            if val_schema["kind"] in ("int", "float", "bool", "str"):
                # Primitive values use insert; the key tag is encoded in the path extras.
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
                # Complex values allocate or fetch a nested slot, then pack into prefix_at.
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

    if kind == "tbuffer":
        # Store the host TBuffer native pointer; the worker reads through that handle.
        inner = schema.get("inner") or {}
        inner_name = inner.get("type_name")
        ptr = _tbuffer_native_ptr(
            value,
            inner_type_name=inner_name if inner.get("kind") == "threadable" else None,
        )
        fn = _fn(lib, f"{symbol}__set_{prefix}_ptr")
        _call(
            fn,
            None,
            [ctypes.c_void_p, ctypes.c_void_p],
            pack,
            ctypes.c_void_p(ptr),
        )
        return

    if kind == "sync":
        # Store the host sync object native pointer for Lock, Event, and similar types.
        ptr = _sync_native_ptr(value)
        fn = _fn(lib, f"{symbol}__set_{prefix}_ptr")
        _call(
            fn,
            None,
            [ctypes.c_void_p, ctypes.c_void_p],
            pack,
            ctypes.c_void_p(ptr),
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
    """
    Recursively read one value from the pack according to `schema`.

    When `into` is provided for mutable types, fields are written into that
    existing Python object instead of allocating a new container.

    #### Args:
    - lib: CDLL = loaded kernel library
    - symbol: str = compiled kernel export prefix
    - prefix: str = pack field segment inside accessor names
    - schema: dict = compile-time TypeSchema for this value
    - path: _Path = addressing trail for nested list or dict elements
    - pack: c_void_p = per-job args struct pointer
    - types: dict = maps Threadable type names to Python classes
    - schemas: dict | None = forward-reference lookup from kernel meta
    - into: Any | None = existing list, dict, or Threadable to mutate in place

    #### Returns
    - Any = decoded Python value

    #### Raises
    - TypeError = schema kind is not supported for unpacking
    - RuntimeError = accessor name or Threadable class registration is missing

    #### Technical terms:
    - pack: per-job C++ args struct (see module docstring).
    - symbol: compiled kernel name prefix for accessor names.
    - prefix: field segment such as `a0` or `a0_elem` inside those names.
    - TypeSchema: type layout dict guiding recursive dispatch.
    - trampoline call: one native get or size accessor call.
    - writeback: when `into` is set, mutate the caller's object in place.
    """
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
                # Older kernels may omit unused nested accessors; skip those field segments.
                if "missing symbol" in str(e):
                    continue
                raise
            _attr_set(obj, f["name"], child_val)
        return obj

    if kind == "list":
        inner = schema["inner"]
        # Read native length, then unpack each element at index i into out_list.
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
        # Iterate native map entries by index and rebuild a Python dict.
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
    """
    Pack all kernel parameters into a freshly allocated args struct at spawn time.

    Called from `module.cpp` once per job. Each parameter `a{i}` is packed
    according to its compile-time schema from kernel meta.

    #### Args:
    - symbol: str = compiled kernel export prefix
    - params: list[dict] = parameter metadata from kernel meta
    - values: list[Any] = Python arguments in call order
    - pack_ptr: int = native address of the new per-job pack
    - types: dict | None = reserved; layouts come from schemas today
    - schemas: dict | None = forward-reference lookup from kernel meta

    #### Technical terms:
    - pack: per-job C++ args struct allocated at spawn (see module docstring).
    - symbol: compiled kernel name; parameter `a{i}` accessors use prefix `a0`, `a1`, etc.
    - kernel meta: compile output describing params and schemas.
    - TypeSchema: per-parameter layout dict passed to `pack_value`.
    """
    del types  # Reserved for future use; schemas carry layouts today.
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


def promote_shared_to_host(
    symbol: str,
    params: list[dict],
    pack_ptr: int,
    host_ptr: int,
) -> None:
    """
    Copy staged Shared[T] pack slots into the cooperative SharedHost heap.

    Runs at spawn time before the worker thread starts. Only parameters with
    `pass_as == "shared"` are promoted.

    #### Args:
    - symbol: str = compiled kernel export prefix
    - params: list[dict] = parameter metadata from kernel meta
    - pack_ptr: int = native address of the per-job pack
    - host_ptr: int = native address of the SharedHost for this job

    #### Technical terms:
    - promote: copy from staged pack slot into SharedHost (see module docstring).
    - staged pack slot: temporary `a{i}` field holding the value before promotion.
    - SharedHost: cooperative heap shared by `Shared[T]` kernel access.
    - symbol: compiled kernel name; promote accessors are `{symbol}__promote_a{i}_shared`.
    """
    if not host_ptr:
        return
    lib = _lib()
    pack = _pack_c(pack_ptr)
    host = _pack_c(host_ptr)
    for i, p in enumerate(params):
        if p.get("pass_as") != "shared":
            continue
        fn = _fn(lib, f"{symbol}__promote_a{i}_shared")
        _call(fn, None, [ctypes.c_void_p, ctypes.c_void_p], pack, host)


def demote_shared_from_host(
    symbol: str,
    params: list[dict],
    pack_ptr: int,
    host_ptr: int,
    meta: dict[str, Any] | None = None,
) -> None:
    """
    Refresh staged pack slots from the SharedHost before writeback.

    Shared values live in cooperative host memory during the kernel run. This
    step copies the latest host state back into the pack so trampoline accessors
    can read it. Also demotes a shared return slot when `meta["return_pass_as"]`
    is `"shared"`.

    #### Args:
    - symbol: str = compiled kernel export prefix
    - params: list[dict] = parameter metadata from kernel meta
    - pack_ptr: int = native address of the per-job pack
    - host_ptr: int = native address of the SharedHost for this job
    - meta: dict | None = full kernel meta; needed for shared return demotion

    #### Technical terms:
    - demote: copy from SharedHost back into the staged pack slot (see module docstring).
    - staged pack slot: temporary `a{i}` or `ret` field updated before read or writeback.
    - SharedHost: cooperative heap where `Shared[T]` values live during the kernel run.
    - trampoline accessor: pack getters need demoted data before unpack can run.
    """
    if not host_ptr:
        return
    lib = _lib()
    pack = _pack_c(pack_ptr)
    host = _pack_c(host_ptr)
    for i, p in enumerate(params):
        if p.get("pass_as") != "shared":
            continue
        fn = _fn(lib, f"{symbol}__demote_a{i}_shared")
        _call(fn, None, [ctypes.c_void_p, ctypes.c_void_p], pack, host)
    if meta and meta.get("return_pass_as") == "shared":
        # Shared return values live in SharedHost until demoted into the pack ret slot.
        fn = _fn(lib, f"{symbol}__demote_return_shared")
        _call(fn, None, [ctypes.c_void_p, ctypes.c_void_p], pack, host)


def writeback_job_state(
    symbol: str,
    params: list[dict],
    values: list[Any],
    pack_ptr: int,
    host_ptr: int,
    types: dict[str, Any] | None = None,
    schemas: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """
    Sync shared host memory and mirror mutable reference arguments into Python.

    This is the main post-kernel and mid-run sync entry point called from
    `module.cpp`. It demotes shared slots, then writebacks Threadables, lists,
    and dicts into the original caller objects.

    #### Args:
    - symbol: str = compiled kernel export prefix
    - params: list[dict] = parameter metadata from kernel meta
    - values: list[Any] = original Python arguments to mutate in place
    - pack_ptr: int = native address of the per-job pack
    - host_ptr: int = native address of the SharedHost for this job
    - types: dict | None = Threadable name to class map for writeback
    - schemas: dict | None = forward-reference lookup from kernel meta
    - meta: dict | None = full kernel meta for shared return handling

    #### Technical terms:
    - writeback: mirror native pack fields into the caller's Python objects.
    - demote: refresh staged pack slots from SharedHost before writeback.
    - kernel meta: compile output passed from `module.cpp`.
    """
    meta = meta or {}
    demote_shared_from_host(symbol, params, pack_ptr, host_ptr, meta)
    writeback_params(symbol, params, values, pack_ptr, types, schemas)


def writeback_params(
    symbol: str,
    params: list[dict],
    values: list[Any],
    pack_ptr: int,
    types: dict[str, Any] | None = None,
    schemas: dict[str, Any] | None = None,
) -> None:
    """
    Mirror mutable reference parameters from the pack into live Python objects.

    Scalars are skipped because they were copied by value at spawn time. Plain
    dict stand-ins for Threadables are also skipped because they are not the
    live object the caller owns.

    #### Args:
    - symbol: str = compiled kernel export prefix
    - params: list[dict] = parameter metadata from kernel meta
    - values: list[Any] = original Python arguments to mutate in place
    - pack_ptr: int = native address of the per-job pack
    - types: dict | None = Threadable name to class map for writeback
    - schemas: dict | None = forward-reference lookup from kernel meta

    #### Technical terms:
    - writeback: copy mutable pack fields into the caller's live Python objects.
    - prefix: each parameter uses `a{i}` as its pack field segment.
    - pack: native args struct read by trampoline get accessors during unpack.
    """
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


def unpack_return(
    meta: dict[str, Any], pack_ptr: int, host_ptr: int = 0
) -> Any:
    """
    Build the Python return value after a kernel finishes.

    Handles shared returns by demoting from SharedHost first. Falls back to
    legacy flat return metadata when `return_schema` is absent in meta.

    #### Args:
    - meta: dict = kernel meta for the compiled function
    - pack_ptr: int = native address of the per-job pack
    - host_ptr: int = native SharedHost address (default 0 when unused)

    #### Returns
    - Any = decoded return value, or `None` for void kernels

    #### Technical terms:
    - pack: per-job args struct; return field uses prefix `ret`.
    - symbol: taken from `meta["symbol"]` for return accessor names.
    - demote: copy shared return value from SharedHost into the pack before read.
    - kernel meta: compile output describing return layout and types.
    - TypeSchema: return layout dict passed to `unpack_value`.
    """
    meta = dict(meta)
    if meta.get("return_pass_as") == "shared" and host_ptr:
        demote_shared_from_host(
            meta["symbol"],
            meta.get("params") or [],
            pack_ptr,
            host_ptr,
            meta,
        )
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
    """
    Upgrade older flat parameter metadata into the nested schema shape.

    Newer builds store full schemas on each param. This helper keeps marshal
    compatible with metadata emitted before that change.

    #### Args:
    - p: dict = single parameter entry from kernel meta

    #### Returns
    - dict = schema dict understood by `pack_value` and `unpack_value`

    #### Raises
    - TypeError = parameter kind cannot be converted

    #### Technical terms:
    - TypeSchema: nested type layout dict expected by pack and unpack (see module docstring).
    - kernel meta: older param entries stored flat `kind` and `fields` without `schema`.
    """
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
    if kind == "tbuffer":
        inner = p.get("schema", {}).get("inner")
        if inner is None:
            raise TypeError("legacy tbuffer param missing inner schema")
        return {
            "kind": "tbuffer",
            "cpp_type": p.get("cpp_type", ""),
            "inner": inner,
        }
    if kind == "sync":
        return {
            "kind": "sync",
            "cpp_type": p.get("cpp_type", ""),
            "type_name": p.get("schema", {}).get("type_name") or p.get("type_name"),
        }
    if p.get("pass_as") == "shared":
        schema = p.get("schema") or {}
        if schema:
            return schema
    raise TypeError(f"cannot legacy-upgrade kind {kind!r}")
