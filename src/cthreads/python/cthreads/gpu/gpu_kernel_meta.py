"""
Compile-time metadata for one `@Gpu` kernel.

Shape matches `launch_gpu_kernel` in `module.hpp`: `symbol`, binding layout,
`params` with flat `kind` / `pass_as` / `elem_kind` / `elem_bytes`, and optional
dispatch overrides.

v1 is in-place list writeback only (`pass_as="ref"` on lists). Kernels are
`-> None`; scalars are not written back on join.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

from ..types import (
    PyBool,
    PyDict,
    PyFloat,
    PyInt,
    PyList,
    PyString,
    PyThreadable,
    PyType,
    hint_to_pytype,
    is_shared_pytype,
    is_sync_pytype,
    is_tbuffer_pytype,
)

# Populated by build_gpu_kernel_meta(); keyed by symbol.
GPU_KERNELS: dict[str, "GpuKernelMeta"] = {}

# Host / std430 sizes used by module.cpp py_size_of (GLSL bool is 32-bit).
_GPU_SCALAR_BYTES: dict[str, int] = {
    "bool": 4,
    "int": 4,
    "float": 4,
    "double": 8,
}

_DEFAULT_LOCAL_SIZE_X = 64


def _align_up(value: int, alignment: int) -> int:
    """
    Round `value` up to the next multiple of `alignment` (std430 field packing).

    #### Args:
    - value: int = current byte offset
    - alignment: int = required alignment in bytes

    #### Returns
    - int = aligned offset
    """
    return (value + alignment - 1) & ~(alignment - 1)


def _gpu_kind_and_bytes(py_type: PyType) -> tuple[str, int]:
    """
    Map a scalar PyType to the launch `kind` string and std430 byte size.

    #### Args:
    - py_type: PyType = scalar type from hint_to_pytype

    #### Returns
    - tuple[str, int] = (kind, elem_bytes) for int / float / bool

    #### Raises
    - TypeError = type is not a supported GPU scalar
    """
    if isinstance(py_type, PyInt):
        return "int", _GPU_SCALAR_BYTES["int"]
    if isinstance(py_type, PyFloat):
        # Shader float (f32). CPU @Thread uses double; GPU v1 follows GLSL float.
        return "float", _GPU_SCALAR_BYTES["float"]
    if isinstance(py_type, PyBool):
        return "bool", _GPU_SCALAR_BYTES["bool"]
    raise TypeError(
        f"GPU kernel: unsupported scalar type {py_type.name!r} "
        f"(allowed: int, float, bool)"
    )


def pytype_to_gpu_schema(py_type: PyType) -> GpuTypeSchema:
    """
    Convert a PyType into a GPU schema for marshal and codegen.

    Lists must be `list[int|float|bool]`. Dict, str, Threadable, sync, and
    shared types are rejected.

    #### Args:
    - py_type: PyType = type from hint_to_pytype

    #### Returns
    - GpuTypeSchema = scalar or list schema with spirv_type / elem_bytes

    #### Raises
    - TypeError = unsupported GPU type
    """
    if is_sync_pytype(py_type) or is_shared_pytype(py_type) or is_tbuffer_pytype(
        py_type
    ):
        raise TypeError(
            f"GPU kernel: {py_type.name!r} is not supported on the GPU path"
        )
    if isinstance(py_type, (PyDict, PyString, PyThreadable)):
        raise TypeError(
            f"GPU kernel: {py_type.name!r} is not supported on the GPU path"
        )
    if isinstance(py_type, PyList):
        kind, elem_bytes = _gpu_kind_and_bytes(py_type.inner_type)
        return GpuTypeSchema(
            kind="list",
            spirv_type=f"{kind}[]",
            inner=GpuTypeSchema(
                kind=kind,
                spirv_type=kind,
                elem_bytes=elem_bytes,
            ),
            elem_bytes=elem_bytes,
        )
    kind, elem_bytes = _gpu_kind_and_bytes(py_type)
    return GpuTypeSchema(kind=kind, spirv_type=kind, elem_bytes=elem_bytes)


@dataclass
class GpuTypeSchema:
    """
    Layout for one marshal or codegen slot (scalar or list of scalars).

    Serialized under each param's `schema` key. Launch also reads flat
    `kind` / `elem_kind` / `elem_bytes` from GpuParamMeta.to_dict.

    #### Technical terms:
    - std430: GLSL/SPIR-V storage buffer packing rules for scalar fields
    """

    kind: str  # int, float, bool, list
    spirv_type: str
    elem_bytes: int = 0
    inner: GpuTypeSchema | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this schema for `fn.__gpu_kernel_meta__`.

        #### Returns
        - dict[str, Any] = JSON-friendly schema node for marshal and tests

        #### Raises
        - ValueError = list schema is missing an inner element schema
        """
        d: dict[str, Any] = {
            "kind": self.kind,
            "spirv_type": self.spirv_type,
            "elem_bytes": self.elem_bytes,
        }
        if self.kind == "list":
            if self.inner is None:
                raise ValueError("list schema requires inner element schema")
            d["inner"] = self.inner.to_dict()
        return d


@dataclass
class GpuParamMeta:
    """
    One kernel parameter: Python name, pass mode, and GpuTypeSchema.

    Scalars default to `pass_as="value"` (packed into binding 0). Lists default
    to `pass_as="ref"` (downloaded into the same Python list on join).

    #### Technical terms:
    - pass_as: value packs into the scalar SSBO; ref lists are written back on join
    - binding 0: scalar storage buffer; list buffers use bindings 1..N
    """

    name: str
    pass_as: str  # value | ref
    schema: GpuTypeSchema

    def __post_init__(self) -> None:
        if self.pass_as not in ("value", "ref"):
            raise TypeError(
                f"GPU param {self.name!r}: pass_as must be 'value' or 'ref', "
                f"got {self.pass_as!r}"
            )
        if self.schema.kind == "list" and self.pass_as not in ("value", "ref"):
            raise TypeError(
                f"GPU list param {self.name!r}: pass_as must be 'value' or 'ref'"
            )

    @property
    def kind(self) -> str:
        """
        Return the top-level schema kind for this parameter.

        #### Returns
        - str = `int`, `float`, `bool`, or `list`
        """
        return self.schema.kind

    @property
    def elem_kind(self) -> str | None:
        """
        Return the list element kind, or None for scalars.

        #### Returns
        - str | None = inner kind when `kind == "list"`, else None
        """
        if self.schema.kind == "list" and self.schema.inner is not None:
            return self.schema.inner.kind
        return None

    @property
    def elem_bytes(self) -> int | None:
        """
        Return the list element size in bytes, or None for scalars.

        #### Returns
        - int | None = element byte width when `kind == "list"`, else None
        """
        if self.schema.kind == "list":
            return self.schema.elem_bytes
        return None

    def to_dict(self) -> dict[str, Any]:
        """
        Flatten this parameter for `launch_gpu_kernel`.

        Emits top-level `kind` / `pass_as` and, for lists, `elem_kind` /
        `elem_bytes` as read by module.cpp.

        #### Returns
        - dict[str, Any] = parameter metadata consumed by marshal and launch
        """
        d: dict[str, Any] = {
            "name": self.name,
            "pass_as": self.pass_as,
            "kind": self.schema.kind,
            "schema": self.schema.to_dict(),
        }
        if self.schema.kind == "list":
            d["elem_kind"] = self.elem_kind
            d["elem_bytes"] = self.elem_bytes
        return d


@dataclass
class GpuKernelMeta:
    """
    Compile-time record for one `@Gpu` kernel (ShaderCache key + launch meta).

    No trampolines or DLL symbols. `symbol` is the ShaderCache key. Return is
    always void (`-> None`); results are ref-list writeback on join.

    #### Technical terms:
    - ShaderCache: process map of symbol to reusable pipeline and set layout
    - writeback: download of ref list buffers into caller-owned Python lists
    """

    symbol: str
    binding_count: int
    scalar_bytes: int
    params: list[GpuParamMeta]
    local_size_x: int = _DEFAULT_LOCAL_SIZE_X
    # Optional dispatch overrides; None => launch may ceil(n / local_size_x).
    group_count_x: int | None = None
    group_count_y: int | None = 1
    group_count_z: int | None = 1
    types: dict[str, Any] = field(default_factory=dict)
    schemas: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize kernel metadata for `fn.__gpu_kernel_meta__` and launch.

        #### Returns
        - dict[str, Any] = dict accepted by `launch_gpu_kernel`
        """
        return {
            "symbol": self.symbol,
            "binding_count": self.binding_count,
            "scalar_bytes": self.scalar_bytes,
            "local_size_x": self.local_size_x,
            "group_count_x": self.group_count_x,
            "group_count_y": self.group_count_y,
            "group_count_z": self.group_count_z,
            "params": [p.to_dict() for p in self.params],
            "types": dict(self.types),
            "schemas": dict(self.schemas),
        }


def build_gpu_kernel_meta(
    fn: Callable,
    *,
    symbol: str | None = None,
    local_size_x: int = _DEFAULT_LOCAL_SIZE_X,
) -> GpuKernelMeta:
    """
    Build and attach metadata for one `@Gpu` function from its annotations.

    Stores the record in `GPU_KERNELS[symbol]` and sets
    `fn.__gpu_kernel_meta__` to `meta.to_dict()`.

    #### Args:
    - fn: Callable = `@Gpu` function with type hints
    - symbol: str | None = ShaderCache key (default: `fn.__name__`)
    - local_size_x: int = compute workgroup size (default: 64)

    #### Returns
    - GpuKernelMeta = full metadata record for this kernel

    #### Raises
    - TypeError = missing annotations, non-None return, or unsupported types
    """
    hints = get_type_hints(fn)
    ret = hints.get("return", None)
    if ret not in (None, type(None)):
        raise TypeError(
            f"GPU kernel {fn.__qualname__}: return must be None "
            f"(in-place list writeback only), got {ret!r}"
        )

    sig = inspect.signature(fn)
    if any(
        p.kind
        in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        for p in sig.parameters.values()
    ):
        raise TypeError(
            f"GPU kernel {fn.__qualname__}: *args / **kwargs / keyword-only "
            "args are not supported"
        )

    params: list[GpuParamMeta] = []
    scalar_bytes = 0
    list_count = 0

    for pname in sig.parameters:
        if pname not in hints:
            raise TypeError(
                f"GPU kernel {fn.__qualname__}: parameter {pname!r} needs a "
                "type annotation"
            )
        py_type = hint_to_pytype(hints[pname])
        schema = pytype_to_gpu_schema(py_type)
        if schema.kind == "list":
            list_count += 1
            pass_as = "ref"
        else:
            align = schema.elem_bytes
            scalar_bytes = _align_up(scalar_bytes, align)
            scalar_bytes += schema.elem_bytes
            pass_as = "value"
        params.append(GpuParamMeta(name=pname, pass_as=pass_as, schema=schema))

    # Binding 0 = scalar SSBO (reserved when any scalars); lists at 1..N.
    binding_count = (1 if scalar_bytes > 0 else 0) + list_count
    sym = symbol if symbol is not None else fn.__name__

    meta = GpuKernelMeta(
        symbol=sym,
        binding_count=binding_count,
        scalar_bytes=scalar_bytes,
        params=params,
        local_size_x=local_size_x,
    )
    GPU_KERNELS[sym] = meta
    fn.__gpu_kernel_meta__ = meta.to_dict()  # type: ignore[attr-defined]
    return meta
