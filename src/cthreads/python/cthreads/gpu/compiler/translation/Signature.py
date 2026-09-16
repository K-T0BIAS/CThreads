from __future__ import annotations

import ast
from typing import get_type_hints

from ....types import PyList, hint_to_pytype
from .Glsl import align_bytes, elem_type_name, size_bytes, type_name
from .context import GpuTranslationContext
from .result import GpuSignatureResult, ListField, ScalarField


class GpuSignature:
    """Emit GLSL buffer preamble; fill ctx.symbols and binding_names."""

    @staticmethod
    def translate(
        func_def: ast.FunctionDef, ctx: GpuTranslationContext
    ) -> GpuSignatureResult:
        # No Threadable owners yet — no localns.
        hints = get_type_hints(ctx.fn)
        args = list(func_def.args.args)

        if func_def.args.vararg or func_def.args.kwarg or func_def.args.kwonlyargs:
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                "*args/**kwargs/kw-only args are not supported"
            )

        ret_hint = hints.get("return")
        if ret_hint is not None and ret_hint is not type(None):
            raise TypeError(
                f"GPU function {ctx.func_name}: return must be None "
                f"(in-place list writeback only), got {ret_hint!r}"
            )

        scalar_fields: list[ScalarField] = []
        # (name, glsl_elem) — binding numbers assigned after we know if scalars exist.
        pending_lists: list[tuple[str, str]] = []
        scalar_bytes = 0

        for arg in args:
            if arg.arg not in hints:
                raise TypeError(
                    f"GPU function {ctx.func_name}: "
                    f"parameter {arg.arg!r} needs a type annotation"
                )
            py_type = hint_to_pytype(hints[arg.arg])
            ctx.symbols[arg.arg] = py_type

            if isinstance(py_type, PyList):
                glsl_elem = elem_type_name(py_type)
                pending_lists.append((arg.arg, glsl_elem))
                ctx.list_params.add(arg.arg)
            else:
                glsl_ty = type_name(py_type)
                align = align_bytes(py_type)
                scalar_bytes = (scalar_bytes + align - 1) & ~(align - 1)
                scalar_bytes += size_bytes(py_type)
                scalar_fields.append((arg.arg, glsl_ty))
                ctx.scalar_params.add(arg.arg)

        # Scalars at binding 0 when present; lists start at 1 or 0 accordingly.
        list_base = 1 if scalar_fields else 0
        list_fields: list[ListField] = []
        for i, (name, glsl_elem) in enumerate(pending_lists):
            binding = list_base + i
            list_fields.append((binding, name, glsl_elem))
            ctx.binding_names.append((binding, name, "list"))

        if scalar_fields:
            ctx.binding_names.insert(0, (0, "scalars", "scalar"))

        binding_count = (1 if scalar_fields else 0) + len(list_fields)

        lines: list[str] = [f"layout(local_size_x = {ctx.local_size_x}) in;", ""]
        if scalar_fields:
            lines.append("layout(set = 0, binding = 0, std430) buffer Scalars {")
            for name, glsl_ty in scalar_fields:
                lines.append(f"    {glsl_ty} {name};")
            lines.append("} scalars;")
            lines.append("")
        for binding, name, glsl_elem in list_fields:
            block = name[:1].upper() + name[1:]
            lines.append(
                f"layout(set = 0, binding = {binding}, std430) buffer {block} {{"
            )
            lines.append(f"    {glsl_elem} data[];")
            lines.append(f"}} {name};")
            lines.append("")

        preamble = "\n".join(lines).rstrip() + "\n"

        return GpuSignatureResult(
            func_name=func_def.name,
            preamble=preamble,
            binding_count=binding_count,
            scalar_bytes=scalar_bytes,
            local_size_x=ctx.local_size_x,
            scalar_fields=scalar_fields,
            list_fields=list_fields,
        )
