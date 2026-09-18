"""
One-shot GPU translate: parse -> Signature -> Syntax -> assemble -> optional SPIR-V.
"""

from __future__ import annotations

import ast
from typing import Callable

from ....compiler.translation.Source import Source
from .Signature import GpuSignature
from .assemble import assemble_comp
from .context import GpuTranslationContext
from .result import GpuTranslationResult
from .spirv import compile_glsl_to_spirv
from .syntax.Syntax import GpuSyntax


def translate_function_for_gpu(
    fn: Callable,
    *,
    local_size_x: int = 64,
    compile_spirv: bool = False,
) -> GpuTranslationResult:
    """
    Translate one `@Gpu` function to GLSL, optionally compile to SPIR-V.

    #### Args:
    - fn: Callable = annotated GPU kernel function
    - local_size_x: int = workgroup size x (default 64)
    - compile_spirv: bool = if True, run shaderc (`glslc` / native) on `source`

    #### Returns
    - GpuTranslationResult = preamble, body, `source`, optional `spirv`, layout

    #### Raises
    - RuntimeError = SPIR-V compile requested but compiler missing / failed
    """
    ctx: GpuTranslationContext = GpuTranslationContext(
        fn=fn, local_size_x=local_size_x
    )
    func_def: ast.FunctionDef = Source.parse_function(fn)
    sig = GpuSignature.translate(func_def, ctx)

    body_lines: list[str] = []
    for stmt in func_def.body:
        body_lines.extend(GpuSyntax.stmt(stmt, ctx))
    body: str = "\n".join(body_lines)
    if body and not body.endswith("\n"):
        body += "\n"

    source: str = assemble_comp(sig.preamble, body)
    spirv: bytes | None = None
    if compile_spirv:
        spirv = compile_glsl_to_spirv(source)

    return GpuTranslationResult(
        func_name=sig.func_name,
        preamble=sig.preamble,
        body=body,
        source=source,
        binding_count=sig.binding_count,
        scalar_bytes=sig.scalar_bytes,
        local_size_x=sig.local_size_x,
        scalar_fields=sig.scalar_fields,
        list_fields=sig.list_fields,
        spirv=spirv,
    )
