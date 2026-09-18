from __future__ import annotations

from dataclasses import dataclass

# (param_name, glsl_type) e.g. ("n", "int")
ScalarField = tuple[str, str]
# (binding, param_name, glsl_elem_type) e.g. (1, "x", "float")
ListField = tuple[int, str, str]


@dataclass
class GpuSignatureResult:
    """GLSL buffer preamble + layout facts for one @Gpu Signature."""

    func_name: str
    preamble: str
    binding_count: int
    scalar_bytes: int
    local_size_x: int
    scalar_fields: list[ScalarField]
    list_fields: list[ListField]


@dataclass
class GpuTranslationResult:
    """
    Emit contract for one translated `@Gpu` function.

    `source` is the full `.comp` text (`#version` + preamble + `main`).
    `spirv` is set when compile_spirv was requested and compilation succeeded.
    """

    func_name: str
    preamble: str
    body: str
    source: str
    binding_count: int
    scalar_bytes: int
    local_size_x: int
    scalar_fields: list[ScalarField]
    list_fields: list[ListField]
    spirv: bytes | None = None
