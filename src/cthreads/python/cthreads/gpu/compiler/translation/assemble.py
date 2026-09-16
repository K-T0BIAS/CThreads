"""
Assemble a full GLSL compute shader (`.comp`) from preamble + body.
"""

from __future__ import annotations

_DEFAULT_GLSL_VERSION: int = 450


def assemble_comp(
    preamble: str,
    body: str,
    *,
    version: int = _DEFAULT_GLSL_VERSION,
) -> str:
    """
    Build a complete compute shader string: `#version`, buffers, `void main()`.

    Body lines from GpuSyntax are already indented; they become the main body.

    #### Args:
    - preamble: str = Signature output (local_size + buffer layouts)
    - body: str = lowered statement lines (already indented)
    - version: int = GLSL version directive (default 450)

    #### Returns
    - str = full `.comp` source text
    """
    pre: str = preamble.rstrip()
    bod: str = body.rstrip()
    lines: list[str] = [f"#version {version}", ""]
    if pre:
        lines.append(pre)
        lines.append("")
    lines.append("void main() {")
    if bod:
        lines.append(bod)
    lines.append("}")
    lines.append("")
    return "\n".join(lines)
