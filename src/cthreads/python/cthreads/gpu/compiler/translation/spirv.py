"""
Compile GLSL compute source to SPIR-V.

Prefers in-process `_ext.gpu.compile_glsl` (vendored Khronos glslang linked
into the GPU extension — no extra user tools). Falls back to the Vulkan SDK
`glslc` CLI when the native binding is unavailable (CPU-only builds / dev).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _find_glslc() -> str | None:
    """
    Locate the shaderc `glslc` executable (dev fallback only).

    #### Returns
    - str | None = path to glslc, or None if not found
    """
    found: str | None = shutil.which("glslc")
    if found:
        return found
    sdk: str | None = os.environ.get("VULKAN_SDK")
    if not sdk:
        return None
    for rel in ("Bin/glslc.exe", "Bin/glslc", "bin/glslc"):
        candidate: Path = Path(sdk) / rel
        if candidate.is_file():
            return str(candidate)
    return None


def compile_glsl_to_spirv(source: str) -> bytes:
    """
    Compile a GLSL compute shader string to SPIR-V bytes.

    Uses native `cthreads._ext.gpu.compile_glsl` when the GPU extension was
    built with glslang. Otherwise invokes `glslc` if present.

    #### Args:
    - source: str = full `.comp` text (`#version` + buffers + `main`)

    #### Returns
    - bytes = SPIR-V binary (multiple of 4 bytes, magic 0x07230203)

    #### Raises
    - RuntimeError = no compiler available, or compile failed

    #### Technical terms:
    - SPIR-V: intermediate binary Vulkan drivers consume
    - glslang: Khronos GLSL compiler linked into `_ext` for GPU builds
    """
    try:
        from cthreads._ext import gpu as _gpu  # type: ignore

        compile_native = getattr(_gpu, "compile_glsl", None)
        if callable(compile_native):
            try:
                out = compile_native(source)
            except Exception as exc:
                raise RuntimeError(_format_glsl_compile_error(str(exc))) from exc
            if not isinstance(out, (bytes, bytearray)):
                raise RuntimeError(
                    "cthreads.gpu: _ext.gpu.compile_glsl did not return bytes"
                )
            data: bytes = bytes(out)
            if not data or (len(data) % 4) != 0:
                raise RuntimeError(
                    "cthreads.gpu: native compile_glsl returned invalid SPIR-V"
                )
            if data[:4] != b"\x03\x02\x23\x07":
                raise RuntimeError(
                    "cthreads.gpu: native compile_glsl missing SPIR-V magic"
                )
            return data
    except ImportError:
        pass

    glslc: str | None = _find_glslc()
    if glslc is None:
        raise RuntimeError(
            "cthreads.gpu: no GLSL compiler found. Rebuild with "
            "-DCTHREADS_GPU=ON (vendors glslang into _ext), or install the "
            "Vulkan SDK glslc for a temporary CLI fallback"
        )

    with tempfile.TemporaryDirectory(prefix="cthreads_glsl_") as tmp:
        tmp_path: Path = Path(tmp)
        comp_path: Path = tmp_path / "kernel.comp"
        spv_path: Path = tmp_path / "kernel.spv"
        comp_path.write_text(source, encoding="utf-8")
        cmd: list[str] = [
            glslc,
            "-fshader-stage=compute",
            str(comp_path),
            "-o",
            str(spv_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            err: str = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                _format_glsl_compile_error(
                    "glslc (shaderc CLI fallback) failed:\n" + err
                )
            )
        data = spv_path.read_bytes()

    if not data or (len(data) % 4) != 0:
        raise RuntimeError("cthreads.gpu: glslc produced invalid SPIR-V")
    if data[:4] != b"\x03\x02\x23\x07":
        raise RuntimeError("cthreads.gpu: glslc output missing SPIR-V magic")
    return data


def _format_glsl_compile_error(detail: str) -> str:
    """
    Wrap a glslang/glslc failure with a short identifier hint.

    Reserved GLSL words used as buffer instance names (e.g. `out`, `in`)
    fail at compile time; we surface that instead of maintaining a keyword list.
    """
    return (
        "cthreads.gpu: GLSL compile failed:\n"
        f"{detail.strip()}\n"
        "Hint: kernel / parameter names become GLSL identifiers. Avoid "
        "reserved words (e.g. out, in, buffer, shared, uniform, flat)."
    )
