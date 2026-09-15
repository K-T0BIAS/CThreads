from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....cache import source_fingerprint, write_if_changed
from ....compiler.orchestrator.units.baseUnit import BaseUnit
from ....types.pyType import PyType
from ... import _ext_gpu_api
from ...gpu_kernel_meta import build_gpu_kernel_meta
from ..translation.translate import translate_function_for_gpu


@dataclass
class GpuUnit(BaseUnit):
    """
    One `@Gpu` free function: translate GLSL, compile SPIR-V, register shader.
    """

    params: list[tuple[str, PyType]]
    return_type: PyType | None

    def validate(self) -> None:
        """
        Ensure the handle target is a `@Gpu` function.

        #### Raises
        - TypeError = target is not marked `@Gpu`
        """
        fn = self.handle.target
        if not getattr(fn, "__gpu__", False):
            raise TypeError(f"{self.handle.name} is not a @Gpu function")

    def emit(self, *, force: bool = False, cache: dict[str, Any] | None = None) -> bool:
        """
        Translate, compile with shaderc, write artifacts, register SPIR-V.

        Always registers into the process ShaderCache (Vulkan state is not on
        disk). The source fingerprint only skips rewriting `__Gpu__` files.

        #### Args:
        - force: bool = rewrite `__Gpu__` artifacts even if the hash matches
        - cache: dict[str, Any] | None = shared `.cthreads_cache.json` document

        #### Returns
        - bool = True if `__Gpu__` files were rewritten

        #### Raises
        - RuntimeError = GLSL/SPIR-V compile failed or GPU ext missing
        """
        self.validate()
        fn = self.handle.target
        meta = getattr(fn, "__gpu_kernel_meta__", None)
        if not isinstance(meta, dict):
            meta = build_gpu_kernel_meta(fn).to_dict()

        src_hash: str = source_fingerprint(fn)
        local_size_x: int = int(meta.get("local_size_x", 64))
        result = translate_function_for_gpu(
            fn, local_size_x=local_size_x, compile_spirv=True
        )
        if result.spirv is None:
            raise RuntimeError(
                f"GPU unit {self.handle.name}: SPIR-V compile produced no bytes"
            )

        src_file: Path = Path(self.handle.path).resolve()
        out_dir: Path = src_file.parent / "__Gpu__"
        out_dir.mkdir(parents=True, exist_ok=True)
        comp_path: Path = out_dir / f"{result.func_name}.comp"
        spv_path: Path = out_dir / f"{result.func_name}.spv"

        rewritten: bool = False
        gpu_units: dict[str, Any] = {}
        if cache is not None:
            gpu_units = cache.setdefault("gpu_units", {})
            prev = gpu_units.get(self.handle.name)
            hash_ok = (
                isinstance(prev, dict)
                and prev.get("hash") == src_hash
                and not force
            )
        else:
            hash_ok = False

        if not hash_ok:
            rewritten = write_if_changed(comp_path, result.source)
            prev_spv: bytes | None = None
            if spv_path.is_file():
                try:
                    prev_spv = spv_path.read_bytes()
                except OSError:
                    prev_spv = None
            if prev_spv != result.spirv:
                spv_path.write_bytes(result.spirv)
                rewritten = True

        symbol: str = str(meta.get("symbol", result.func_name))
        binding_count: int = int(meta.get("binding_count", result.binding_count))
        # Always populate process ShaderCache (also after shutdown released it).
        # Disk fingerprint only gates `__Gpu__` file rewrites above.
        try:
            _ext_gpu_api.register_shader(symbol, result.spirv, binding_count)
        except Exception as exc:
            msg: str = str(exc)
            if "already exists" not in msg:
                raise

        if cache is not None:
            gpu_units[self.handle.name] = {
                "hash": src_hash,
                "symbol": symbol,
                "binding_count": binding_count,
                "registered": True,
                "comp": str(comp_path),
                "spv": str(spv_path),
            }
        return rewritten
