"""
Drain `REGISTRY.gpu_functions` into `GpuUnit`s and run emit.

Does not keep its own unit maps; units live on REGISTRY.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, get_type_hints

from ....cache import ensure_gitignore, load_cache, save_cache
from ....compiler.orchestrator.units.handle import Handle
from ....frontend.Registry import REGISTRY
from ....gpu.gpu_kernel_meta import GPU_KERNELS, build_gpu_kernel_meta
from ....types import PyType, hint_to_pytype
from .gpu_unit import GpuUnit


class GpuCompileSession:
    """
    Stateless GPU compile pass: registry functions -> units -> emit -> cache.
    """

    @staticmethod
    def compile(force: bool = False) -> dict[str, Any]:
        """
        Build GpuUnits for every registered `@Gpu` function and emit.

        #### Args:
        - force: bool = passed through to `GpuUnit.emit` (default False)

        #### Returns
        - dict[str, Any] = `root`, `cache`, `rewritten` (unit names that rewrote)

        #### Raises
        - RuntimeError = nothing registered
        - TypeError = bad annotations / non-None return
        """
        REGISTRY.gpu_function_units.clear()
        GPU_KERNELS.clear()

        if not REGISTRY.gpu_functions:
            raise RuntimeError("Nothing registered to compile")

        sample: Any = next(iter(REGISTRY.gpu_functions.values()))
        root = Path(inspect.getfile(sample)).resolve().parent
        ensure_gitignore(root)
        cache = load_cache(root)
        rewritten: list[str] = []

        for qualname, fn in list(REGISTRY.gpu_functions.items()):
            src_file = Path(inspect.getfile(fn)).resolve()
            hints = get_type_hints(fn)
            params: list[tuple[str, PyType]] = []
            for pname in inspect.signature(fn).parameters:
                if pname not in hints:
                    raise TypeError(
                        f"GPU function {qualname}: "
                        f"parameter {pname!r} needs a type annotation"
                    )
                params.append((pname, hint_to_pytype(hints[pname])))

            ret_hint = hints.get("return")
            if ret_hint not in (None, type(None)):
                raise TypeError(
                    f"GPU function {qualname}: "
                    f"return type {ret_hint!r} is not allowed; use -> None "
                    f"with in-place list updates"
                )
            return_type = None

            # Rebuild meta after GPU_KERNELS.clear (decorator may have built it earlier).
            build_gpu_kernel_meta(fn)

            gpu_unit = GpuUnit(
                handle=Handle(
                    name=qualname,
                    path=str(src_file),
                    target=fn,
                ),
                params=params,
                return_type=return_type,
            )
            REGISTRY.gpu_function_units[qualname] = gpu_unit

        for unit in REGISTRY.gpu_function_units.values():
            if unit.emit(force=force, cache=cache):
                rewritten.append(unit.handle.name)

        save_cache(root, cache)
        return {"root": root, "cache": cache, "rewritten": rewritten}
