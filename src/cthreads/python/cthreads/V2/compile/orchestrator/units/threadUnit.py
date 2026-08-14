from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .baseUnit import BaseUnit
from ....types import PyType
from ....frontend.Registry import REGISTRY
from ....io import write_if_changed
from ....kernel_meta import build_kernel_meta, emit_trampoline_cpp, emit_trampoline_decls
from ..session.export_macro import EXPORT_HPP

if TYPE_CHECKING:
    from .threadableUnit import ThreadableUnit


@dataclass
class ThreadUnit(BaseUnit):

    # None => free function; otherwise this fn is a method of a Threadable
    owner: ThreadableUnit | None
    params: list[tuple[str, PyType]]
    return_type: PyType | None
    hpp_path: Path | None
    cpp_path: Path | None

    def validate(self) -> None:
        fn = self.handle.target
        if not getattr(fn, "__threaded", False):
            raise TypeError(
                f"{self.handle.name} is not a @Thread function"
            )
        if self.owner is not None:
            if self.owner.handle.name not in REGISTRY.threadables:
                raise ValueError(
                    f"@Thread function {self.handle.name} is a method of class "
                    f"{self.owner.handle.name} which is not wrapped with @Threadable. "
                    f"Please wrap the class with @Threadable to use {self.handle.name}."
                )

    def emit(self) -> None:
        """Write ``__Thread__/name.{hpp,cpp}`` for a free ``@Thread`` function."""
        if self.owner is not None:
            return
        if self.hpp_path is None or self.cpp_path is None:
            raise RuntimeError(
                f"ThreadUnit {self.handle.name} has no output paths"
            )

        from ...translation import translate_function

        fn = self.handle.target
        result = translate_function(
            fn,
            this_file=self.hpp_path,
            owner=None,
        )
        signature = result.free_signature()
        meta = build_kernel_meta(fn, symbol=result.func_name, owner_name=None)
        export_path = self.hpp_path.parent / "cthreads_export.hpp"
        seen_sig = set(result.sig_includes)
        extra_body = "".join(
            line for line in result.body_includes if line not in seen_sig
        )

        write_if_changed(export_path, EXPORT_HPP)

        hpp = "#pragma once\n\n"
        hpp += '#include "cthreads_export.hpp"\n\n'
        hpp += "".join(result.sig_includes)
        if result.sig_includes:
            hpp += "\n"
        hpp += f"{signature};\n"
        hpp += emit_trampoline_decls(meta)

        cpp = f'#include "{result.func_name}.hpp"\n'
        if extra_body:
            cpp += "\n" + extra_body
        cpp += f"\n{signature} {{\n{result.body}}}\n"
        cpp += "\n" + emit_trampoline_cpp(meta, real_call=result.func_name)

        write_if_changed(self.hpp_path, hpp)
        write_if_changed(self.cpp_path, cpp)
