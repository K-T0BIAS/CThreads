from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .baseUnit import BaseUnit
from ....types import PyType
from ....io import write_if_changed
from ....kernel_meta import build_kernel_meta, emit_trampoline_cpp, emit_trampoline_decls
from ...translation import include_for
from ..session.export_macro import EXPORT_HPP

if TYPE_CHECKING:
    from .threadUnit import ThreadUnit


@dataclass
class ThreadableUnit(BaseUnit):

    fields: dict[str, PyType]
    hpp_path: Path
    cpp_path: Path
    methods: list[ThreadUnit] = field(default_factory=list)

    def validate(self) -> None:
        cls = self.handle.target
        if not isinstance(cls, type) or not getattr(cls, "__threadable", False):
            raise TypeError(
                f"{self.handle.name} is not a @Threadable class"
            )

    def emit(self) -> None:
        """Write `__Threadable__/Name.{hpp,cpp}` for this class and its methods."""
        from ...translation import translate_function

        name = self.handle.name
        cls = self.handle.target
        results = [
            translate_function(
                method.handle.target,
                this_file=self.hpp_path,
                owner=self,
            )
            for method in self.methods
        ]

        includes: list[str] = []
        seen_includes: set[str] = set()
        fields: list[str] = []

        for field_name, py_type in self.fields.items():
            decl, _ = py_type.to_cpp(field_name)
            fields.append(f"    {decl}")
            for line in include_for(py_type, self.hpp_path).splitlines(keepends=True):
                if line and line not in seen_includes:
                    seen_includes.add(line)
                    includes.append(line)

        for result in results:
            for line in result.sig_includes:
                if not line or line in seen_includes:
                    continue
                if name in line and "__Threadable__" in line:
                    continue
                seen_includes.add(line)
                includes.append(line)

        include_block = "".join(includes)
        field_block = "\n".join(fields)
        if field_block:
            field_block += "\n"

        method_decls = "\n".join(r.method_decl() for r in results)
        if method_decls:
            method_decls += "\n"

        c_wrappers_decl: list[str] = []
        c_wrappers_def: list[str] = []
        trampoline_decls: list[str] = []
        trampoline_defs: list[str] = []
        for method, result in zip(self.methods, results):
            export = f"{name}_{result.func_name}"
            params = result.params_csv
            ret = (
                "void"
                if result.return_type is None
                else result.return_type.cpp_name
            )
            c_params = f"{name}* self" + (f", {params}" if params else "")
            c_sig = f"CTHREADS_API {ret} {export}({c_params})"
            c_wrappers_decl.append(f"{c_sig};")
            call_args = (
                ", ".join(
                    part.strip().split()[-1].lstrip("&*")
                    for part in params.split(",")
                    if part.strip()
                )
                if params.strip()
                else ""
            )
            if result.return_type is None:
                body = f"    self->{result.func_name}({call_args});\n"
            else:
                body = f"    return self->{result.func_name}({call_args});\n"
            c_wrappers_def.append(f"{c_sig} {{\n{body}}}")

            meta = build_kernel_meta(
                method.handle.target,
                symbol=export,
                owner_name=name,
                owner_cls=cls,
            )
            trampoline_decls.append(emit_trampoline_decls(meta))
            trampoline_defs.append(emit_trampoline_cpp(meta, real_call=export))

        src_file = Path(self.handle.path)
        thread_dir = src_file.parent / "__Thread__"
        export_path = thread_dir / "cthreads_export.hpp"
        write_if_changed(export_path, EXPORT_HPP)

        hpp = "#pragma once\n\n"
        hpp += '#include "../__Thread__/cthreads_export.hpp"\n\n'
        if include_block:
            hpp += include_block + "\n"
        hpp += f"struct {name} {{\n{field_block}{method_decls}}};\n"
        if c_wrappers_decl:
            hpp += "\n" + "\n".join(c_wrappers_decl) + "\n"
        if trampoline_decls:
            hpp += "\n" + "".join(trampoline_decls)

        cpp = f'#include "{name}.hpp"\n'
        body_extra_seen = set(includes)
        for result in results:
            for line in result.body_includes:
                if not line or line in body_extra_seen:
                    continue
                if name in line and "__Threadable__" in line:
                    continue
                body_extra_seen.add(line)
                cpp += line
            cpp += f"\n{result.method_def_signature(name)} {{\n{result.body}}}\n"
        for wrapper in c_wrappers_def:
            cpp += f"\n{wrapper}\n"
        for block in trampoline_defs:
            cpp += "\n" + block

        write_if_changed(self.hpp_path, hpp)
        write_if_changed(self.cpp_path, cpp)
