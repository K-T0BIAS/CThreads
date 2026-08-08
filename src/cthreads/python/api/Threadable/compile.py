import inspect
from pathlib import Path
from typing import get_type_hints

from ..pyTypes import hint_to_pytype
from ..CONFIG import STORE, VERSION
from ..Thread.compile.compile import translate_thread


def compile_threadable(cls: type, methods: list) -> None:
    if not getattr(cls, "__threadable", False):
        raise TypeError(f"Class {cls.__name__} is not a Threadable class")
    if getattr(cls, "__threadable_version", "") != VERSION:
        raise TypeError(f"Class {cls.__name__} has an invalid version")

    name = cls.__name__
    src_file = Path(inspect.getfile(cls)).resolve()
    out_dir = src_file.parent / "__Threadable__"
    out_dir.mkdir(parents=True, exist_ok=True)
    hpp_path = out_dir / f"{name}.hpp"
    cpp_path = out_dir / f"{name}.cpp"

    # Register early so method signatures can #include / resolve this type.
    STORE[name] = str(hpp_path)

    includes: list[str] = []
    fields: list[str] = []
    seen_includes: set[str] = set()

    for field_name, hint in get_type_hints(cls).items():
        py_type = hint_to_pytype(hint)
        decl, include = py_type.to_cpp(field_name)
        fields.append(f"    {decl}")
        for line in include.splitlines(keepends=True):
            if line and line not in seen_includes:
                seen_includes.add(line)
                includes.append(line)

    method_results = [translate_thread(fn, owner_name=name) for fn in methods]

    for result in method_results:
        for line in result.sig_includes:
            if line and line not in seen_includes:
                # Skip self-include of this class header.
                if name in line and "__Threadable__" in line:
                    continue
                seen_includes.add(line)
                includes.append(line)

    include_block = "".join(includes)
    field_block = "\n".join(fields)
    if field_block:
        field_block += "\n"

    method_decls = "\n".join(r.method_decl() for r in method_results)
    if method_decls:
        method_decls += "\n"

    hpp = "#pragma once\n\n"
    if include_block:
        hpp += include_block + "\n"
    hpp += f"struct {name} {{\n{field_block}{method_decls}}};\n"

    cpp = f'#include "{name}.hpp"\n'
    body_extra_seen = set(includes)
    for result in method_results:
        for line in result.body_includes:
            if line and line not in body_extra_seen:
                if name in line and "__Threadable__" in line:
                    continue
                body_extra_seen.add(line)
                cpp += line
        cpp += f"\n{result.method_def_signature(name)} {{\n{result.body}}}\n"

    hpp_path.write_text(hpp, encoding="utf-8")
    cpp_path.write_text(cpp, encoding="utf-8")


# Old name used by earlier imports
def compile(cls: type) -> None:
    compile_threadable(cls, methods=[])
