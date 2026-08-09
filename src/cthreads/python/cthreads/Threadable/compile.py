"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import inspect
from pathlib import Path
from typing import Any, get_type_hints

from ..pyTypes import hint_to_pytype
from ..CONFIG import STORE, VERSION
from ..Thread.compile.compile import translate_thread
from ..kernel_meta import build_kernel_meta, emit_trampoline_cpp, emit_trampoline_decls
from ..cache import source_fingerprint, write_if_changed

# define the export macro to convert CTHREADS_API to extern "C" __declspec(dllexport) on Windows and extern "C" otherwise
_EXPORT_HPP = (
    "#pragma once\n\n"
    "#ifndef CTHREADS_API\n"
    "#  if defined(_WIN32)\n"
    '#    define CTHREADS_API extern "C" __declspec(dllexport)\n'
    "#  else\n"
    '#    define CTHREADS_API extern "C"\n'
    "#  endif\n"
    "#endif\n"
)


def compile_threadable(
    cls: type,
    methods: list,
    force: bool = False,
    cache: dict[str, Any] | None = None,
) -> bool:
    """
    Generate Threadable C++. Returns True if any output file was rewritten.

    #### Args
    - cls: type - the class to compile
    - methods: list - the methods to compile
    - force: bool - whether to force compile even if the output files are up to date
    - cache: dict[str, Any] | None - the cache to use (optional, if present it will be checked to see if the code needs to be compiled)

    #### Returns
    - bool - True if any output file was rewritten, False otherwise
    """
    # check if the class is a Threadable class
    if not getattr(cls, "__threadable", False):
        raise TypeError(f"Class {cls.__name__} is not a Threadable class")
    # check if the class has the correct version
    if getattr(cls, "__threadable_version", "") != VERSION:
        raise TypeError(f"Class {cls.__name__} has an invalid version")

    name = cls.__name__ # get the name of the class
    src_file = Path(inspect.getfile(cls)).resolve() # get the source file of the class (where the user declared the python code)
    out_dir = src_file.parent / "__Threadable__" # get the output directory for the Threadable C++ code
    out_dir.mkdir(parents=True, exist_ok=True)
    hpp_path = out_dir / f"{name}.hpp" # get the path to the Threadable header file
    cpp_path = out_dir / f"{name}.cpp" # get the path to the Threadable source file

    STORE[name] = str(hpp_path) # store the path to the Threadable header file in the STORE

    src_hash = source_fingerprint(cls, *methods) # get the hash of the class (used to check if the code was changed comapred to the prev compile)
    units = (cache or {}).setdefault("units", {})
    cached = units.get(name, {})
    outputs_ok = hpp_path.is_file() and cpp_path.is_file() # verify if the output files exist

    thread_dir = src_file.parent / "__Thread__" # get the output directory for the Thread C++ code
    thread_dir.mkdir(parents=True, exist_ok=True) # create the output directory if it doesn't exist
    export_path = thread_dir / "cthreads_export.hpp" # get the path to the Thread export header file

    if ( # if the output files are up to date and the force flag is not set
        not force
        and outputs_ok
        and cached.get("src_hash") == src_hash
    ):
        for fn in methods: # build the kernel meta for each method
            export = f"{name}_{fn.__name__}"
            build_kernel_meta(fn, symbol=export, owner_name=name, owner_cls=cls)
        write_if_changed(export_path, _EXPORT_HPP)
        return False # false means no changes were made (the above code is only a failsafe)

    includes: list[str] = []
    fields: list[str] = []
    seen_includes: set[str] = set()

    # iter the class fileds/attributes and add them to the fields list
    for field_name, hint in get_type_hints(cls).items():
        py_type = hint_to_pytype(hint) # convert the hint to a PyType
        decl, include = py_type.to_cpp(field_name) # convert the field to a C++ declaration (returns type name and include)
        fields.append(f"    {decl}") # add the declaration to the fields list
        for line in include.splitlines(keepends=True): # iter over the inlcudes that this field needs and collect them
            if line and line not in seen_includes:
                seen_includes.add(line)
                includes.append(line)

    # translate the methods to C++ code
    method_results = [translate_thread(fn, owner_name=name) for fn in methods]
    # iter the method results and add the includes to the includes list
    for result in method_results:
        for line in result.sig_includes:
            if line and line not in seen_includes:
                if name in line and "__Threadable__" in line:
                    continue
                seen_includes.add(line)
                includes.append(line)

    # build the include block
    include_block = "".join(includes)
    field_block = "\n".join(fields)
    if field_block:
        field_block += "\n"

    # build the method declaration block
    method_decls = "\n".join(r.method_decl() for r in method_results)
    if method_decls:
        method_decls += "\n"

    c_wrappers_decl: list[str] = []
    c_wrappers_def: list[str] = []
    trampoline_defs: list[str] = []
    trampoline_decls: list[str] = []
    # build the code for the methods
    for fn, result in zip(methods, method_results):
        export = f"{name}_{result.func_name}"
        params = result.params_csv
        c_params = f"{name}* self" + (f", {params}" if params else "")
        c_sig = f"CTHREADS_API {result.return_type} {export}({c_params})"
        c_wrappers_decl.append(f"{c_sig};")
        call_args = ", ".join(
            part.strip().split()[-1].lstrip("&*")
            for part in params.split(",")
            if part.strip()
        ) if params.strip() else ""
        if result.return_type == "void":
            body = f"    self->{result.func_name}({call_args});\n"
        else:
            body = f"    return self->{result.func_name}({call_args});\n"
        c_wrappers_def.append(f"{c_sig} {{\n{body}}}")

        meta = build_kernel_meta(
            fn, symbol=export, owner_name=name, owner_cls=cls
        )
        trampoline_decls.append(emit_trampoline_decls(meta))
        trampoline_defs.append(emit_trampoline_cpp(meta, real_call=export))

    write_if_changed(export_path, _EXPORT_HPP) # write the code (internally it check sif write is needed)

    # build the header file
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
    for result in method_results:
        for line in result.body_includes:
            if line and line not in body_extra_seen:
                if name in line and "__Threadable__" in line:
                    continue
                body_extra_seen.add(line)
                cpp += line
        cpp += f"\n{result.method_def_signature(name)} {{\n{result.body}}}\n"
    for wrapper in c_wrappers_def:
        cpp += f"\n{wrapper}\n"
    for block in trampoline_defs:
        cpp += "\n" + block

    wrote_hpp = write_if_changed(hpp_path, hpp) # write the header file (internally it check sif write is needed)
    wrote_cpp = write_if_changed(cpp_path, cpp) # write the source file (internally it check sif write is needed)
    changed = wrote_hpp or wrote_cpp # check if any files were written

    units[name] = {
        "src_hash": src_hash,
        "hpp": str(hpp_path),
        "cpp": str(cpp_path),
    }
    return changed 


def compile(cls: type) -> None:
    """
    Compile a Threadable class.
    """
    compile_threadable(cls, methods=[], force=True)
