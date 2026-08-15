"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Emit free @Thread functions to __Thread__/name.{hpp,cpp}.
Method emission is owned by Threadable.compile (same class files).
"""

import inspect
from pathlib import Path
from typing import Any, Optional, get_type_hints

from ...CONFIG import STORE, VERSION
from ...cache import source_fingerprint, write_if_changed
from ...kernel_meta import build_kernel_meta, emit_trampoline_cpp, emit_trampoline_decls
from .AstTranslators import translate_function
from .AstTranslators.translate import TranslateResult
from .lib import parse_function_def

# defines the export header for the cthreads hpp
# includes the export macro to turn CTHREADS_API into a dll export macro based on the platform
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


def translate_thread(fn, owner_name: Optional[str] = None) -> TranslateResult:
    """
    Takes a python function that was wrapped with @Thread and translates it into a c++ function

    #### Args
    - fn: function - the function to translate
    - owner_name: Optional[str] - the name of the owner of the function (if the function is a method of a @Threadable class)

    #### Returns
    - TranslateResult - the result of the translation

    #### Example:
    ```python
    @Thread
    def my_function(x: int) -> int:
        return x + 1
    ```
    will be translated into:
    ```cpp
    CTHREADS_API int my_function(int x) {
        return x + 1;
    }
    ```
    """
    if not getattr(fn, "__threaded", False):
        raise TypeError(f"Function {fn.__name__} is not a Thread function")
    if getattr(fn, "__thread_version", "") != VERSION:
        raise TypeError(f"Function {fn.__name__} has an invalid version")

    func_def = parse_function_def(fn) # cnvert the function to an ast.FunctionDef node
    hints = get_type_hints(fn) # get type hints from the function
    # translate the function to a c++ function
    return translate_function(fn, func_def, hints, owner_name=owner_name) # returns a TranslateResult object


def compile_free_thread(
    fn,
    force: bool = False,
    cache: dict[str, Any] | None = None,
) -> bool:
    """
    Takes a free @Thread function and compiles it into a c++ function.

    if the function was previously compiled, no chnages where made and force is False, the function returns False and doesnt write code

    #### Args
    - fn: function - the function to compile
    - force: bool - if True, the function will compile the function even if it was previously compiled
    - cache: dict[str, Any] | None - the cache to store compilation data in (uses the cache to chekc if recompiling is required)

    #### Returns
    - bool - True if the function was compiled, False if it was not compiled
    """
    name = fn.__name__ # get the name of the function
    src_file = Path(inspect.getfile(fn)).resolve() # the path to the file that the function is defined in
    out_dir = src_file.parent / "__Thread__" # the directory to store the compiled function in
    out_dir.mkdir(parents=True, exist_ok=True) # ensure the __Thread__ dir exists
    
    # prep the c++ code paths and files
    hpp_path = out_dir / f"{name}.hpp"
    cpp_path = out_dir / f"{name}.cpp"
    export_path = out_dir / "cthreads_export.hpp"

    src_hash = source_fingerprint(fn) # get the source fingerprint of the function (used to check for src updates and recompilation necessity)
    units = (cache or {}).setdefault("units", {}) # prep the cache
    cached = units.get(name, {}) # get the cached data for the function
    outputs_ok = hpp_path.is_file() and cpp_path.is_file() # check if the output files exist

    STORE[name] = str(hpp_path) # store the path to the hpp file in the store

    if (
        not force
        and outputs_ok
        and cached.get("src_hash") == src_hash
    ):
        # Fast path: sources unchanged — refresh kernel meta only.
        build_kernel_meta(fn, symbol=name, owner_name=None)
        write_if_changed(export_path, _EXPORT_HPP)
        return False

    # ---- recompile is necessary ----

    # translate the function to a c++ function
    result = translate_thread(fn, owner_name=None)
    signature = result.free_signature() # get the free signature of the function (CTHREADS_API + return type + function name + parameters)
    meta = build_kernel_meta(fn, symbol=name, owner_name=None) # build the kernel meta data for the function

    seen_sig = set(result.sig_includes) # add the signature includes to a set to avoid duplicates
    extra_body = "".join( # join includes from types that where used in the function body
        line for line in result.body_includes if line not in seen_sig
    )

    write_if_changed(export_path, _EXPORT_HPP) # write the export header to the export path

    # ---- write the c++ code to the output files ----

    hpp = "#pragma once\n\n" # start the hpp file with a pragma once and include the export header
    hpp += '#include "cthreads_export.hpp"\n\n' # include the export header (hlds the export macro)
    hpp += "".join(result.sig_includes)
    if result.sig_includes:
        hpp += "\n"
    hpp += f"{signature};\n"
    hpp += emit_trampoline_decls(meta) # emit the trampoline declarations for the function

    cpp = f'#include "{name}.hpp"\n'
    if extra_body:
        cpp += "\n" + extra_body
    cpp += f"\n{signature} {{\n{result.body}}}\n"
    cpp += "\n" + emit_trampoline_cpp(meta, real_call=name)

    wrote_hpp = write_if_changed(hpp_path, hpp)
    wrote_cpp = write_if_changed(cpp_path, cpp)
    changed = wrote_hpp or wrote_cpp

    units[name] = { # store the compilation data in the cache
        "src_hash": src_hash,
        "hpp": str(hpp_path),
        "cpp": str(cpp_path),
    }
    return changed # return True if the files were changed, False if they were not


def compile(fn) -> None:
    """
    Compiles a free @Thread function and all its dependencies.
    """
    compile_free_thread(fn, force=True)
