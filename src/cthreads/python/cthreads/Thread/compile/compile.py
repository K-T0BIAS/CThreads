"""
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
    if not getattr(fn, "__threaded", False):
        raise TypeError(f"Function {fn.__name__} is not a Thread function")
    if getattr(fn, "__thread_version", "") != VERSION:
        raise TypeError(f"Function {fn.__name__} has an invalid version")

    func_def = parse_function_def(fn)
    hints = get_type_hints(fn)
    return translate_function(fn, func_def, hints, owner_name=owner_name)


def compile_free_thread(
    fn,
    force: bool = False,
    cache: dict[str, Any] | None = None,
) -> bool:
    """
    Generate free-thread C++. Returns True if any output file was rewritten.
    """
    name = fn.__name__
    src_file = Path(inspect.getfile(fn)).resolve()
    out_dir = src_file.parent / "__Thread__"
    out_dir.mkdir(parents=True, exist_ok=True)
    hpp_path = out_dir / f"{name}.hpp"
    cpp_path = out_dir / f"{name}.cpp"
    export_path = out_dir / "cthreads_export.hpp"

    src_hash = source_fingerprint(fn)
    units = (cache or {}).setdefault("units", {})
    cached = units.get(name, {})
    outputs_ok = hpp_path.is_file() and cpp_path.is_file()

    STORE[name] = str(hpp_path)

    if (
        not force
        and outputs_ok
        and cached.get("src_hash") == src_hash
    ):
        # Fast path: sources unchanged — refresh kernel meta only.
        build_kernel_meta(fn, symbol=name, owner_name=None)
        write_if_changed(export_path, _EXPORT_HPP)
        return False

    result = translate_thread(fn, owner_name=None)
    signature = result.free_signature()
    meta = build_kernel_meta(fn, symbol=name, owner_name=None)

    seen_sig = set(result.sig_includes)
    extra_body = "".join(
        line for line in result.body_includes if line not in seen_sig
    )

    write_if_changed(export_path, _EXPORT_HPP)

    hpp = "#pragma once\n\n"
    hpp += '#include "cthreads_export.hpp"\n\n'
    hpp += "".join(result.sig_includes)
    if result.sig_includes:
        hpp += "\n"
    hpp += f"{signature};\n"
    hpp += emit_trampoline_decls(meta)

    cpp = f'#include "{name}.hpp"\n'
    if extra_body:
        cpp += "\n" + extra_body
    cpp += f"\n{signature} {{\n{result.body}}}\n"
    cpp += "\n" + emit_trampoline_cpp(meta, real_call=name)

    wrote_hpp = write_if_changed(hpp_path, hpp)
    wrote_cpp = write_if_changed(cpp_path, cpp)
    changed = wrote_hpp or wrote_cpp

    units[name] = {
        "src_hash": src_hash,
        "hpp": str(hpp_path),
        "cpp": str(cpp_path),
    }
    return changed


def compile(fn) -> None:
    compile_free_thread(fn, force=True)
