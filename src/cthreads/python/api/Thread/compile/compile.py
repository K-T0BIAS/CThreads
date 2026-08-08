"""
Emit free @Thread functions to __Thread__/name.{hpp,cpp}.
Method emission is owned by Threadable.compile (same class files).
"""

import inspect
from pathlib import Path
from typing import Optional, get_type_hints

from ...CONFIG import STORE, VERSION
from .AstTranslators import translate_function
from .AstTranslators.translate import TranslateResult
from .lib import parse_function_def


def translate_thread(fn, owner_name: Optional[str] = None) -> TranslateResult:
    if not getattr(fn, "__threaded", False):
        raise TypeError(f"Function {fn.__name__} is not a Thread function")
    if getattr(fn, "__thread_version", "") != VERSION:
        raise TypeError(f"Function {fn.__name__} has an invalid version")

    func_def = parse_function_def(fn)
    hints = get_type_hints(fn)
    return translate_function(fn, func_def, hints, owner_name=owner_name)


def compile_free_thread(fn) -> None:
    name = fn.__name__
    src_file = Path(inspect.getfile(fn)).resolve()
    out_dir = src_file.parent / "__Thread__"
    out_dir.mkdir(parents=True, exist_ok=True)
    hpp_path = out_dir / f"{name}.hpp"
    cpp_path = out_dir / f"{name}.cpp"

    result = translate_thread(fn, owner_name=None)
    signature = result.free_signature()

    seen_sig = set(result.sig_includes)
    extra_body = "".join(
        line for line in result.body_includes if line not in seen_sig
    )

    hpp = "#pragma once\n\n"
    hpp += "".join(result.sig_includes)
    if result.sig_includes:
        hpp += "\n"
    hpp += f"{signature};\n"

    cpp = f'#include "{name}.hpp"\n'
    if extra_body:
        cpp += "\n" + extra_body
    cpp += f"\n{signature} {{\n{result.body}}}\n"

    hpp_path.write_text(hpp, encoding="utf-8")
    cpp_path.write_text(cpp, encoding="utf-8")
    STORE[name] = str(hpp_path)


# Back-compat name if something still imports compile from this package
def compile(fn) -> None:
    compile_free_thread(fn)
