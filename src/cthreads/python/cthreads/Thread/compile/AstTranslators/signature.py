"""
Build C++ signature parts from `FunctionDef` args + return annotation.
"""

import ast
from dataclasses import dataclass

from ....CONFIG import STORE
from ....pyTypes import PyThreadable, hint_to_pytype
from ..lib import add_include, include_for
from .context import TranslateContext


@dataclass
class SignatureParts:
    return_type: str
    func_name: str
    params_csv: str  # contents inside (...)


def translate_signature(func_def: ast.FunctionDef, ctx: TranslateContext) -> SignatureParts:
    params: list[str] = []
    args = list(func_def.args.args)

    # Methods: drop `self` from the C++ parameter list; keep it in symbols for body.
    if ctx.owner_name and args and args[0].arg == "self":
        location = STORE.get(ctx.owner_name, f"__Threadable__/{ctx.owner_name}.hpp")
        ctx.symbols["self"] = PyThreadable(ctx.owner_name, location)
        args = args[1:]

    for arg in args:
        if arg.arg not in ctx.hints:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"parameter {arg.arg!r} needs a type annotation"
            )
        hint = ctx.hints[arg.arg]
        py_type = hint_to_pytype(hint)
        ctx.symbols[arg.arg] = py_type
        add_include(ctx.sig_includes, ctx.seen_sig, include_for(py_type))
        if isinstance(py_type, PyThreadable):
            params.append(f"{py_type.cpp_name}& {arg.arg}")
        else:
            params.append(f"{py_type.cpp_name} {arg.arg}")

    if func_def.args.vararg or func_def.args.kwarg or func_def.args.kwonlyargs:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            "*args/**kwargs/kw-only args are not supported"
        )

    ret_hint = ctx.hints.get("return", None)
    if ret_hint is None or ret_hint is type(None):
        ret_cpp = "void"
    else:
        ret_type = hint_to_pytype(ret_hint)
        add_include(ctx.sig_includes, ctx.seen_sig, include_for(ret_type))
        ret_cpp = ret_type.cpp_name

    return SignatureParts(
        return_type=ret_cpp,
        func_name=ctx.func_name,
        params_csv=", ".join(params),
    )
