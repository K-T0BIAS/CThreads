"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Build C++ signature parts from `FunctionDef` args + return annotation.
"""

import ast
from dataclasses import dataclass

from ....CONFIG import STORE
from ....pyTypes import PyDict, PyList, PyThreadable, hint_to_pytype, is_tbuffer_pytype
from ..lib import add_include, include_for
from .context import TranslateContext


@dataclass
class SignatureParts:
    """
    A dataclass to store the signature parts of a function

    #### Attributes
    - return_type: str - the return type of the function
    - func_name: str - the name of the function
    - params_csv: str - the parameters of the function as a comma separated string
    """
    return_type: str
    func_name: str
    params_csv: str  # contents inside (...)


def translate_signature(func_def: ast.FunctionDef, ctx: TranslateContext) -> SignatureParts:
    """
    Takes an ast.FunctionDef node and a TranslateContext object and returns a SignatureParts object, containing
    the return type, the name of the function, and the parameters of the function as a comma separated string

    #### Args
    - func_def: ast.FunctionDef - the function definition to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - SignatureParts - a SignatureParts object containing the return type, the name of the function, and the parameters of the function as a comma separated string
    """
    params: list[str] = []
    args = list(func_def.args.args) # get the arguments from the function definition

    # Methods: drop `self` from the C++ parameter list; keep it in symbols for body. (c++ uses this ptr instead)
    if ctx.owner_name and args and args[0].arg == "self":
        location = STORE.get(ctx.owner_name, f"__Threadable__/{ctx.owner_name}.hpp")
        ctx.symbols["self"] = PyThreadable(ctx.owner_name, location)
        args = args[1:]

    for arg in args:
        if arg.arg not in ctx.hints: # the arg wasnt hinted so it is unsupported (safety check)
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"parameter {arg.arg!r} needs a type annotation"
            )
        # get the hint for the argument
        hint = ctx.hints[arg.arg]
        py_type = hint_to_pytype(hint) # convert the hint to a pytype
        ctx.symbols[arg.arg] = py_type
        add_include(ctx.sig_includes, ctx.seen_sig, include_for(py_type)) # add the include for the pytype
        # Mutables bind pack slots by ref; triple buffers pass a native pointer.
        if is_tbuffer_pytype(py_type):
            params.append(f"{py_type.cpp_name}& {arg.arg}")
        elif isinstance(py_type, (PyThreadable, PyList, PyDict)):
            params.append(f"{py_type.cpp_name}& {arg.arg}")
        else:
            params.append(f"{py_type.cpp_name} {arg.arg}")

    # vararg, kwarg, and kwonlyargs are not supported
    if func_def.args.vararg or func_def.args.kwarg or func_def.args.kwonlyargs:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            "*args/**kwargs/kw-only args are not supported"
        )

    # check if the function has a return type
    ret_hint = ctx.hints.get("return", None)
    # no return type so void
    if ret_hint is None or ret_hint is type(None):
        ret_cpp = "void"
    else: # return type so convert to cpp type
        ret_type = hint_to_pytype(ret_hint)
        add_include(ctx.sig_includes, ctx.seen_sig, include_for(ret_type)) # add the include for the return type
        ret_cpp = ret_type.cpp_name # get the cpp name of the return type

    # return the SignatureParts object
    return SignatureParts(
        return_type=ret_cpp,
        func_name=ctx.func_name,
        params_csv=", ".join(params), # join the parameters into a comma separated string
    )
