from __future__ import annotations

import ast
from typing import get_type_hints

from ...types import (
    PyDict,
    PyList,
    PyThreadable,
    hint_to_pytype,
    is_tbuffer_pytype,
)
from .context import TranslationContext
from .include import add_include, include_for
from .result import SignatureResult


class Signature:
    """Build C++ signature parts; fills ctx.symbols and ctx.sig_includes."""

    @staticmethod
    def translate(
        func_def: ast.FunctionDef, ctx: TranslationContext
    ) -> SignatureResult:
        localns: dict = {}
        if ctx.owner is not None:
            cls = ctx.owner.handle.target
            localns = {cls.__name__: cls}
        hints = get_type_hints(ctx.fn, localns=localns)

        params: list[str] = []
        args = list(func_def.args.args)

        # Methods: drop self from the C++ param list; keep it in symbols for the body.
        if ctx.owner is not None and args and args[0].arg == "self":
            ctx.symbols["self"] = PyThreadable(ctx.owner.handle.name)
            args = args[1:]

        for arg in args:
            if arg.arg not in hints:
                raise TypeError(
                    f"Thread function {ctx.func_name}: "
                    f"parameter {arg.arg!r} needs a type annotation"
                )
            py_type = hint_to_pytype(hints[arg.arg])
            ctx.symbols[arg.arg] = py_type
            add_include(
                ctx.sig_includes,
                ctx.seen_sig,
                include_for(py_type, ctx.this_file),
            )
            if is_tbuffer_pytype(py_type) or isinstance(
                py_type, (PyThreadable, PyList, PyDict)
            ):
                params.append(f"{py_type.cpp_name}& {arg.arg}")
            else:
                params.append(f"{py_type.cpp_name} {arg.arg}")

        if func_def.args.vararg or func_def.args.kwarg or func_def.args.kwonlyargs:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                "*args/**kwargs/kw-only args are not supported"
            )

        ret_hint = hints.get("return")
        if ret_hint is None or ret_hint is type(None):
            return_type = None
        else:
            return_type = hint_to_pytype(ret_hint)
            add_include(
                ctx.sig_includes,
                ctx.seen_sig,
                include_for(return_type, ctx.this_file),
            )

        return SignatureResult(
            return_type=return_type,
            func_name=func_def.name,
            params_csv=", ".join(params),
        )
