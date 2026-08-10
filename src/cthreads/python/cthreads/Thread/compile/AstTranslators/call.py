"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.Call` - builtins + list/dict + sync methods + math.* / cthreads.math.*.

Python:  len(xs)          xs.append(v)     lock.acquire()   math.sqrt(x)
C++:     (xs).size()      (xs).push_back(v) (lock).acquire()  std::sqrt(x)
"""

from __future__ import annotations

import ast

from ....pyOps import is_builtin_call
from ..lib import add_include
from ..mathLibTranslators import CMATH_INCLUDE, resolve_math_call
from ..pythonContainerLibTranslators import resolve_container_op
from ..syncBindingTranslators import resolve_sync_op
from .context import TranslateContext


def translate(node: ast.Call, ctx: TranslateContext) -> str:
    """
    Translates call expressions to a C++ expression

    #### Args
    - node: ast.Call - the call node to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - str - a C++ expression that represents the translated ast.Call node
    """
    from .translate import translate_expr

    # if the call was len then translate it to (xs).size()
    if is_builtin_call(node, "len"):
        if node.keywords: # len cant have keyword args
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                "len() keyword args are not supported"
            )
        if len(node.args) != 1: # len expects 1 arg
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"len() expects 1 arg, got {len(node.args)}"
            )
        # translate the argument
        arg = translate_expr(node.args[0], ctx)
        # return the translated argument wrapped in parentheses and size()
        # since we use std::vector<T> we can use size() to get the length of the container
        return f"({arg}).size()"

    # list/dict methods: resolve picks the op; translate recv+args here (same as math).
    # Emit lambdas stay pure C++-string templates — do not call translate_expr inside them.
    container_op = resolve_container_op(node, ctx)
    if container_op is not None:
        assert isinstance(node.func, ast.Attribute)  # guaranteed by resolve
        if container_op.cpp_include:
            add_include(
                ctx.body_includes,
                ctx.seen_body,
                f'#include "{container_op.cpp_include}"\n',
            )
        recv = translate_expr(node.func.value, ctx)
        args = [translate_expr(a, ctx) for a in node.args]
        return container_op.emit(recv, args)

    # cthreads.sync.Lock / Event / RWLock methods on a typed Name receiver.
    sync_op = resolve_sync_op(node, ctx)
    if sync_op is not None:
        assert isinstance(node.func, ast.Attribute)
        if sync_op.cpp_include:
            add_include(
                ctx.body_includes,
                ctx.seen_body,
                f'#include "{sync_op.cpp_include}"\n',
            )
        recv = translate_expr(node.func.value, ctx)
        args = [translate_expr(a, ctx) for a in node.args]
        return sync_op.emit(recv, args)

    # resolve the math call
    op = resolve_math_call(node, ctx)
    if op is None: # unsupported math call
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported call (only whitelisted builtins / "
            f"list|dict|sync methods / math.* / cthreads.math.*)"
        )

    # add the include for the math call
    if op.cpp_include: # if the math call has a cpp include
        add_include(ctx.body_includes, ctx.seen_body, f'#include "{op.cpp_include}"\n')
    else: # if the math call doesnt have a cpp include use the default math include
        add_include(ctx.body_includes, ctx.seen_body, CMATH_INCLUDE)

    # translate the arguments
    args = ", ".join(translate_expr(a, ctx) for a in node.args)
    # return the translated arguments wrapped in the cpp function
    return f"{op.cpp_func}({args})"
