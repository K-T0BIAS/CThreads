import ast
import os
from pathlib import Path

from ...context import TranslationContext
from ...include import add_include
from ..base import CallPlugin, TranslateExpr
from .....frontend.Registry import REGISTRY

class FreeFnCallPlugin(CallPlugin):

    def try_lower(
        self,
        node: ast.Call,
        ctx: TranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        # ensure this is a function call (if not another plugin will handle this)
        if not isinstance(node.func, ast.Name):
            return None

        # get the object from the globals of the current node in ctx
        obj = ctx.fn.__globals__.get(node.func.id, None)
        # if the object is not found return None to let other plugins check against 
        # translated 3rd party libs functions (sin, len ...)
        if obj is None:
            return None
        # if the obj is not __threaded then its not valid per default
        # we return None to let other call based plugins check for edgecases
        if not getattr(obj, "__threaded", False):
            return None
        # get the registered unit from the registry to check calling context
        registered_unit = REGISTRY.thread_units.get(obj.__qualname__, None)
        # if this wasnt registered sth went wrong somwhere (if this happens we are screwed ...)
        if registered_unit is None:
            raise TypeError(f"Something went wrong. The function {node.func.id} is not registered in the registry.")
        # check if this fun is actually free (not owned by a class) [methods are handled somwehere else]
        if registered_unit.owner is not None:
            return None # this is a method call, so itll be rendered in attr
        # check if the fn was called with the libs syntax rules
        if node.keywords:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"{node.func.id} keyword args are not supported"
            )

        # add the hpp path of the import of the fn source file to the body of the file of this node
        hpp = registered_unit.hpp_path
        if hpp is not None:
            hpp = hpp.resolve()
            dest = Path(ctx.this_file).resolve()
            if hpp != dest:
                rel = os.path.relpath(hpp, dest.parent).replace("\\", "/")
                add_include(
                    ctx.body_includes,
                    ctx.seen_body,
                    f'#include "{rel}"\n',
                )

        # translate the arguments and return the call
        args = ", ".join(translate_expr(a, ctx) for a in node.args)
        return f"{obj.__name__}({args})"
