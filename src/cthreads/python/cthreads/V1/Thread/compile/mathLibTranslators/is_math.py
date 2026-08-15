"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Detect `math` / `cthreads.math` calls for lowering.

Resolve names through `ctx.fn.__globals__`. Stdlib math uses `__module__`;
`cthreads.math` is marked `__cthreads_internal__` on the *module* (pybind11
bound functions cannot carry that attribute).
"""

import ast
import sys
from typing import Any, Optional

from ..AstTranslators.context import TranslateContext
from .mathOps import CTHREADS_MATHOPS, MATHCONSTS, MATHOPS, MathOp


def _globals(ctx: TranslateContext) -> dict[str, Any]:
    f"""
    returns a dict of global variables for the function using the functions translate context

    NOTE: __globals__ is a list of global variables the function has captured

    #### Args
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - dict[str, Any] - a dict of global variables for the function (empty dict if the function has no globals)
    """
    fn = getattr(ctx, "fn", None) # get the function from the translate context
    g = getattr(fn, "__globals__", None) # get the globals from the function
    return g if isinstance(g, dict) else {} # return the globals if they are a dict, otherwise return an empty dict


def _resolve_call_obj(node: ast.Call, ctx: TranslateContext) -> tuple[Any, Any]:
    """
    From an ast.Call node, resolve the callable and parent module

    #### Args
    - node: ast.Call - the call node to resolve
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - tuple[Any, Any] - a tuple of the callable and parent module (None, None if the callable or parent module is not found)
    """
    func = node.func # get the function from the call node

    # if the function is an attribute and the vaue is a Name
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        # resolve the globals and get the value for the node function id
        mod = _globals(ctx).get(func.value.id)
        # if this is none then its not a function and theres no module
        if mod is None:
            return None, None
        # otherwise try to get the function as an attribute of the module
        return getattr(mod, func.attr, None), mod
    # if the function is an ast.Name then its a global variable
    if isinstance(func, ast.Name):
        # resolve the globals and get the value for the node function id
        return _globals(ctx).get(func.id), None
    # if the function is not an attribute or name then its not a function and theres no module
    return None, None


def _stdlib_math_op(obj: Any) -> Optional[MathOp]:
    """
    Checks if a given Object is a math operation from python's stdlib math module

    #### Args
    - obj: Any - the object to check
    
    #### Returns
    - Optional[MathOp] - the math operation if the object is a math operation from python's stdlib math module, otherwise None
    """
    if obj is None: # if the object is None then return None
        return None
    # try to get the module of the object and check if its the math module
    if getattr(obj, "__module__", None) != "math":
        return None # not math module so not a math operation

    # at this point its a math operation from the stdlib math module
    # get the name of the operation
    name = getattr(obj, "__name__", None)
    # if the name is not a string or not in the MATHOPS dict then return None
    if not isinstance(name, str) or name not in MATHOPS:
        return None # not a valid math operation
    # return the math operation
    return MATHOPS[name]


def _owner_is_cthreads_math(obj: Any, parent_mod: Any) -> bool:
    """
    Checks if an Object is a math operation from the cthreads.math module

    #### Args
    - obj: Any - the object to check
    - parent_mod: Any - the parent module of the object (may be None)

    #### Returns
    - bool - True if the object is a math operation from the cthreads.math module, otherwise False
    """
    # if theres a parent module and the object has the __cthreads_internal__ attribute then its a math operation from the cthreads.math module
    if parent_mod is not None and getattr(parent_mod, "__cthreads_internal__", False):
        return True

    # if the object has the __cthreads_internal__ attribute then its a math operation from the cthreads.math module
    if getattr(obj, "__cthreads_internal__", False):
        return True
    # get the module name of the object
    mod_name = getattr(obj, "__module__", None)
    if isinstance(mod_name, str):
        # get the module from the module name
        mod = sys.modules.get(mod_name)
        # if the module is not None and the module has the __cthreads_internal__ attribute then its a math operation from the cthreads.math module
        if mod is not None and getattr(mod, "__cthreads_internal__", False):
            return True
    return False


def _cthreads_math_op(obj: Any, parent_mod: Any = None) -> Optional[MathOp]:
    """
    Validates if an Object comes from the cthreads.math module and returns the MathOp if it does

    #### Args
    - obj: Any - the object to check
    - parent_mod: Any - the parent module of the object (may be None)

    #### Returns
    - Optional[MathOp] - the math operation if the object is a math operation from the cthreads.math module, otherwise None
    """
    if obj is None: # no object so no math operation
        return None
    name = getattr(obj, "__name__", None) # get the name of the object
    # if the name is not a string or not in the CTHREADS_MATHOPS dict then return None
    if not isinstance(name, str) or name not in CTHREADS_MATHOPS:
        return None
    # validate if the object comes from the cthreads.math module
    if not _owner_is_cthreads_math(obj, parent_mod):
        return None
    # return the math operation
    return CTHREADS_MATHOPS[name]


def resolve_math_call(node: ast.AST, ctx: TranslateContext) -> Optional[MathOp]:
    """
    Takes an ast.AST node and a TranslateContext and returns the MathOp if the node is a math operation

    #### Args
    - node: ast.AST - the ast node to check
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - Optional[MathOp] - the math operation if the node is a math operation, otherwise None
    """
    if not isinstance(node, ast.Call): # the node is not a call so no math operation
        return None
    if node.keywords: # the node has keywords so no math operation
        return None

    obj, parent = _resolve_call_obj(node, ctx) # resolves the python object and module (either or both may be None)
    # try to get the math operation from the cthreads.math module or the stdlib math module
    op = _cthreads_math_op(obj, parent) or _stdlib_math_op(obj)
    if op is None: # no math operation found
        return None
    if len(node.args) != op.arity: # the number of arguments does not match the arity of the math operation
        return None
    return op # return the math operation


def resolve_math_const(node: ast.AST, ctx: TranslateContext) -> Optional[str]:
    """
    Resolves a math constant from an ast.AST node and a TranslateContext

    #### Args
    - node: ast.AST - the ast node to check
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - Optional[str] - the math constant if the node is a math constant, otherwise None
    """
    if not isinstance(node, ast.Attribute): # the node is not an attribute so no math constant
        return None
    if not isinstance(node.value, ast.Name): # the value of the attribute is not a name so no math constant
        return None
    if node.attr not in MATHCONSTS: # the attribute is not a math constant
        return None
    # get the module from the globals (where this name was used to define a global variable)
    mod = _globals(ctx).get(node.value.id)
    if mod is None: # no module so this cant be resolved
        return None
    # if the module is not the math module then this cant be a math constant 
    if getattr(mod, "__name__", None) != "math":
        return None
    # without an attr this cant be resolved to MATHCONSTS
    if getattr(mod, node.attr, None) is None:
        return None
    return MATHCONSTS[node.attr] # return the math constant


def is_math(node: ast.AST, ctx: TranslateContext) -> bool:
    """
    Checks if an ast.AST node is a math operation from the cthreads.math module or the stdlib math module

    #### Args
    - node: ast.AST - the ast node to check
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - bool - True if the node is a math operation from the cthreads.math module or the stdlib math module, otherwise False
    """
    # is true if the node was resolved to a math operation otherwise false
    return resolve_math_call(node, ctx) is not None
