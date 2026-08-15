"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Shared helpers for Thread function codegen.

Python's `ast` module turns source text into a tree of nodes. Example:

    def move(p: Particle, dt: float) -> None:
        scale: float = 1.0
        p.x += dt

becomes roughly:

    Module(
      body=[
        FunctionDef(
          name='move',
          args=arguments(args=[arg('p'), arg('dt')], ...),
          body=[
            AnnAssign(target=Name('scale'), annotation=Name('float'),
                      value=Constant(1.0)),
            AugAssign(target=Attribute(Name('p'), 'x'), op=Add(),
                      value=Name('dt')),
          ],
        )
      ]
    )

We walk that tree and emit C++. Nested helpers below are the small utilities
used while doing that walk.
"""

import ast
import inspect
import textwrap
from typing import Any

from ...pyTypes import PyThreadable, PyType
from ...Threadable.lib import is_threadable


def add_include(bucket: list[str], seen: set[str], text: str) -> None:
    """
    Append unique `#include ...` lines (text may contain several).
    
    #### Args
    - bucket: list[str] - the list of includes to add to
    - seen: set[str] - the set of includes that have already been added
    - text: str - the text to add the includes to
    """
    for line in text.splitlines(keepends=True):
        if line and line not in seen:
            seen.add(line)
            bucket.append(line)


def include_for(py_type: PyType) -> str:
    """
    Resolves the `#include...` string needed to use this type in generated C++.

    #### Args
    - py_type: PyType - the type to get the include for

    #### Returns
    - str - the `#include ...` string needed to use this type in generated C++.
    """
    # if this is a Threadable type defined by the user (not a stdlib type)
    # we include the generated headerfile for the user defined type
    # the headerfile is located in the `__Threadable__` subfolder nextto the file that the python type is defined in
    # the file is always named after the type the user defined
    if isinstance(py_type, PyThreadable):
        return f'#include "../__Threadable__/{py_type.cpp_name}.hpp"\n'
    # if this is a stdlib type, the py_type defines its own build include method
    return py_type.build_include()


def cpp_literal(value: Any) -> str:
    """
    Map a Python runtime value (from `ast.Constant.value`) to a C++ literal.

    in ast nodes, literals are represented as `ast.Constant` nodes with a `value` attribute
    this function maps the value to a C++ literal

    #### Args
    - value: Any - the value to map to a C++ literal

    #### Returns
    - str - the C++ literal
    """
    # if the value is a bool, return true or false
    if isinstance(value, bool):
        # bool is a subclass of int in Python — check before int-like handling
        return "true" if value else "false"
    # if the value is a string, escape the string and return it
    if isinstance(value, str):
        # escape the string by replacing \ with \\ and " with \"
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    # if the value is None, raise an error as its not safe
    if value is None:
        # None is not a supported Thread literal
        raise TypeError("None is not a supported Thread literal")
    return str(value) # any other value defaults to a string


def resolve_annotation(node: ast.expr, globals_ns: dict[str, Any]) -> Any:
    """
    Turn an annotation AST node into a real Python type object.

    #### Example: 
    
    takes an `ast.expr` node like `ast.Name(id='float')` and resolves it to `"float"`,
    the uses buildin eval to map the string name against the object type from
    the globals namespace and finally checks if the type is a valid threadable type.
    If the type is not a valid threadable type, an error is raised, otherwise the type is returned.

    #### Args
    - node: ast.expr - the node to resolve the annotation for
    - globals_ns: dict[str, Any] - the global namespace to resolve the annotation in

    #### Returns
    - Any - the resolved annotation
    """
    # unparse the node to turn the nodes identifier into a string
    # eg. `ast.Name(id='float')` -> `"float"`
    # then use the building eval to resolve the string name against the object type from
    # the globals namespace in the given context
    # NOTE: we use {} for the local namespace since we dont have any local variables
    hint = eval(ast.unparse(node), globals_ns, {})
    # check if the hint (object type) is a valid threadable type
    is_threadable(hint)
    # return the hint
    return hint


def parse_function_def(fn) -> ast.FunctionDef:
    """
    converts a python function into an ast.FunctionDef node

    #### Args
    - fn: function - the function to convert to an ast.FunctionDef node

    #### Returns
    - ast.FunctionDef - the ast.FunctionDef node
    """
    # convert the function src to a string and remove indentation
    source = textwrap.dedent(inspect.getsource(fn))
    # use ast.parse to convert the src string into an ast tree
    tree = ast.parse(source)
    # return the first function def node in the tree
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef))
