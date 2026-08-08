"""
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
    """Append unique `#include ...` lines (text may contain several)."""
    for line in text.splitlines(keepends=True):
        if line and line not in seen:
            seen.add(line)
            bucket.append(line)


def include_for(py_type: PyType) -> str:
    """
    `#include` needed to name this type in generated C++.

    Threadable types live next to generated Thread code as
    `../__Threadable__/Name.hpp`. Stdlib types use angle includes from PyType.
    """
    if isinstance(py_type, PyThreadable):
        return f'#include "../__Threadable__/{py_type.cpp_name}.hpp"\n'
    return py_type.build_include()


def cpp_literal(value: Any) -> str:
    """
    Map a Python runtime value (from `ast.Constant.value`) to a C++ literal.

    `ast.Constant` is how literals appear in the tree, e.g. `1.0`, `"hi"`, `True`.
    """
    if isinstance(value, bool):
        # bool is a subclass of int in Python — check before int-like handling
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if value is None:
        raise TypeError("None is not a supported Thread literal")
    return str(value)


def resolve_annotation(node: ast.expr, globals_ns: dict[str, Any]) -> Any:
    """
    Turn an annotation AST node into a real Python type object.

    Example: the `float` in `scale: float = 1.0` is `ast.Name(id='float')`.
    `ast.unparse` rebuilds the source fragment (`"float"` / `"list[int]"`),
    then `eval` resolves it in the function's globals (so user Threadables work).
    """
    hint = eval(ast.unparse(node), globals_ns, {})
    is_threadable(hint)
    return hint


def parse_function_def(fn) -> ast.FunctionDef:
    """
    Load the function's source and return its `ast.FunctionDef` node.

    `inspect.getsource` includes the `@Thread` decorator line(s). After
    `ast.parse`, the module body still contains a single `FunctionDef`
    (decorators hang off `func_def.decorator_list`; we ignore them here).
    """
    source = textwrap.dedent(inspect.getsource(fn))
    # Module.body is a list of top-level statements; we want the function.
    tree = ast.parse(source)
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef))
