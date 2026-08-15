import ast
import inspect
import textwrap
from typing import Any

from ...types import hint_to_pytype


class Source:
    """Python source / annotation helpers for translation."""

    @staticmethod
    def parse_function(fn) -> ast.FunctionDef:
        source = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(source)
        return next(node for node in tree.body if isinstance(node, ast.FunctionDef))

    @staticmethod
    def resolve_annotation(node: ast.expr, globals_ns: dict[str, Any]) -> Any:
        hint = eval(ast.unparse(node), globals_ns, {})
        hint_to_pytype(hint)
        return hint
