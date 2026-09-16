"""
AttrPlugin: ThreadIdx / BlockIdx / GlobalIdx / ... .x|.y|.z -> GLSL builtins.
"""
import ast
from typing import Any

from ....frontend.indexes import GpuIndexBuiltin
from ..context import GpuTranslationContext
from .base import AttrPlugin, TranslateExpr

_AXES: frozenset[str] = frozenset({"x", "y", "z"})


def _globals(ctx: GpuTranslationContext) -> dict:
    g = getattr(ctx.fn, "__globals__", None)
    return g if isinstance(g, dict) else {}


def _resolve_value_obj(node: ast.expr, globals_ns: dict) -> Any:
    """
    Resolve a simple Name or module.Name expression to a Python object.
    """
    if isinstance(node, ast.Name):
        return globals_ns.get(node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        mod = globals_ns.get(node.value.id)
        if mod is None:
            return None
        return getattr(mod, node.attr, None)
    return None


def _glsl_base_for(obj: Any) -> str | None:
    if obj is None:
        return None
    cls = obj if isinstance(obj, type) else type(obj)
    if not isinstance(cls, type) or not issubclass(cls, GpuIndexBuiltin):
        return None
    base: str = getattr(cls, "_glsl_base", "")
    return base or None


class IndexAttrPlugin(AttrPlugin):
    """
    Lower `GlobalIdx.x` / `gpu.ThreadIdx.y` to `gl_*Invocation*.*`.
    """

    def try_lower(
        self,
        node: ast.Attribute,
        ctx: GpuTranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        if node.attr not in _AXES:
            return None
        obj = _resolve_value_obj(node.value, _globals(ctx))
        base = _glsl_base_for(obj)
        if base is None:
            return None
        # GLSL invocation IDs are uint; cast to int so `i: int = GlobalIdx.x` typechecks.
        return f"int({base}.{node.attr})"
