"""
Workgroup barrier lowering for @Gpu.

Maps CUDA-style `__sync_threads()` and beginner-friendly
`Barrier.arrive_and_wait()` (no instance) to the same GLSL workgroup barrier.
Does not implement host lock semantics.
"""

from __future__ import annotations

import ast

from ..context import GpuTranslationContext
from .base import CallPlugin, TranslateExpr

# Single GLSL fragment used as an expression-statement body (trailing ; added
# by GpuFlow.expr_stmt). Workgroup scope only — not grid-wide.
_GPU_BARRIER_GLSL = "barrier(); memoryBarrierShared()"


def _require_no_args(node: ast.Call, ctx: GpuTranslationContext, label: str) -> None:
    if node.keywords or node.args:
        raise TypeError(
            f"GPU function {ctx.func_name}: {label} takes no arguments"
        )


class SyncThreadsPlugin(CallPlugin):
    """
    `__sync_threads()` / `Barrier.arrive_and_wait()` -> GLSL workgroup barrier.

    Rejects `Barrier(...)` construction inside @Gpu (use arrive_and_wait).
    """

    def try_lower(
        self,
        node: ast.Call,
        ctx: GpuTranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        del translate_expr  # barrier forms take no nested exprs
        fn = node.func

        # CUDA-style free call.
        if isinstance(fn, ast.Name) and fn.id == "__sync_threads":
            _require_no_args(node, ctx, "__sync_threads()")
            return _GPU_BARRIER_GLSL

        # Beginner form: Barrier.arrive_and_wait() — not Barrier(...).
        if (
            isinstance(fn, ast.Attribute)
            and fn.attr == "arrive_and_wait"
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "Barrier"
        ):
            _require_no_args(node, ctx, "Barrier.arrive_and_wait()")
            return _GPU_BARRIER_GLSL

        # Clear error if someone writes Barrier(n) or Barrier() in a @Gpu body.
        if isinstance(fn, ast.Name) and fn.id == "Barrier":
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                "Barrier(...) construction is not valid inside @Gpu; "
                "use Barrier.arrive_and_wait() or __sync_threads() "
                "(workgroup barrier)"
            )

        return None
