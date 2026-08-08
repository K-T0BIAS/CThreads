"""Shared test helpers (importable; conftest re-exports fixtures only)."""

from __future__ import annotations

import ast
import textwrap
from typing import Any

from cthreads.Thread.compile.AstTranslators.context import TranslateContext


def make_ctx(
    func_name: str = "f",
    *,
    symbols: dict[str, Any] | None = None,
    owner_name: str | None = None,
    hints: dict[str, Any] | None = None,
) -> TranslateContext:
    class _DummyFn:
        __name__ = func_name
        __globals__ = {
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "str": str,
        }

    return TranslateContext(
        fn=_DummyFn(),
        func_name=func_name,
        hints=hints or {},
        owner_name=owner_name,
        symbols=symbols or {},
    )


def parse_expr(src: str) -> ast.expr:
    return ast.parse(src, mode="eval").body


def parse_stmt(src: str) -> ast.stmt:
    body = ast.parse(textwrap.dedent(src)).body
    assert len(body) == 1
    return body[0]


def make_threadable_type(name: str = "Particle") -> type:
    return type(
        name,
        (),
        {
            "__threadable": True,
            "__annotations__": {"x": float, "y": float, "velocity": float},
        },
    )
