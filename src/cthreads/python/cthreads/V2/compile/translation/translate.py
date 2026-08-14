"""One-shot: parse → Signature → Syntax.stmt → TranslationResult."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .Source import Source
from .Signature import Signature
from .syntax import Syntax
from .context import TranslationContext
from .result import TranslationResult

if TYPE_CHECKING:
    from ..orchestrator.units import ThreadableUnit


def translate_function(
    fn: Callable,
    *,
    this_file: Path,
    owner: ThreadableUnit | None = None,
) -> TranslationResult:
    """
    Translate one `@Thread` function into a `TranslationResult`.

    `this_file` is the generated `.hpp` path used for relative includes.
    `owner` is set for methods of a `@Threadable` class.
    """
    func_def = Source.parse_function(fn)
    ctx = TranslationContext(fn=fn, this_file=this_file, owner=owner)
    sig = Signature.translate(func_def, ctx)
    body_lines: list[str] = []
    for stmt in func_def.body:
        body_lines.extend(Syntax.stmt(stmt, ctx))
    body = "\n".join(body_lines) + ("\n" if body_lines else "")
    return TranslationResult(
        return_type=sig.return_type,
        func_name=sig.func_name,
        params_csv=sig.params_csv,
        sig_includes=list(ctx.sig_includes),
        body=body,
        body_includes=list(ctx.body_includes),
    )
