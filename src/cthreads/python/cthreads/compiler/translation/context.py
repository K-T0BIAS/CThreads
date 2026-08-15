from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ...types import PyType

if TYPE_CHECKING:
    from ..orchestrator.units import ThreadableUnit


@dataclass
class TranslationContext:
    """Mutable codegen state for one @Thread function."""

    fn: Callable
    this_file: Path
    owner: ThreadableUnit | None = None
    symbols: dict[str, PyType] = field(default_factory=dict)
    sig_includes: list[str] = field(default_factory=list)
    body_includes: list[str] = field(default_factory=list)
    seen_sig: set[str] = field(default_factory=set)
    seen_body: set[str] = field(default_factory=set)

    @property
    def func_name(self) -> str:
        return self.fn.__name__
