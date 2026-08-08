"""
Shared state passed into every AST translator unit.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class TranslateContext:
    """Mutable codegen state for one @Thread function."""

    fn: Any
    func_name: str
    hints: dict[str, Any]
    # If set, this is a method on that Threadable (C++ member function).
    owner_name: Optional[str] = None
    # local / param name -> PyType
    symbols: dict[str, Any] = field(default_factory=dict)
    sig_includes: list[str] = field(default_factory=list)
    body_includes: list[str] = field(default_factory=list)
    seen_sig: set[str] = field(default_factory=set)
    seen_body: set[str] = field(default_factory=set)


StmtTranslator = Callable[[Any, "TranslateContext"], list[str]]
ExprTranslator = Callable[[Any, "TranslateContext"], str]
