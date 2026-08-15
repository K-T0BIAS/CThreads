"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Shared state passed into every AST translator unit.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class TranslateContext:
    """Mutable codegen state for one @Thread function."""

    fn: Any # the function being translated
    func_name: str # the name of the function being translated
    hints: dict[str, Any] # the hints for the function being translated
    # If set, this is a method on that Threadable (C++ member function).
    owner_name: Optional[str] = None
    # local / param name -> PyType
    symbols: dict[str, Any] = field(default_factory=dict)
    sig_includes: list[str] = field(default_factory=list) # the includes for the function signature
    body_includes: list[str] = field(default_factory=list) # the includes for the function body
    seen_sig: set[str] = field(default_factory=set) # the symbols that have been seen in the function signature
    seen_body: set[str] = field(default_factory=set) # the symbols that have been seen in the function body

# specific types for the translators
StmtTranslator = Callable[[Any, "TranslateContext"], list[str]]
ExprTranslator = Callable[[Any, "TranslateContext"], str]
