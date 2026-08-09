"""
Python ``math`` → C++ ``<cmath>`` lowering table.

``MATHOPS`` is filtered with ``hasattr(math, name)`` so entries that only
exist on newer CPython (e.g. ``math.fma`` on 3.13+) are omitted on older
interpreters. Arity is the std::-compatible positional count (exact).

Notes:
  - ``math.log(x, base)`` is rejected at the call site (arity 1 → ``std::log``).
  - ``math.hypot(*coords)`` is limited to 2 args (``std::hypot``).
  - ``math.gamma`` → ``std::tgamma``.
  - Types are not stored; numeric exprs lower to C++ overloads.
"""

from __future__ import annotations

import math
from typing import NamedTuple

CMATH_INCLUDE = "#include <cmath>\n"
NUMBERS_INCLUDE = "#include <numbers>\n"


class MathOp(NamedTuple):
    cpp_func: str
    arity: int


# Full support set; filtered below for the running interpreter.
_ALL_MATHOPS: dict[str, MathOp] = {
    "sqrt": MathOp("std::sqrt", 1),
    "cbrt": MathOp("std::cbrt", 1),
    "pow": MathOp("std::pow", 2),
    "hypot": MathOp("std::hypot", 2),
    "fabs": MathOp("std::fabs", 1),
    "floor": MathOp("std::floor", 1),
    "ceil": MathOp("std::ceil", 1),
    "trunc": MathOp("std::trunc", 1),
    "fmod": MathOp("std::fmod", 2),
    "remainder": MathOp("std::remainder", 2),
    "copysign": MathOp("std::copysign", 2),
    "fma": MathOp("std::fma", 3),  # math.fma: 3.13+
    "exp": MathOp("std::exp", 1),
    "exp2": MathOp("std::exp2", 1),
    "expm1": MathOp("std::expm1", 1),
    "log": MathOp("std::log", 1),
    "log2": MathOp("std::log2", 1),
    "log10": MathOp("std::log10", 1),
    "log1p": MathOp("std::log1p", 1),
    "sin": MathOp("std::sin", 1),
    "cos": MathOp("std::cos", 1),
    "tan": MathOp("std::tan", 1),
    "asin": MathOp("std::asin", 1),
    "acos": MathOp("std::acos", 1),
    "atan": MathOp("std::atan", 1),
    "atan2": MathOp("std::atan2", 2),
    "sinh": MathOp("std::sinh", 1),
    "cosh": MathOp("std::cosh", 1),
    "tanh": MathOp("std::tanh", 1),
    "asinh": MathOp("std::asinh", 1),
    "acosh": MathOp("std::acosh", 1),
    "atanh": MathOp("std::atanh", 1),
    "erf": MathOp("std::erf", 1),
    "erfc": MathOp("std::erfc", 1),
    "gamma": MathOp("std::tgamma", 1),
    "lgamma": MathOp("std::lgamma", 1),
    "ldexp": MathOp("std::ldexp", 2),
    "isfinite": MathOp("std::isfinite", 1),
    "isinf": MathOp("std::isinf", 1),
    "isnan": MathOp("std::isnan", 1),
}

MATHOPS: dict[str, MathOp] = {
    name: op for name, op in _ALL_MATHOPS.items() if hasattr(math, name)
}

# Attribute constants (not calls). Value is a C++ expression.
_ALL_MATHCONSTS: dict[str, str] = {
    "pi": "std::numbers::pi",
    "e": "std::numbers::e",
    "tau": "std::numbers::pi * 2.0",
}

MATHCONSTS: dict[str, str] = {
    name: expr for name, expr in _ALL_MATHCONSTS.items() if hasattr(math, name)
}
