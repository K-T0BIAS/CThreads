"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Python `math` / `cthreads.math` -> C++ lowering tables.

`MATHOPS`: stdlib `math.*` -> `std::*` + `<cmath>` (hasattr-filtered).
`CTHREADS_MATHOPS`: `cthreads.math.*` (module `__cthreads_internal__`) ->
`cthreads::math::*` + `math/*.hpp`.
"""

import math
from typing import NamedTuple

# defines the include for the cmath library
CMATH_INCLUDE = "#include <cmath>\n"
# defines the include for the numbers library
NUMBERS_INCLUDE = "#include <numbers>\n"


class MathOp(NamedTuple):
    cpp_func: str
    arity: int
    # None -> use CMATH_INCLUDE; else e.g. 'math/abs.hpp'
    cpp_include: str | None = None


# Full stdlib support set; filtered below for the running interpreter.
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

# filter the math ops based on the math module in the users python interpreter
MATHOPS: dict[str, MathOp] = {
    name: op for name, op in _ALL_MATHOPS.items() if hasattr(math, name)
}

# Attribute constants (not calls). Value is a C++ expression.
_ALL_MATHCONSTS: dict[str, str] = {
    "pi": "std::numbers::pi",
    "e": "std::numbers::e",
    "tau": "std::numbers::pi * 2.0",
}

# filter the math constants based on the math module in the users python interpreter
MATHCONSTS: dict[str, str] = {
    name: expr for name, expr in _ALL_MATHCONSTS.items() if hasattr(math, name)
}

# from cthreads import math; math.abs / clamp / ...
# math implementations comming from the cthreads.math module 
# (defined in the c++ math library and bound to the cthreads namespace via pybind11)
CTHREADS_MATHOPS: dict[str, MathOp] = {
    "abs": MathOp("cthreads::math::abs", 1, "math/abs.hpp"),
    "min": MathOp("cthreads::math::min", 2, "math/clamps.hpp"),
    "max": MathOp("cthreads::math::max", 2, "math/clamps.hpp"),
    "clamp": MathOp("cthreads::math::clamp", 3, "math/clamps.hpp"),
    "random": MathOp("cthreads::math::random", 0, "math/random.hpp"),
    "uniform": MathOp("cthreads::math::uniform", 2, "math/random.hpp"),
    "randint": MathOp("cthreads::math::randint", 2, "math/random.hpp"),
    "seed": MathOp("cthreads::math::seed", 1, "math/random.hpp"),
}
