"""
Runtime tests for `cthreads.math` pybind bindings.

Skipped when `_ext` cannot load (e.g. Windows policy blocking a fresh DLL).

Usage pattern (same as sync)::

    from cthreads import math
    math.abs(-1.0)
"""

from __future__ import annotations

import pytest

try:
    from cthreads import math as cmath
except ImportError:
    pytest.skip("cthreads._ext not available", allow_module_level=True)

if cmath is None:
    pytest.skip("cthreads.math submodule missing (rebuild _ext)", allow_module_level=True)


REQUIRED = ("abs", "min", "max", "clamp", "random", "uniform", "randint", "seed")


@pytest.mark.parametrize("name", REQUIRED)
def test_binding_exists_and_marked(name):
    assert hasattr(cmath, name)
    assert callable(getattr(cmath, name))
    # Module is marked; pybind bound functions cannot carry arbitrary attrs.
    assert getattr(cmath, "__cthreads_internal__", False) is True


def test_cthreads_import_math_like_sync():
    from cthreads import math

    assert math.abs(-4.0) == pytest.approx(4.0)
    assert math.min(2.0, 5.0) == 2.0
    assert math.max(2.0, 5.0) == 5.0
    assert math.clamp(5.0, 0.0, 1.0) == 1.0
    assert math.clamp(-1.0, 0.0, 1.0) == 0.0
    math.seed(42)
    assert 0.0 <= math.uniform(0.0, 1.0) <= 1.0
    assert 0.0 <= math.random() < 1.0
    assert 1 <= math.randint(1, 3) <= 3


def test_abs_min_max_clamp_values():
    assert cmath.abs(-3.5) == pytest.approx(3.5)
    assert cmath.min(1.0, 2.0) == 1.0
    assert cmath.max(1.0, 2.0) == 2.0
    assert cmath.clamp(0.5, 0.0, 1.0) == 0.5


def test_random_seed_reproducible_enough():
    cmath.seed(123)
    a = [cmath.random() for _ in range(5)]
    cmath.seed(123)
    b = [cmath.random() for _ in range(5)]
    assert a == b
