"""
Pytest configuration and shared fixtures for cthreads tests.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON_PKG_ROOT = ROOT / "src" / "cthreads" / "python"
if str(PYTHON_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PKG_ROOT))

from cthreads.CONFIG import KERNELS, REGISTRY, STORE  # noqa: E402

# Re-export helpers so older `from conftest import ...` still works in this package.
from helpers import make_ctx, make_threadable_type, parse_expr, parse_stmt  # noqa: E402,F401


@pytest.fixture(autouse=True)
def _reset_cthreads_state():
    """Isolate global REGISTRY / STORE / KERNELS between tests."""
    REGISTRY.clear()
    STORE.clear()
    KERNELS.clear()
    yield
    REGISTRY.clear()
    STORE.clear()
    KERNELS.clear()


@pytest.fixture
def tmp_module(tmp_path: Path):
    """
    Write a Python module under tmp_path and import it so
    inspect.getfile / codegen output land in the temp tree.
    """

    def _load(source: str, name: str = "cthreads_test_mod"):
        path = tmp_path / f"{name}.py"
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    return _load
