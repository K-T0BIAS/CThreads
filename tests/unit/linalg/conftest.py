"""Skip the linalg suite when `cthreads._ext.linalg` is missing."""

from __future__ import annotations

import pytest

try:
    from cthreads import linalg
except ImportError:
    pytest.skip("cthreads._ext not available", allow_module_level=True)

if linalg is None:
    pytest.skip("cthreads.linalg submodule missing (rebuild _ext)", allow_module_level=True)
