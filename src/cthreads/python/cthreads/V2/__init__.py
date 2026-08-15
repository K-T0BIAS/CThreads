"""
Compat shim: V2 was promoted to the ``cthreads`` package root.

Prefer ``import cthreads`` / ``from cthreads import Thread, prepare, ...``.
"""

from __future__ import annotations

from typing import Any

# Re-export the public root surface without circular package init issues:
# import submodule attributes from the parent package after it is loaded.
def __getattr__(name: str) -> Any:
    import cthreads as _root

    return getattr(_root, name)


def __dir__() -> list[str]:
    import cthreads as _root

    return sorted(set(dir(_root)) | {"__getattr__", "__dir__"})
