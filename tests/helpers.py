"""Shared test helpers (importable; conftest re-exports fixtures only)."""

from __future__ import annotations

import ast
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

from cthreads.compiler.orchestrator.units import Handle, ThreadableUnit
from cthreads.compiler.translation.context import TranslationContext
from cthreads.frontend.Registry import REGISTRY
from cthreads.types import hint_to_pytype


def make_ctx(
    func_name: str = "f",
    *,
    symbols: dict[str, Any] | None = None,
    owner_name: str | None = None,
    globals_extra: dict[str, Any] | None = None,
    this_file: Path | str = "x.hpp",
) -> TranslationContext:
    class _DummyFn:
        __name__ = func_name
        __globals__ = {
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "str": str,
        }

    if globals_extra:
        _DummyFn.__globals__ = {**_DummyFn.__globals__, **globals_extra}

    owner = None
    if owner_name is not None:
        owner = REGISTRY.threadable_units.get(owner_name)
        if owner is None:
            cls = REGISTRY.threadables.get(owner_name) or type(
                owner_name, (), {"__threadable": True, "__annotations__": {}}
            )
            owner = ThreadableUnit(
                handle=Handle(name=owner_name, path="dummy.py", target=cls),
                fields={},
                hpp_path=Path("__Threadable__") / f"{owner_name}.hpp",
                cpp_path=Path("__Threadable__") / f"{owner_name}.cpp",
            )
            REGISTRY.threadable_units[owner_name] = owner

    return TranslationContext(
        fn=_DummyFn(),
        this_file=Path(this_file),
        owner=owner,
        symbols=dict(symbols or {}),
    )


def parse_expr(src: str) -> ast.expr:
    return ast.parse(src, mode="eval").body


def parse_stmt(src: str) -> ast.stmt:
    body = ast.parse(textwrap.dedent(src)).body
    assert len(body) == 1
    return body[0]


def make_threadable_type(name: str = "Particle") -> type:
    return type(
        name,
        (),
        {
            "__threadable": True,
            "__annotations__": {"x": float, "y": float, "velocity": float},
        },
    )


@contextmanager
def registered_threadable(cls: type) -> Iterator[type]:
    """Register class + ThreadableUnit so Typeof / include_for can resolve fields."""
    REGISTRY.register_threadable(cls)
    fields = {
        name: hint_to_pytype(hint)
        for name, hint in getattr(cls, "__annotations__", {}).items()
    }
    unit = ThreadableUnit(
        handle=Handle(name=cls.__name__, path="dummy.py", target=cls),
        fields=fields,
        hpp_path=Path("__Threadable__") / f"{cls.__name__}.hpp",
        cpp_path=Path("__Threadable__") / f"{cls.__name__}.cpp",
    )
    REGISTRY.threadable_units[cls.__name__] = unit
    try:
        yield cls
    finally:
        REGISTRY.threadables.pop(cls.__name__, None)
        REGISTRY.threadable_units.pop(cls.__name__, None)


def is_kernel_load_failure(exc: BaseException) -> bool:
    msg = str(exc)
    return "LoadLibrary" in msg or "dlopen" in msg


def skip_if_kernel_runtime_error(exc: RuntimeError) -> None:
    """
    Skip integration tests when the native kernel cannot be built or loaded.

    On Windows 11 with Smart App Control, unsigned ``cthreads_kernels.dll`` can
    fail LoadLibrary (GetLastError=4551) even when the file exists locally.
    CI/Linux is unaffected; skip + warn instead of failing the suite.
    """
    import warnings

    msg = str(exc)
    if is_kernel_load_failure(exc):
        warnings.warn(
            f"kernel DLL load blocked — skipping test ({msg})",
            UserWarning,
            stacklevel=3,
        )
        pytest.skip(f"kernel DLL load failed: {msg}")
    if "compiler" in msg.lower() or "Nothing registered" in msg:
        pytest.skip(msg)
    raise exc


def load_kernels_or_skip(path: str) -> None:
    from cthreads import load_kernels

    try:
        load_kernels(path)
    except RuntimeError as e:
        skip_if_kernel_runtime_error(e)


def prepare_and_load_or_skip(*, force: bool = True) -> Path:
    from cthreads import prepare, unload_kernels

    unload_kernels()
    try:
        binary = prepare(force=force)
    except RuntimeError as e:
        skip_if_kernel_runtime_error(e)
    load_kernels_or_skip(str(binary))
    return Path(binary)
