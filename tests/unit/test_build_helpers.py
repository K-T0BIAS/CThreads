"""Unit tests for build helpers (no full link required)."""

from pathlib import Path

import pytest

import cthreads  # noqa: F401
from cthreads.compiler.orchestrator.units import Handle, ThreadUnit
from cthreads.frontend.Registry import REGISTRY

build_mod = __import__("cthreads.build", fromlist=["*"])


def test_project_root_from_units_empty():
    REGISTRY.threadable_units.clear()
    REGISTRY.thread_units.clear()
    with pytest.raises(RuntimeError, match="No compiled units"):
        build_mod._project_root_from_units()


def test_project_root_from_units_paths(tmp_path):
    thread = tmp_path / "__Thread__"
    thread.mkdir()
    hpp = thread / "add.hpp"
    cpp = thread / "add.cpp"
    hpp.write_text("x", encoding="utf-8")
    cpp.write_text("x", encoding="utf-8")

    def add(a: int, b: int) -> int:
        return a + b

    add.__threaded = True
    unit = ThreadUnit(
        handle=Handle(name="add", path=str(tmp_path / "m.py"), target=add),
        owner=None,
        params=[],
        return_type=None,
        hpp_path=hpp,
        cpp_path=cpp,
    )
    REGISTRY.thread_units["add"] = unit
    assert build_mod._project_root_from_units() == tmp_path.resolve()


def test_detect_compiler_returns_flavor():
    path, flavor = build_mod._detect_compiler()
    assert path
    assert flavor in ("msvc", "gnu")


def test_collect_sources_includes_sync_bridge(tmp_path):
    thread = tmp_path / "__Thread__"
    thread.mkdir()
    hpp = thread / "add.hpp"
    cpp = thread / "add.cpp"
    hpp.write_text("x", encoding="utf-8")
    cpp.write_text("x", encoding="utf-8")

    def add(a: int, b: int) -> int:
        return a + b

    add.__threaded = True
    REGISTRY.thread_units["add"] = ThreadUnit(
        handle=Handle(name="add", path=str(tmp_path / "m.py"), target=add),
        owner=None,
        params=[],
        return_type=None,
        hpp_path=hpp,
        cpp_path=cpp,
    )
    sources, includes = build_mod._collect_sources_and_includes()
    assert any(p.name == "sync_bridge.cpp" for p in sources)
    assert any(p.name == "headers" or (p / "sync").is_dir() for p in includes) or any(
        "headers" in str(p) for p in includes
    )
