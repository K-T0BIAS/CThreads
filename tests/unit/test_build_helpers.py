"""Unit tests for build helpers (no full link required)."""

import sys

import pytest

import cthreads  # noqa: F401 — ensure package (and build submodule) loaded
from cthreads.CONFIG import STORE

build_mod = sys.modules["cthreads.build"]


def test_project_root_from_store_empty():
    STORE.clear()
    with pytest.raises(RuntimeError, match="STORE is empty"):
        build_mod._project_root_from_store()


def test_project_root_from_store_paths(tmp_path):
    thread = tmp_path / "__Thread__"
    thread.mkdir()
    hpp = thread / "add.hpp"
    hpp.write_text("x", encoding="utf-8")
    STORE["add"] = str(hpp)
    assert build_mod._project_root_from_store() == tmp_path.resolve()


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
    STORE["add"] = str(hpp)
    sources, includes = build_mod._collect_sources_and_includes()
    assert any(p.name == "sync_bridge.cpp" for p in sources)
    assert any(p.name == "headers" or (p / "sync").is_dir() for p in includes) or any(
        "headers" in str(p) for p in includes
    )
