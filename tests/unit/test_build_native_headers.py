"""
Regression tests for kernel runtime header discovery (wheel vs editable).

PyPI 0.2.0 wheels shipped `_ext` but not `cpp/headers`. `build.py` then only
looked at a monorepo-relative path and silently skipped includes, so Colab /
`pip install cthreads` failed with missing `shared_host.hpp`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

build_mod = __import__("cthreads.build", fromlist=["*"])


def test_runtime_headers_dir_finds_shared_host_in_editable_checkout():
    headers = build_mod.runtime_headers_dir()
    assert (headers / "shared_host.hpp").is_file()
    assert (headers / "sync" / "syncState.hpp").is_file()


def test_sync_bridge_source_present_in_editable_checkout():
    bridge = build_mod.sync_bridge_source()
    assert bridge is not None
    assert bridge.is_file()
    assert bridge.name == "sync_bridge.cpp"


def test_packaged_native_layout_is_preferred(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Simulate a wheel install: headers live under cthreads/_native/, not cpp/.
    """
    pkg = tmp_path / "cthreads"
    native_headers = pkg / "_native" / "headers"
    native_headers.mkdir(parents=True)
    (native_headers / "shared_host.hpp").write_text("#pragma once\n", encoding="utf-8")
    sync = native_headers / "sync"
    sync.mkdir()
    (sync / "syncState.hpp").write_text("#pragma once\n", encoding="utf-8")
    runtime = pkg / "_native" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "sync_bridge.cpp").write_text(
        '#include "../headers/sync/syncState.hpp"\n',
        encoding="utf-8",
    )
    fake_build_py = pkg / "build.py"
    fake_build_py.write_text("# fake\n", encoding="utf-8")

    monkeypatch.setattr(build_mod, "__file__", str(fake_build_py))

    headers = build_mod.runtime_headers_dir()
    assert headers == native_headers.resolve()
    bridge = build_mod.sync_bridge_source()
    assert bridge is not None
    assert bridge == (runtime / "sync_bridge.cpp").resolve()


def test_missing_headers_raise_instead_of_silent_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Old bug: missing dir was ignored and compile failed later with a cryptic
    missing include. Locator must fail fast with an actionable message.
    """
    pkg = tmp_path / "cthreads"
    pkg.mkdir()
    fake_build_py = pkg / "build.py"
    fake_build_py.write_text("# fake\n", encoding="utf-8")
    monkeypatch.setattr(build_mod, "__file__", str(fake_build_py))

    with pytest.raises(RuntimeError, match="runtime headers are missing"):
        build_mod.runtime_headers_dir()


def test_collect_sources_adds_headers_include_with_shared_host():
    """
    Collecting includes must inject the runtime headers dir (kernel meta always
    pulls shared_host.hpp), even when no ThreadUnits are registered yet.
    """
    _sources, include_dirs = build_mod._collect_sources_and_includes()
    assert any((Path(d) / "shared_host.hpp").is_file() for d in include_dirs)
    assert build_mod.sync_bridge_source() is not None


def test_cmake_installs_native_headers_into_wheel_layout():
    """
    Guard the CMake install rule that puts headers into cthreads/_native/.
    Deleting this silently reintroduces the PyPI/Colab shared_host.hpp bug.
    """
    cmake = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "cthreads"
        / "cpp"
        / "CMakeLists.txt"
    )
    text = cmake.read_text(encoding="utf-8")
    assert "cthreads/_native/headers" in text
    assert "cthreads/_native/runtime" in text
    assert "sync_bridge.cpp" in text
