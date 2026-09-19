"""
Integration: wheel must ship kernel native headers; packaged layout must compile.

These catch the PyPI/Colab regression where `@Thread` failed with
`shared_host.hpp: No such file or directory` after `pip install cthreads`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

build_mod = __import__("cthreads.build", fromlist=["*"])
from cthreads import Thread, thread
from helpers import skip_if_kernel_runtime_error

REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_HEADERS = REPO_ROOT / "src" / "cthreads" / "cpp" / "headers"
SYNC_BRIDGE = REPO_ROOT / "src" / "cthreads" / "cpp" / "runtime" / "sync_bridge.cpp"


@pytest.mark.integration
def test_installed_package_ships_native_headers_or_monorepo_fallback():
    """
    Wheel installs must expose `cthreads/_native/headers``.
    Editable checkouts may use monorepo ``cpp/headers`` instead.
    """
    packaged = build_mod._packaged_native_root()
    mono = build_mod._monorepo_cpp_root()
    assert packaged is not None or mono is not None, (
        "neither wheel _native/ nor monorepo cpp/headers found — "
        "kernel builds cannot succeed"
    )
    headers = build_mod.runtime_headers_dir()
    assert (headers / "shared_host.hpp").is_file()


@pytest.mark.integration
def test_non_editable_install_requires_packaged_native_headers():
    """
    cibuildwheel / ``pip install`` layout: monorepo ``cpp/`` is not next to
    ``build.py``, so ``_native/headers/shared_host.hpp`` must be in the wheel.
    """
    if build_mod._monorepo_cpp_root() is not None:
        pytest.skip("editable / source tree — monorepo headers are enough")
    packaged = build_mod._packaged_native_root()
    assert packaged is not None, (
        "wheel install missing cthreads/_native/headers/shared_host.hpp — "
        "this is the PyPI/Colab shared_host.hpp regression"
    )
    assert (packaged / "runtime" / "sync_bridge.cpp").is_file()


@pytest.mark.integration
def test_built_wheel_contains_native_headers(tmp_path: Path):
    """
    ``python -m build --wheel`` must install ``cthreads/_native/headers/...``.

    Soft-skips when the toolchain/`build` frontend is unavailable. Set
    ``CTHREADS_REQUIRE_WHEEL_HEADERS=1`` to fail hard (manual / release gate).
    cibuildwheel coverage is ``test_non_editable_install_requires_packaged_native_headers``.
    """
    require = bool(os.environ.get("CTHREADS_REQUIRE_WHEEL_HEADERS"))
    try:
        import build as _build_frontend  # noqa: F401
    except ImportError:
        if require:
            pytest.fail("python `build` package required to verify wheel header install")
        pytest.skip("python `build` package not installed")

    out = tmp_path / "dist"
    out.mkdir()
    proc = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "")[-3000:]
        if require:
            pytest.fail(f"wheel build failed (headers packaging unverified):\n{detail}")
        pytest.skip(
            "wheel build failed in this environment "
            f"(need scikit-build toolchain):\n{detail}"
        )
    wheels = list(out.glob("*.whl"))
    assert wheels, "no wheel produced"
    with zipfile.ZipFile(wheels[0]) as zf:
        names = zf.namelist()
    assert any(
        n.replace("\\", "/").endswith("cthreads/_native/headers/shared_host.hpp")
        for n in names
    ), f"shared_host.hpp missing from wheel; sample entries: {names[:20]}"
    assert any(
        n.replace("\\", "/").endswith("cthreads/_native/runtime/sync_bridge.cpp")
        for n in names
    ), "sync_bridge.cpp missing from wheel"


@pytest.mark.integration
def test_thread_compiles_when_only_packaged_native_layout_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Hide the monorepo ``cpp/`` path and expose only a wheel-like ``_native/``
    tree next to a fake ``build.py``. Kernel link must still find headers.
    """
    if not CPP_HEADERS.is_dir() or not SYNC_BRIDGE.is_file():
        pytest.skip("monorepo cpp assets missing")

    pkg = tmp_path / "cthreads"
    native = pkg / "_native"
    shutil.copytree(CPP_HEADERS, native / "headers")
    (native / "runtime").mkdir(parents=True)
    shutil.copy2(SYNC_BRIDGE, native / "runtime" / "sync_bridge.cpp")
    fake_build_py = pkg / "build.py"
    fake_build_py.write_text("# fake package build module path\n", encoding="utf-8")

    monkeypatch.setattr(build_mod, "__file__", str(fake_build_py))

    # Confirm monorepo fallback is not used for this process's locator.
    assert build_mod._packaged_native_root() is not None
    assert build_mod.runtime_headers_dir() == (native / "headers").resolve()

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    @Thread
    def add_one(n: int, out: list[int]) -> None:
        i: int = 0
        while i < n:
            out[i] = out[i] + 1
            i = i + 1

    out: list[int] = [0, 1, 2, 3]
    try:
        thread(add_one, len(out), out).join()
    except RuntimeError as exc:
        skip_if_kernel_runtime_error(exc)
        raise

    assert out == [1, 2, 3, 4]
