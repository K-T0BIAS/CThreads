"""prepare / thread load policy - no auto-unload under concurrency."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

from cthreads.job import Job

# cthreads.prepare attribute is the function (shadows the submodule).
prepare_mod = importlib.import_module("cthreads.prepare")
ext_api = importlib.import_module("cthreads._ext_api")


def _install_fake_ext(monkeypatch, *, path: str | None):
    fake = types.ModuleType("cthreads._ext")
    state = {"path": path, "prepare": 0, "load": 0, "spawn": 0, "unload": 0}

    def kernel_path():
        return state["path"]

    def load_kernels(p):
        state["load"] += 1
        state["path"] = p

    def unload_kernels():
        state["unload"] += 1
        state["path"] = None

    def thread(fn, *args, **kwargs):
        state["spawn"] += 1
        return object()

    fake.kernel_path = kernel_path
    fake.load_kernels = load_kernels
    fake.unload_kernels = unload_kernels
    fake.thread = thread

    monkeypatch.setitem(sys.modules, "cthreads._ext", fake)
    # _ext_api caches the import result; force it to use our fake.
    monkeypatch.setattr(ext_api, "_ext", fake)
    return state


def test_thread_uses_loaded_kernels_without_prepare(monkeypatch):
    state = _install_fake_ext(monkeypatch, path="C:/fake/kernels.dll")

    def boom_prepare(force=False):
        state["prepare"] += 1
        raise AssertionError("prepare should not run when kernels are loaded")

    monkeypatch.setattr(prepare_mod, "prepare", boom_prepare)

    job = prepare_mod.thread(lambda: None, force=False)
    assert isinstance(job, Job)
    assert state["prepare"] == 0
    assert state["load"] == 0
    assert state["spawn"] == 1


def test_thread_prepare_and_load_when_unloaded(monkeypatch):
    state = _install_fake_ext(monkeypatch, path=None)

    def fake_prepare(force=False):
        state["prepare"] += 1
        return Path("C:/fake/kernels.dll")

    monkeypatch.setattr(prepare_mod, "prepare", fake_prepare)

    job = prepare_mod.thread(lambda: None)
    assert isinstance(job, Job)
    assert state["prepare"] == 1
    assert state["load"] == 1
    assert state["spawn"] == 1
    assert state["path"] == str(Path("C:/fake/kernels.dll"))


def test_thread_force_while_loaded_raises(monkeypatch):
    _install_fake_ext(monkeypatch, path="C:/fake/kernels.dll")
    with pytest.raises(RuntimeError, match="still loaded"):
        prepare_mod.thread(lambda: None, force=True)


def test_prepare_does_not_unload(monkeypatch):
    state = _install_fake_ext(monkeypatch, path="C:/fake/kernels.dll")
    monkeypatch.setattr(prepare_mod, "compile", lambda force=False: {"root": "."})
    monkeypatch.setattr(
        prepare_mod,
        "build",
        lambda project_root=None, force=False: Path("."),
    )

    prepare_mod.prepare(force=False)
    assert state["unload"] == 0
