"""Smoke tests for package public surface."""

import cthreads


def test_public_exports():
    for name in (
        "Threadable",
        "Thread",
        "compile",
        "build",
        "prepare",
        "thread",
        "spawn",
        "Job",
        "sync",
        "load_kernels",
        "unload_kernels",
        "host_os",
        "VERSION",
        "ThreadPool",
    ):
        assert hasattr(cthreads, name)


def test_binary_path_getattr_live():
    # may be None before build
    _ = cthreads.BINARY_PATH
    assert cthreads.VERSION
