"""Integration tests for compile / build / thread public API."""

from pathlib import Path

import pytest

from cthreads.CONFIG import BINARY_PATH, KERNELS, REGISTRY, STORE
from cthreads.compile import compile as ct_compile


def test_compile_empty_registry_errors():
    with pytest.raises(RuntimeError, match="Nothing registered"):
        ct_compile(force=True)


def test_compile_generates_cpp_for_registered_units(tmp_module):
    mod = tmp_module(
        """
        from cthreads import Threadable, Thread

        @Threadable
        class Particle:
            x: float
            y: float
            velocity: float

            @Thread
            def step(self, dt: float) -> None:
                self.x += self.velocity * dt

        @Thread
        def move(p: Particle, dt: float) -> None:
            p.x += p.velocity * dt

        @Thread
        def is_fast(p: Particle, limit: float) -> bool:
            return p.velocity > limit
        """,
        name="ct_integ_compile",
    )

    info = ct_compile(force=True)
    root = Path(mod.__file__).parent
    assert info["root"] == root
    assert "Particle" in info["rewritten"] or (root / "__Threadable__" / "Particle.hpp").is_file()
    assert (root / "__Thread__" / "move.hpp").is_file()
    assert (root / "__Thread__" / "is_fast.cpp").is_file()
    assert "move" in STORE and "Particle" in STORE
    assert "move" in KERNELS and "Particle_step" in KERNELS
    # REGISTRY retained for re-compile
    assert "Particle" in REGISTRY.threadables
    assert (root / ".cthreads_cache.json").is_file()

    # idempotent: second compile should rewrite nothing if unchanged
    info2 = ct_compile(force=False)
    assert info2["rewritten"] == []


def test_compile_force_still_succeeds(tmp_module):
    mod = tmp_module(
        """
        from cthreads import Thread

        @Thread
        def add(a: int, b: int) -> int:
            return a + b
        """,
        name="ct_integ_force",
    )
    ct_compile(force=True)
    info = ct_compile(force=True)
    # force skips hash short-circuit; identical content may not rewrite files
    assert info["root"] == Path(mod.__file__).parent
    assert "add" in STORE
    assert "add" in KERNELS


@pytest.mark.integration
def test_prepare_build_and_thread_job(tmp_module):
    """Needs a C++ compiler + built cthreads._ext."""
    try:
        import cthreads._ext  # noqa: F401
    except ImportError as e:
        pytest.skip(f"cthreads._ext unavailable: {e}")
    from cthreads import Job, prepare, thread, unload_kernels

    mod = tmp_module(
        """
        from cthreads import Thread

        @Thread
        def add(a: int, b: int) -> int:
            return a + b
        """,
        name="ct_integ_run",
    )

    try:
        binary = prepare(force=True)
        assert Path(binary).is_file()
        job = thread(mod.add, 2, 3, force=False)
        assert isinstance(job, Job)
        job.start()
        job.join()
        assert job.done()
        assert job.result() == 5
    except RuntimeError as e:
        # No compiler / app-control DLL lock — skip rather than fail CI without toolchain
        if "compiler" in str(e).lower() or "STORE" in str(e):
            pytest.skip(str(e))
        raise
    finally:
        try:
            unload_kernels()
        except Exception:
            pass
