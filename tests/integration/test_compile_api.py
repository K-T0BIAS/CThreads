"""Integration tests for compile / build / thread public API."""

from pathlib import Path

import pytest

from cthreads import compile as ct_compile
from cthreads.frontend.Registry import REGISTRY
from cthreads.kernel_meta import KERNELS


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
    assert "Particle" in REGISTRY.threadable_units
    assert "move" in REGISTRY.thread_units
    assert "move" in KERNELS and "Particle_step" in KERNELS
    assert "Particle" in REGISTRY.threadables
    assert (root / ".cthreads_cache.json").is_file()

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
    assert info["root"] == Path(mod.__file__).parent
    assert "add" in REGISTRY.thread_units
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
        if "compiler" in str(e).lower() or "Nothing registered" in str(e):
            pytest.skip(str(e))
        raise
    finally:
        try:
            unload_kernels()
        except Exception:
            pass


@pytest.mark.integration
def test_thread_job_await(tmp_module):
    """Preferred async API: thread() then await job."""
    try:
        import cthreads._ext  # noqa: F401
    except ImportError as e:
        pytest.skip(f"cthreads._ext unavailable: {e}")

    import asyncio

    from cthreads import Job, thread, unload_kernels

    mod = tmp_module(
        """
        from cthreads import Thread

        @Thread
        def add(a: int, b: int) -> int:
            return a + b
        """,
        name="ct_integ_await",
    )

    async def main():
        job = thread(mod.add, 10, 5, force=False)
        assert isinstance(job, Job)
        return await job

    try:
        assert asyncio.run(main()) == 15
    except RuntimeError as e:
        if "compiler" in str(e).lower() or "Nothing registered" in str(e):
            pytest.skip(str(e))
        raise
    finally:
        try:
            unload_kernels()
        except Exception:
            pass


@pytest.mark.integration
def test_mutable_list_dict_writeback_by_ref(tmp_module):
    """List/dict params must bind pack by ref so mutations survive writeback."""
    try:
        import cthreads._ext  # noqa: F401
    except ImportError as e:
        pytest.skip(f"cthreads._ext unavailable: {e}")

    from cthreads import prepare, thread, unload_kernels
    from cthreads.kernel_meta import KERNELS

    mod = tmp_module(
        """
        from cthreads import Thread

        @Thread
        def bump_list(xs: list[int]) -> None:
            xs.append(99)

        @Thread
        def bump_dict(d: dict[str, int]) -> None:
            d.clear()
        """,
        name="ct_integ_mut_ref",
    )

    try:
        prepare(force=True)
        root = Path(mod.__file__).parent
        list_cpp = (root / "__Thread__" / "bump_list.cpp").read_text(encoding="utf-8")
        dict_cpp = (root / "__Thread__" / "bump_dict.cpp").read_text(encoding="utf-8")
        assert "std::vector<int>& xs" in list_cpp
        assert "std::unordered_map<std::string, int>& d" in dict_cpp
        assert KERNELS["bump_list"].params[0].pass_as == "ref"
        assert KERNELS["bump_dict"].params[0].pass_as == "ref"

        xs = [1, 2]
        job = thread(mod.bump_list, xs, force=False)
        job.start()
        job.join()
        assert xs == [1, 2, 99]

        d = {"a": 1}
        job = thread(mod.bump_dict, d, force=False)
        job.start()
        job.join()
        assert d == {}
    except RuntimeError as e:
        if "compiler" in str(e).lower() or "Nothing registered" in str(e):
            pytest.skip(str(e))
        raise
    finally:
        try:
            unload_kernels()
        except Exception:
            pass


@pytest.mark.integration
def test_kernel_sync_state_mid_run_writeback(tmp_module):
    """In-kernel __sync_state() must hit _ext TLS (not a per-DLL empty slot)."""
    try:
        import cthreads._ext as ext
    except ImportError as e:
        pytest.skip(f"cthreads._ext unavailable: {e}")

    import time

    from cthreads import prepare, thread, unload_kernels

    mod = tmp_module(
        """
        from cthreads import Thread, __sync_state

        @Thread
        def pulse(xs: list[int], sink: list[int]) -> None:
            xs.append(1)
            __sync_state()
            # Side-effecting busy work so /O2 cannot delete the delay window.
            i: int = 0
            while i < 8000000:
                sink.append(0)
                sink.pop()
                i = i + 1
            xs.append(2)
        """,
        name="ct_integ_sync_tls",
    )

    try:
        unload_kernels()
        ext._debug_reset_ext_sync_invocations()
        prepare(force=True)
        xs: list[int] = []
        sink: list[int] = []
        job = None
        last_err: Exception | None = None
        for _ in range(5):
            try:
                job = thread(mod.pulse, xs, sink, force=False)
                break
            except RuntimeError as e:
                last_err = e
                if "LoadLibrary" not in str(e) and "dlopen" not in str(e):
                    raise
                unload_kernels()
                time.sleep(0.15)
        if job is None:
            raise last_err  # type: ignore[misc]
        job.start()
        saw_mid = False
        deadline = time.perf_counter() + 60.0
        while time.perf_counter() < deadline and not job.done():
            if xs == [1]:
                saw_mid = True
                break
            time.sleep(0.0005)
        job.join()
        assert xs == [1, 2]
        assert ext._debug_ext_sync_invocations() >= 1
        assert saw_mid, "host never observed mid-run writeback from __sync_state()"
    except RuntimeError as e:
        if "compiler" in str(e).lower() or "Nothing registered" in str(e):
            pytest.skip(str(e))
        raise
    finally:
        try:
            unload_kernels()
        except Exception:
            pass
