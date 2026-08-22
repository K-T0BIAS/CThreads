"""Extensive unit tests for ``cthreads.ThreadPool`` / ``cthreads.pool``."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from cthreads.job import Job
from cthreads.pool.threadPool import ThreadPool as FacadeThreadPool
from helpers import prepare_and_load_or_skip, skip_if_kernel_runtime_error


# ---------------------------------------------------------------------------
# Package / facade (mocked native — no kernels)
# ---------------------------------------------------------------------------


def test_threadpool_exported_from_cthreads():
    import cthreads
    from cthreads import ThreadPool
    from cthreads.pool import ThreadPool as PoolThreadPool

    assert ThreadPool is FacadeThreadPool
    assert PoolThreadPool is FacadeThreadPool
    assert "ThreadPool" in cthreads.__all__


def test_facade_requires_native(monkeypatch):
    import cthreads.pool.threadPool as mod

    monkeypatch.setattr(mod, "_NativeThreadPool", None)
    with pytest.raises(RuntimeError, match="native extension"):
        mod.ThreadPool(2)


def test_facade_lifecycle_and_submit_marks_started(monkeypatch):
    import cthreads.pool.threadPool as mod

    class _RawJob:
        def start(self):
            raise AssertionError("pool Job.start should not need native start")

        def done(self):
            return True

        def join(self):
            return None

        def wait(self):
            return None

        def result(self):
            return 11

        def sync_state(self):
            return None

    class _RawPool:
        def __init__(self, capacity: int, queue_limit: int = -1):
            self.capacity = capacity
            self.queue_limit = queue_limit
            self.started = False
            self.stopped = False
            self.joined = False
            self.submitted: list[tuple[Any, tuple, dict]] = []

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def join(self):
            self.joined = True

        def is_running(self, thread_id: int | None = None):
            if thread_id is None:
                return [False] * self.capacity
            return False

        def submit(self, fn, *args, **kwargs):
            self.submitted.append((fn, args, dict(kwargs)))
            return _RawJob()

    monkeypatch.setattr(mod, "_NativeThreadPool", _RawPool)

    pool = mod.ThreadPool(4)
    assert pool.capacity == 4
    assert pool.queue_limit == -1
    assert "capacity=4" in repr(pool)
    assert pool.start() is pool
    assert pool._raw.started is True

    def fake_fn():
        return None

    job = pool.submit(fake_fn, 1, 2, x=3)
    assert isinstance(job, Job)
    assert job._started is True
    assert job.result() == 11
    assert pool._raw.submitted == [(fake_fn, (1, 2), {"x": 3})]

    assert pool.is_running() == [False, False, False, False]
    assert pool.is_running(0) is False
    pool.join()
    assert pool._raw.joined is True
    pool.stop()
    assert pool._raw.stopped is True


def test_facade_queue_limit_none_and_int(monkeypatch):
    import cthreads.pool.threadPool as mod

    seen: list[tuple[int, int]] = []

    class _RawPool:
        def __init__(self, capacity: int, queue_limit: int = -1):
            self.capacity = capacity
            self.queue_limit = queue_limit
            seen.append((capacity, queue_limit))

    monkeypatch.setattr(mod, "_NativeThreadPool", _RawPool)
    assert mod.ThreadPool(2).queue_limit == -1
    assert mod.ThreadPool(2, None).queue_limit == -1
    assert mod.ThreadPool(2, 8).queue_limit == 8
    assert seen == [(2, -1), (2, -1), (2, 8)]


def test_facade_group_emits_jobgroup(monkeypatch):
    import cthreads.pool.threadPool as mod
    from cthreads.pool.group import JobGroup

    class _RawJob:
        def __init__(self, value):
            self._value = value

        def start(self):
            return None

        def done(self):
            return True

        def join(self):
            return None

        def result(self):
            return self._value

    class _RawPool:
        def __init__(self, capacity: int, queue_limit: int = -1):
            self.capacity = capacity
            self.queue_limit = queue_limit
            self._n = 0

        def start(self):
            return None

        def submit(self, fn, *args, **kwargs):
            self._n += 1
            return _RawJob(args[0] if args else self._n)

    monkeypatch.setattr(mod, "_NativeThreadPool", _RawPool)
    pool = mod.ThreadPool(2).start()
    group = pool.group(lambda x: x, [10, (20,), [30]])
    assert isinstance(group, JobGroup)
    assert len(group) == 3
    assert group.results() == [10, 20, 30]


def test_facade_is_running_index(monkeypatch):
    import cthreads.pool.threadPool as mod

    class _RawPool:
        def __init__(self, capacity: int, queue_limit: int = -1):
            self.capacity = capacity
            self.queue_limit = queue_limit

        def is_running(self, thread_id: int | None = None):
            if thread_id is None:
                return [True, False]
            if thread_id == 0:
                return True
            if thread_id == 1:
                return False
            raise IndexError(thread_id)

    monkeypatch.setattr(mod, "_NativeThreadPool", _RawPool)
    pool = mod.ThreadPool(2)
    assert pool.is_running() == [True, False]
    assert pool.is_running(0) is True
    assert pool.is_running(1) is False


# ---------------------------------------------------------------------------
# Native pool lifecycle (cthreads._ext.pool) — no kernel compile
# ---------------------------------------------------------------------------


def _require_native_pool():
    try:
        from cthreads import _ext
    except ImportError as e:
        pytest.skip(f"cthreads._ext unavailable: {e}")
    pool_mod = getattr(_ext, "pool", None)
    if pool_mod is None or not hasattr(pool_mod, "ThreadPool"):
        pytest.skip("cthreads._ext.pool.ThreadPool missing — rebuild extension")
    return pool_mod.ThreadPool


def test_native_capacity_zero_rejected():
    Native = _require_native_pool()
    with pytest.raises(Exception):
        Native(0)


def test_native_start_stop_restart():
    Native = _require_native_pool()
    pool = Native(2)
    assert pool.capacity == 2
    pool.start()
    assert list(pool.is_running()) == [False, False]
    assert pool.is_running(0) is False
    with pytest.raises(Exception, match="already started"):
        pool.start()
    pool.stop()
    pool.start()
    pool.stop()


def test_native_is_running_out_of_range():
    Native = _require_native_pool()
    pool = Native(2)
    pool.start()
    try:
        with pytest.raises(IndexError):
            pool.is_running(99)
        with pytest.raises(IndexError):
            pool.is_running(-1)
    finally:
        pool.stop()


def test_native_queue_limit_default_and_reject():
    Native = _require_native_pool()
    unlimited = Native(1)
    assert unlimited.queue_limit == -1
    unlimited = Native(1, None)
    assert unlimited.queue_limit == -1

    pool = Native(1, 1)
    assert pool.queue_limit == 1
    pool.start()
    try:
        # Saturate with a long task via Python facade in kernel tests;
        # here only check the property + double-start still works with limit.
        pass
    finally:
        pool.stop()


def test_python_threadpool_queue_limit_none():
    try:
        import cthreads._ext  # noqa: F401
    except ImportError as e:
        pytest.skip(f"cthreads._ext unavailable: {e}")
    from cthreads import ThreadPool

    assert ThreadPool(2).queue_limit == -1
    assert ThreadPool(2, None).queue_limit == -1
    assert ThreadPool(2, 4).queue_limit == 4


def test_python_threadpool_wraps_native():
    try:
        import cthreads._ext  # noqa: F401
    except ImportError as e:
        pytest.skip(f"cthreads._ext unavailable: {e}")
    from cthreads import ThreadPool

    pool = ThreadPool(3)
    assert pool.capacity == 3
    pool.start()
    assert pool.is_running() == [False, False, False]
    pool.stop()


def test_facade_submit_before_start_and_rejects_plain_fn(tmp_module):
    try:
        import cthreads._ext  # noqa: F401
    except ImportError as e:
        pytest.skip(f"cthreads._ext unavailable: {e}")
    from cthreads import ThreadPool

    pool = ThreadPool(1)
    with pytest.raises(TypeError, match="@Thread"):
        pool.submit(lambda: None)

    mod = tmp_module(
        """
        from cthreads import Thread

        @Thread
        def add(a: int, b: int) -> int:
            return a + b
        """,
        name="ct_pool_before_start",
    )
    # Not started and/or kernels not prepared — must not silently enqueue.
    with pytest.raises(Exception):
        pool.submit(mod.add, 1, 2)


# ---------------------------------------------------------------------------
# Kernel submit / Job surface (needs compile + build)
# ---------------------------------------------------------------------------


def _prepare_pool_kernels(tmp_module, name: str):
    try:
        import cthreads._ext  # noqa: F401
    except ImportError as e:
        pytest.skip(f"cthreads._ext unavailable: {e}")

    from cthreads import unload_kernels

    mod = tmp_module(
        """
        from cthreads import Thread, Threadable

        @Thread
        def add(a: int, b: int) -> int:
            return a + b

        @Thread
        def mul(a: int, b: int) -> int:
            return a * b

        @Threadable
        class Box:
            value: int

        @Thread
        def bump(box: Box, n: int) -> None:
            box.value = box.value + n

        @Thread
        def busy(n: int, sink: list[int]) -> int:
            # Side effects so /O2 cannot delete the delay window.
            i: int = 0
            while i < n:
                sink.append(0)
                sink.pop()
                i = i + 1
            return i
        """,
        name=name,
    )
    try:
        prepare_and_load_or_skip(force=True)
    except RuntimeError as e:
        skip_if_kernel_runtime_error(e)
    return mod, unload_kernels


@pytest.fixture
def pool_kernels(tmp_module):
    mod, unload = _prepare_pool_kernels(tmp_module, "ct_pool_kernels")
    yield mod
    try:
        unload()
    except Exception:
        pass


def test_pool_group_join_results(pool_kernels):
    from cthreads import ThreadPool
    from cthreads.pool import JobGroup

    pool = ThreadPool(4).start()
    try:
        group = pool.group(pool_kernels.add, [(1, 2), (3, 4), (5, 6)])
        assert isinstance(group, JobGroup)
        assert len(group) == 3
        assert group.results() == [3, 7, 11]
    finally:
        pool.stop()


def test_pool_group_await(pool_kernels):
    from cthreads import ThreadPool

    pool = ThreadPool(4).start()

    async def main():
        return await pool.group(pool_kernels.mul, [(2, 3), (4, 5)])

    try:
        assert asyncio.run(main()) == [6, 20]
    finally:
        pool.stop()


def test_pool_queue_limit_rejects(pool_kernels):
    from cthreads import ThreadPool

    busy_n = 2_000_000
    sink: list[int] = []
    pool = ThreadPool(1, queue_limit=1).start()
    try:
        blocker = pool.submit(pool_kernels.busy, busy_n, sink)
        deadline = time.perf_counter() + 15.0
        while (
            time.perf_counter() < deadline
            and not blocker.done()
            and not any(pool.is_running())
        ):
            time.sleep(0.0005)
        if blocker.done():
            pytest.skip("busy finished too quickly to exercise queue limit")
        # One slot filled by the in-flight/queued blocker; next submit must fit
        # or reject depending on whether blocker left the queue. Fill the limit.
        queued = pool.submit(pool_kernels.add, 1, 1)
        with pytest.raises(RuntimeError, match="queue limit"):
            pool.submit(pool_kernels.add, 2, 2)
        pool.stop()
        blocker.join()
        try:
            queued.join()
        except RuntimeError:
            pass
    finally:
        try:
            pool.stop()
        except Exception:
            pass


def test_pool_submit_join_result(pool_kernels):
    from cthreads import ThreadPool

    pool = ThreadPool(2)
    pool.start()
    try:
        job = pool.submit(pool_kernels.add, 2, 3)
        assert isinstance(job, Job)
        assert job._started is True
        job.join()
        assert job.done()
        assert job.result() == 5
    finally:
        pool.stop()


def test_pool_submit_kwargs(pool_kernels):
    from cthreads import ThreadPool

    pool = ThreadPool(1).start()
    try:
        job = pool.submit(pool_kernels.add, a=10, b=7)
        job.join()
        assert job.result() == 17
    finally:
        pool.stop()


def test_pool_submit_await(pool_kernels):
    from cthreads import ThreadPool

    pool = ThreadPool(2).start()

    async def main():
        return await pool.submit(pool_kernels.mul, 6, 7)

    try:
        assert asyncio.run(main()) == 42
    finally:
        pool.stop()


def test_pool_many_jobs_results(pool_kernels):
    from cthreads import ThreadPool

    pool = ThreadPool(4).start()
    try:
        jobs = [pool.submit(pool_kernels.add, i, i * 2) for i in range(32)]
        results = []
        for j in jobs:
            j.join()
            results.append(j.result())
        assert results == [i + i * 2 for i in range(32)]
    finally:
        pool.stop()


def test_pool_concurrent_await(pool_kernels):
    from cthreads import ThreadPool

    pool = ThreadPool(4).start()

    async def main():
        jobs = [pool.submit(pool_kernels.add, i, 1) for i in range(16)]
        return await asyncio.gather(*jobs)

    try:
        assert asyncio.run(main()) == list(range(1, 17))
    finally:
        pool.stop()


def test_pool_threadable_writeback(pool_kernels):
    from cthreads import ThreadPool

    box = pool_kernels.Box()
    box.value = 10
    pool = ThreadPool(2).start()
    try:
        job = pool.submit(pool_kernels.bump, box, 5)
        job.join()
        assert box.value == 15
    finally:
        pool.stop()


def test_pool_submit_rejects_plain_function():
    try:
        import cthreads._ext  # noqa: F401
    except ImportError as e:
        pytest.skip(f"cthreads._ext unavailable: {e}")
    from cthreads import ThreadPool

    pool = ThreadPool(1).start()
    try:
        with pytest.raises(TypeError, match="@Thread"):
            pool.submit(lambda x: x, 1)
    finally:
        pool.stop()


def test_pool_submit_missing_meta_without_compile(tmp_module):
    try:
        import cthreads._ext  # noqa: F401
    except ImportError as e:
        pytest.skip(f"cthreads._ext unavailable: {e}")
    from cthreads import ThreadPool

    mod = tmp_module(
        """
        from cthreads import Thread

        @Thread
        def orphan(a: int) -> int:
            return a
        """,
        name="ct_pool_orphan",
    )
    # Decorated but never prepare/compile → no usable trampoline / meta load.
    pool = ThreadPool(1).start()
    try:
        with pytest.raises(Exception):
            pool.submit(mod.orphan, 1)
    finally:
        pool.stop()


def test_pool_stop_drops_queued_jobs(pool_kernels):
    """Queued (not yet running) jobs error when the pool stops."""
    from cthreads import ThreadPool

    # Keep one worker busy with side-effecting work; extras stay queued.
    busy_n = 2_000_000
    sink: list[int] = []
    pool = ThreadPool(1).start()
    try:
        blocker = pool.submit(pool_kernels.busy, busy_n, sink)
        deadline = time.perf_counter() + 15.0
        while (
            time.perf_counter() < deadline
            and not blocker.done()
            and not any(pool.is_running())
        ):
            time.sleep(0.0005)
        if blocker.done():
            pytest.skip("busy job finished before queue could be filled")
        assert not blocker.done()

        queued = [pool.submit(pool_kernels.add, i, 1) for i in range(8)]
        assert not blocker.done(), "blocker finished before stop — increase busy_n"
        pool.stop()

        blocker.join()
        assert blocker.done()
        assert blocker.result() == busy_n

        dropped = 0
        for j in queued:
            try:
                j.join()
            except RuntimeError as e:
                assert "dropped" in str(e).lower() or "stopped" in str(e).lower()
                dropped += 1
        assert dropped >= 1, f"expected queued jobs dropped; dropped={dropped}"
    finally:
        try:
            pool.stop()
        except Exception:
            pass


def test_pool_restart_does_not_rerun_old_queue(pool_kernels):
    from cthreads import ThreadPool

    busy_n = 2_000_000
    sink: list[int] = []
    pool = ThreadPool(1).start()
    try:
        # Keep the Job alive for the full run (pool workers hold a shared_ptr,
        # but dropping the Python handle early has crashed on Windows).
        busy_job = pool.submit(pool_kernels.busy, busy_n, sink)
        deadline = time.perf_counter() + 15.0
        while (
            time.perf_counter() < deadline
            and not busy_job.done()
            and not any(pool.is_running())
        ):
            time.sleep(0.0005)
        if busy_job.done():
            pytest.skip("busy job finished before queue could be filled")
        assert not busy_job.done()

        abandoned = [pool.submit(pool_kernels.add, 100, i) for i in range(4)]
        assert not busy_job.done(), "busy finished before stop — increase busy_n"
        pool.stop()

        busy_job.join()
        for j in abandoned:
            with pytest.raises(RuntimeError):
                j.join()

        pool.start()
        job = pool.submit(pool_kernels.add, 1, 1)
        job.join()
        assert job.result() == 2
    finally:
        try:
            pool.stop()
        except Exception:
            pass


def test_dedicated_thread_still_independent(pool_kernels):
    """cthreads.thread() must keep working alongside a live pool."""
    from cthreads import ThreadPool, thread

    pool = ThreadPool(2).start()
    try:
        pj = pool.submit(pool_kernels.add, 1, 2)
        dj = thread(pool_kernels.mul, 3, 4, force=False)
        dj.start()
        pj.join()
        dj.join()
        assert pj.result() == 3
        assert dj.result() == 12
    finally:
        pool.stop()


def test_pool_job_start_is_noop(pool_kernels):
    from cthreads import ThreadPool

    pool = ThreadPool(1).start()
    try:
        job = pool.submit(pool_kernels.add, 4, 5)
        assert job.start() is job
        job.start()
        job.join()
        assert job.result() == 9
    finally:
        pool.stop()


def test_pool_binary_path_exists_after_prepare(pool_kernels):
    from cthreads import BINARY_PATH

    assert BINARY_PATH is not None
    assert Path(BINARY_PATH).is_file()


def test_pool_wait_and_done(pool_kernels):
    from cthreads import ThreadPool

    pool = ThreadPool(2).start()
    try:
        job = pool.submit(pool_kernels.add, 8, 1)
        job.wait()
        assert job.done()
        assert job.result() == 9
    finally:
        pool.stop()
