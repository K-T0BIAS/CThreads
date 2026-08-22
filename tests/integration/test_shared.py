"""Integration tests for cooperative Shared[T] (SharedHost + marshal promote/demote)."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

import pytest

from helpers import prepare_and_load_or_skip, skip_if_kernel_runtime_error

_INTEGRATION = pytest.mark.integration


def _require_ext():
    try:
        import cthreads._ext  # noqa: F401
    except ImportError as e:
        pytest.skip(f"cthreads._ext unavailable: {e}")


def _shared_kernels(tmp_module, name: str = "ct_shared_integ"):
    _require_ext()
    from cthreads import unload_kernels

    mod = tmp_module(
        textwrap.dedent(
            """
            from cthreads import Thread, Threadable, Shared, __sync_state
            from cthreads.sync import Event, Lock

            @Thread
            def bump_at(head: Shared[list[int]], i: int) -> None:
                head[i] = head[i] + 1

            @Thread
            def bump_zero(head: Shared[list[int]]) -> None:
                head[0] = head[0] + 1

            @Thread
            def bump_zero_locked(head: Shared[list[int]], lock: Lock) -> None:
                lock.acquire()
                head[0] = head[0] + 1
                lock.release()

            @Thread
            def bump_zero_hold(
                head: Shared[list[int]], ready: Event, release: Event
            ) -> None:
                head[0] = head[0] + 1
                ready.set()
                release.wait()

            @Thread
            def sync_bump(head: Shared[list[int]], gate: Event) -> None:
                head.append(10)
                __sync_state()
                gate.wait()
                head.append(20)

            @Thread
            def answer() -> Shared[int]:
                return 42

            @Thread
            def bump_ref(xs: list[int]) -> None:
                xs[0] = xs[0] + 1

            @Threadable
            class Counter:
                n: int

            @Thread
            def inc_counter(c: Shared[Counter], lock: Lock) -> None:
                lock.acquire()
                c.n = c.n + 1
                lock.release()

            @Thread
            def bump_dict(d: Shared[dict[str, int]], lock: Lock) -> None:
                lock.acquire()
                d["k"] = d.get("k", 0) + 1
                lock.release()
            """
        ),
        name=name,
    )
    try:
        prepare_and_load_or_skip(force=True)
    except RuntimeError as e:
        skip_if_kernel_runtime_error(e)
    root = Path(mod.__file__).parent
    bump_cpp = (root / "__Thread__" / "bump_at.cpp").read_text(encoding="utf-8")
    assert "bump_at__promote_a0_shared" in bump_cpp
    return mod, unload_kernels


@pytest.fixture
def shared_kernels(tmp_module):
    mod, unload = _shared_kernels(tmp_module)
    yield mod
    try:
        unload()
    except Exception:
        pass


@_INTEGRATION
def test_shared_list_multi_thread_cooperative(shared_kernels):
    from cthreads import thread

    head = [0, 0, 0, 0]
    jobs = [
        thread(shared_kernels.bump_at, head, i, force=False) for i in range(4)
    ]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    assert head == [1, 1, 1, 1]


@_INTEGRATION
def test_shared_sequential_reseed_after_jobs_finish(shared_kernels):
    from cthreads import thread

    head = [0, 0]
    job = thread(shared_kernels.bump_at, head, 0, force=False)
    job.start()
    job.join()
    assert head == [1, 0]

    head2 = [0, 0]
    job2 = thread(shared_kernels.bump_at, head2, 1, force=False)
    job2.start()
    job2.join()
    assert head2 == [0, 1]


@_INTEGRATION
def test_shared_parallel_two_workers_same_index(shared_kernels):
    from cthreads import thread
    from cthreads.sync import Lock

    head = [0]
    lock = Lock()
    jobs = [
        thread(shared_kernels.bump_zero_locked, head, lock, force=False),
        thread(shared_kernels.bump_zero_locked, head, lock, force=False),
    ]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    assert head[0] == 2


@_INTEGRATION
def test_ref_parallel_two_workers_does_not_double_bump(shared_kernels):
    """Contrast: ref params use per-job pack copies — not cooperative native sharing."""
    from cthreads import thread

    xs = [0]
    jobs = [
        thread(shared_kernels.bump_ref, xs, force=False),
        thread(shared_kernels.bump_ref, xs, force=False),
    ]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    assert xs[0] == 1


@_INTEGRATION
def test_shared_sync_state_mid_run(shared_kernels):
    import time

    import cthreads._ext as ext
    from cthreads import thread
    from cthreads.sync import Event

    ext._debug_reset_ext_sync_invocations()
    head: list[int] = []
    gate = Event()
    job = thread(shared_kernels.sync_bump, head, gate, force=False)
    job.start()
    saw_mid = False
    deadline = time.perf_counter() + 30.0
    try:
        while time.perf_counter() < deadline and not job.done():
            if head == [10]:
                saw_mid = True
                break
            time.sleep(0.0005)
    finally:
        gate.set()
    job.join()
    assert head == [10, 20]
    assert ext._debug_ext_sync_invocations() >= 1
    assert saw_mid


@_INTEGRATION
def test_shared_sync_state_host_api(shared_kernels):
    from cthreads import sync_state, thread
    from cthreads.sync import Event

    head = [1]
    job = thread(shared_kernels.bump_zero, head, force=False)
    job.start()
    job.join()
    assert head == [2]

    head[0] = 5
    ready = Event()
    release = Event()
    job2 = thread(
        shared_kernels.bump_zero_hold, head, ready, release, force=False
    )
    job2.start()
    try:
        ready.wait()
        sync_state(job2)
        assert head == [6]
    finally:
        release.set()
    job2.join()


@_INTEGRATION
def test_shared_return_int(shared_kernels):
    from cthreads import thread

    job = thread(shared_kernels.answer, force=False)
    job.start()
    job.join()
    assert job.result() == 42
    assert job.result() == 42

    job2 = thread(shared_kernels.answer, force=False)
    job2.start()
    job2.join()
    assert job2.result() == 42


@_INTEGRATION
def test_shared_threadable(shared_kernels):
    from cthreads import thread
    from cthreads.sync import Lock

    Counter = shared_kernels.Counter
    c = Counter()
    c.n = 0
    lock = Lock()
    jobs = [
        thread(shared_kernels.inc_counter, c, lock, force=False) for _ in range(3)
    ]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    assert c.n == 3


@_INTEGRATION
def test_shared_dict(shared_kernels):
    from cthreads import thread
    from cthreads.sync import Lock

    d: dict[str, int] = {}
    lock = Lock()
    jobs = [
        thread(shared_kernels.bump_dict, d, lock, force=False) for _ in range(3)
    ]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    assert d["k"] == 3


@_INTEGRATION
def test_pool_submit_shared_workers(shared_kernels):
    from cthreads import ThreadPool

    head = [0, 0, 0, 0]
    pool = ThreadPool(4).start()
    try:
        jobs = [
            pool.submit(shared_kernels.bump_at, head, i) for i in range(4)
        ]
        for j in jobs:
            j.join()
        assert head == [1, 1, 1, 1]
    finally:
        pool.stop()


@_INTEGRATION
def test_two_pools_do_not_share_host_state(shared_kernels):
    """Same Python list passed to two pools must not double-bump via one SharedHost."""
    from cthreads import ThreadPool

    head = [0]
    pool_a = ThreadPool(2).start()
    pool_b = ThreadPool(2).start()
    try:
        ja = pool_a.submit(shared_kernels.bump_zero, head)
        jb = pool_b.submit(shared_kernels.bump_zero, head)
        ja.join()
        jb.join()
        assert head[0] == 1
    finally:
        pool_a.stop()
        pool_b.stop()


@_INTEGRATION
def test_shared_awaitable_job(shared_kernels):
    from cthreads import thread

    async def main():
        head = [0]
        job = thread(shared_kernels.bump_zero, head, force=False)
        await job
        return head

    assert asyncio.run(main()) == [1]
