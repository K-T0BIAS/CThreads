"""Unit tests for awaitable cthreads.Job wrapper."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from cthreads.job import Job, wrap_job


class _FakeRaw:
    def __init__(self, value=42, delay=0.05, exc: BaseException | None = None):
        self._value = value
        self._delay = delay
        self._exc = exc
        self._done = False
        self._started = False
        self._thread: threading.Thread | None = None
        self.sync_calls = 0

    def start(self):
        if self._started:
            return
        self._started = True

        def run():
            time.sleep(self._delay)
            self._done = True

        self._thread = threading.Thread(target=run)
        self._thread.start()

    def done(self):
        return self._done

    def join(self):
        if self._thread is not None:
            self._thread.join()
        if self._exc is not None:
            raise self._exc

    def wait(self):
        while not self._done:
            time.sleep(0.001)

    def result(self):
        return self._value

    def sync_state(self):
        self.sync_calls += 1


def test_wrap_job_idempotent():
    j = Job(_FakeRaw())
    assert wrap_job(j) is j
    assert isinstance(wrap_job(_FakeRaw()), Job)


def test_job_sync_state_requires_start():
    from cthreads.job import sync_state as sync_state_fn

    job = Job(_FakeRaw())
    with pytest.raises(RuntimeError, match="not started"):
        job.sync_state()
    with pytest.raises(RuntimeError, match="not started"):
        sync_state_fn(job)


def test_job_sync_state_delegates_to_raw():
    from cthreads.job import sync_state as sync_state_fn

    raw = _FakeRaw()
    job = Job(raw)
    job.start()
    job.sync_state()
    sync_state_fn(job)
    assert raw.sync_calls == 2


def test_sync_state_rejects_non_job():
    from cthreads.job import sync_state as sync_state_fn

    with pytest.raises(TypeError, match="expected a cthreads.Job"):
        sync_state_fn(object())  # type: ignore[arg-type]


def test_sync_start_join_result():
    job = Job(_FakeRaw(value=7))
    job.start().join()
    assert job.done()
    assert job.result() == 7


def test_join_auto_starts():
    job = Job(_FakeRaw(value=3, delay=0.02))
    job.join()
    assert job.done()
    assert job.result() == 3


def test_await_job_returns_result():
    async def main():
        job = Job(_FakeRaw(value=99, delay=0.05))
        # preferred API: thread() then await job (auto-start)
        return await job

    assert asyncio.run(main()) == 99


def test_await_does_not_block_event_loop():
    async def main():
        ticks = []

        async def ticker():
            for i in range(5):
                ticks.append(i)
                await asyncio.sleep(0.01)

        job = Job(_FakeRaw(value=1, delay=0.08))
        t = asyncio.create_task(ticker())
        result = await job
        await t
        assert result == 1
        # ticker made progress while job ran
        assert ticks == [0, 1, 2, 3, 4]

    asyncio.run(main())


def test_await_propagates_join_error():
    async def main():
        job = Job(_FakeRaw(exc=RuntimeError("boom"), delay=0.02))
        with pytest.raises(RuntimeError, match="boom"):
            await job

    asyncio.run(main())
