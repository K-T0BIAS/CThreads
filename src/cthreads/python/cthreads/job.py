"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Awaitable Job wrapper around `cthreads._ext.Job`.

Public usage::

    job = cthreads.thread(fn, ...)
    result = await job          # auto-starts; does not block the event loop

    # sync still works:
    job = cthreads.thread(fn, ...).start()
    job.join()
    job.result()

    # mid-run Threadable writeback (mirrors kernel `__sync_state()`):
    job.start()
    while not job.done():
        job.sync_state()        # or cthreads.sync_state(job)
        # read Threadable fields on the Python side
"""

import asyncio
from typing import Any, Iterator


class Job:
    """Python-facing Job: delegates to the native handle, adds `await`."""

    __slots__ = ("_raw", "_started")

    def __init__(self, raw: Any) -> None:
        self._raw = raw
        self._started = False

    def start(self) -> "Job":
        if not self._started:
            self._raw.start()
            self._started = True
        return self

    def done(self) -> bool:
        return bool(self._raw.done())

    def join(self) -> None:
        """Block this OS thread until the kernel finishes (releases the GIL)."""
        if not self._started:
            self.start()
        self._raw.join()

    def wait(self) -> None:
        """Block until done (condition wait); prefer `join` to reap the thread."""
        if not self._started:
            self.start()
        self._raw.wait()

    def result(self) -> Any:
        return self._raw.result()

    def sync_state(self) -> None:
        """
        Steal the job's state mutex and writeback Threadable/list/dict args
        into the live Python objects (same effect as kernel `__sync_state()`).

        The job must have been started. No-op once the kernel has finished
        (final writeback already ran). Blocks until it can take the mutex.
        """
        if not self._started:
            raise RuntimeError(
                "cthreads.Job.sync_state: job not started "
                "(call start() or await / join first)"
            )
        sync = getattr(self._raw, "sync_state", None)
        if sync is None:
            raise RuntimeError(
                "cthreads.Job.sync_state: native Job has no sync_state "
                "(rebuild the cthreads extension)"
            )
        sync()

    async def _await_result(self) -> Any:
        if not self._started:
            self.start()
        if self.done():
            return self.result()
        # join on a worker thread so the event loop stays free; GIL released in C++.
        await asyncio.to_thread(self._raw.join)
        return self.result()

    def __await__(self) -> Iterator[Any]:
        """`await job` -> auto-start, wait off the event loop, return result."""
        return self._await_result().__await__()

    def __repr__(self) -> str:
        state = "done" if self._started and self.done() else (
            "running" if self._started else "pending"
        )
        return f"<cthreads.Job {state}>"


def wrap_job(raw: Any) -> Job:
    """Wrap a native `_ext.Job` (or return `raw` if already wrapped)."""
    if isinstance(raw, Job):
        return raw
    return Job(raw)


def sync_state(job: Job) -> None:
    """
    Host mirror of kernel `__sync_state()`: steal `job`'s state mutex and
    writeback Threadable/list/dict pack fields into the live Python objects.

    Equivalent to `job.sync_state()`.
    """
    if not isinstance(job, Job):
        raise TypeError(
            f"cthreads.sync_state: expected a cthreads.Job, got {type(job).__name__}"
        )
    job.sync_state()
