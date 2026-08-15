"""
cthreads.pool.threadPool — fixed thread pools that run @Thread jobs.

Dedicated ``cthreads.thread()`` (one OS thread per Job) is unchanged.
Use a pool when you want a bounded worker set and ``pool.submit(fn, ...)``.

    from cthreads import ThreadPool

    pool = ThreadPool(8)
    pool.start()
    job = pool.submit(fn, *args)   # already queued; await / join / result
    await job
    group = pool.group(fn, [(1, 2), (3, 4)])
    await group
    pool.stop()
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..job import Job, wrap_job
from .group import JobGroup

try:
    from cthreads import _ext as _ext
except ImportError:
    _ext = None  # type: ignore[assignment]

_native = getattr(_ext, "pool", None) if _ext is not None else None
_NativeThreadPool = getattr(_native, "ThreadPool", None) if _native is not None else None


class ThreadPool:
    """Fixed-capacity pool; wraps ``cthreads._ext.pool.ThreadPool``."""

    __slots__ = ("_raw",)

    def __init__(
        self,
        capacity: int,
        queue_limit: int | None = None,
    ) -> None:
        if _NativeThreadPool is None:
            raise RuntimeError(
                "cthreads.ThreadPool: native extension not available — "
                "build/install cthreads._ext"
            )
        lim = -1 if queue_limit is None else int(queue_limit)
        self._raw = _NativeThreadPool(capacity, lim)

    @property
    def capacity(self) -> int:
        return int(self._raw.capacity)

    @property
    def queue_limit(self) -> int:
        """Max pending queued tasks; ``-1`` means unlimited."""
        return int(self._raw.queue_limit)

    def start(self) -> ThreadPool:
        self._raw.start()
        return self

    def stop(self) -> None:
        self._raw.stop()

    def join(self) -> None:
        self._raw.join()

    def is_running(self, thread_id: int | None = None) -> bool | list[bool]:
        if thread_id is None:
            return list(self._raw.is_running())
        return bool(self._raw.is_running(thread_id))

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Job:
        """Enqueue a @Thread function; returns a Job that is already queued."""
        job = wrap_job(self._raw.submit(fn, *args, **kwargs))
        job._started = True
        return job

    def group(self, fn: Any, items: Iterable[Any]) -> JobGroup:
        """
        Submit ``fn`` once per item and return a ``JobGroup``.

        Each item is either:
          - a ``tuple`` / ``list`` of positional args, or
          - a single positional argument.
        """
        jobs: list[Job] = []
        for item in items:
            if isinstance(item, (tuple, list)):
                jobs.append(self.submit(fn, *item))
            else:
                jobs.append(self.submit(fn, item))
        return JobGroup(jobs=jobs)

    def __repr__(self) -> str:
        return (
            f"<cthreads.ThreadPool capacity={self.capacity} "
            f"queue_limit={self.queue_limit}>"
        )


__all__ = ["ThreadPool"]
