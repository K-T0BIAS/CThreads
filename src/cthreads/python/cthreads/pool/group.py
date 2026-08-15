"""Awaitable group of ``cthreads.Job`` handles."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Iterator

from ..job import Job


@dataclass
class JobGroup:
    jobs: list[Job]

    def join(self) -> None:
        for job in self.jobs:
            job.join()

    def results(self) -> list[Any]:
        """Join all jobs (if needed) and return their results in order."""
        self.join()
        return [job.result() for job in self.jobs]

    def done(self) -> bool:
        return all(job.done() for job in self.jobs)

    async def _await_results(self) -> list[Any]:
        return list(await asyncio.gather(*self.jobs))

    def __await__(self) -> Iterator[Any]:
        return self._await_results().__await__()

    def __len__(self) -> int:
        return len(self.jobs)

    def __repr__(self) -> str:
        return f"<cthreads.JobGroup n={len(self.jobs)}>"


__all__ = ["JobGroup"]
