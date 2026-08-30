# Thread pools

`ThreadPool` runs `@Thread` jobs on a **fixed set of worker threads**. Prefer a pool when you have many short or medium jobs and do not want one OS thread per `cthreads.thread(...)`.

Dedicated `thread(fn, ...)` is unchanged: one job, one OS thread. Pools share workers and (for `Shared[T]`) one **SharedHost** per pool.

```python
from cthreads import Thread, ThreadPool

@Thread
def add(a: int, b: int) -> int:
    return a + b

pool = ThreadPool(4).start()
try:
    job = pool.submit(add, 1, 2)
    print(job.result())  # 3 — already queued; join/await as for any Job

    group = pool.group(add, [(1, 2), (3, 4), (5, 6)])
    print(group.results())  # [3, 7, 11]
finally:
    pool.stop()
```

# Contents

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Thread pools](#thread-pools)
- [Contents](#contents)
- [Lifecycle](#lifecycle)
- [`submit` (one job)](#submit-one-job)
- [`group` (batched Shared-safe wave)](#group-batched-shared-safe-wave)
- [`submit_queue` (pin around many submits)](#submit_queue-pin-around-many-submits)
- [Shared state on a pool](#shared-state-on-a-pool)
- [Queue limit](#queue-limit)
- [See also](#see-also)

<!-- /code_chunk_output -->

# Lifecycle

```python
pool = ThreadPool(capacity, queue_limit=None)
pool.start()   # spawn workers; required before submit
# ... submit / group ...
pool.stop()    # stop workers; in-flight finish, queued tasks are dropped
```

| API | Role |
|-----|------|
| `ThreadPool(n)` | Create a pool with `n` workers (`n >= 1`). |
| `queue_limit` | Max pending queued tasks; `None` / omitted means unlimited (`-1`). |
| `start()` | Start workers. Returns `self` for chaining. Raises if already started. |
| `stop()` | Signal stop, join workers, clear the queue. |
| `join()` | Join worker threads (usually via `stop()`). |
| `capacity` | Number of workers. |
| `is_running()` / `is_running(i)` | Busy flags for workers. |

Pool jobs are **queued at submit time**. `Job.start()` is a no-op; `join` / `await` / `result` work like dedicated jobs.

# `submit` (one job)

```python
job = pool.submit(fn, *args, **kwargs)
```

`fn` must be a compiled `@Thread` function. Returns a `Job` that is already queued.

A bare loop of `submit` calls is fine for independent jobs. If several jobs share `Shared[T]` data on **this** pool, prefer `group` or `submit_queue` (see below). Otherwise a fast job can free the pool SharedHost before later submits have registered.

# `group` (batched Shared-safe wave)

```python
group = pool.group(fn, items)
```

Submits `fn` once per item and returns a `JobGroup`. Each item is either:

- a `tuple` / `list` of positional args, or
- a single positional argument.

```python
group = pool.group(add, [(1, 2), (3, 4)])
group.results()           # blocks until all done
await group               # async: list of results
```

**Shared safety:** `group` builds one batch `[(fn, *args), ...]` and calls the native list `submit` under a single SharedHost **pin**. Early finishes cannot destroy shared slots while the rest of the wave is still being registered and queued.

Use `group` when one function is applied to many arg lists that share cooperative state.

# `submit_queue` (pin around many submits)

```python
with pool.submit_queue():
    jobs = [pool.submit(bump_at, head, i) for i in range(4)]
for j in jobs:
    j.join()
```

`submit_queue` is a context manager that calls native `pin_shared` / `unpin_shared` on the pool’s SharedHost.

Use it when you want one-by-one `submit` (different functions, dynamic args, etc.) but still need one Shared wave. Nested `with pool.submit_queue():` blocks are supported (pin depth nests).

`group` does **not** need `submit_queue`; it pins internally.

# Shared state on a pool

Each `ThreadPool` owns one SharedHost. Jobs submitted to that pool that take `Shared[T]` cooperate on that host.

| Pattern | Safe for overlapping `Shared` submits? |
|---------|----------------------------------------|
| `pool.group(fn, items)` | Yes (batched pin) |
| `with pool.submit_queue():` + `submit` | Yes (explicit pin) |
| Loop of bare `pool.submit(...)` | No — early unregister can free the host mid-wave |

Different pools do **not** share one host. Passing the same Python list into two pools does not make them cooperative with each other.

Dedicated `thread(...)` jobs use a process-wide default host, not a pool host.

# Queue limit

```python
pool = ThreadPool(2, queue_limit=8).start()
```

If the pending queue is already at the limit, `submit` / `group` raise. Use this to bound memory when producers outpace workers.

# See also

- [Jobs](./jobs.md) — `Job` lifecycle, `join` / `await`, writeback
- [Marshal and module](./marshal_and_module.md) — SharedHost, promote / demote
- [Sync](./sync.md) — locks and events inside kernels
