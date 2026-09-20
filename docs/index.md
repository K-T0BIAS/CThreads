# CTHREADS GUIDES

Catalog of user-facing docs. For a linear first reading path (with a diagram), open the [CPU quickstart](./quickstart.md) or the [GPU quickstart](./guide/gpu/quickstart.md).

| description | link |
|-|-|
| **Install**: Python, C++ compiler, CMake (including CMake/Ninja in a venv). GPU / Vulkan notes | [link](./install.md) |
| **FAQ**: common pitfalls, marshal traps, cthreads vs cthreads-gpu, where to ask | [link](./FAQ.md) |
| **CPU quickstart**: types, decorators, compile, launch, then short tool map | [link](./quickstart.md) |
| Important concepts for beginners. GIL, pack / writeback, rules, best practices | [link](./concepts.md) |
| `@Thread` / `@Threadable`: types, kernel language subset, structs, methods, constructor | [link](./guide/thread_and_threadable.md) |
| Indepth guide on Thread Pools for professional thread management | [link](./guide/pools.md) |
| `cthreads.sync`: locks, events, TBuffer, and when Python sees kernel writes | [link](./guide/sync.md) |
| How Shared values and packs move between Python and the native module | [link](./guide/marshal_and_module.md) |
| Guide to the linalg and math modules for high performance math tasks | [link](./guide/math_and_linalg.md) |
| How to use `cthreads.Job` for custom thread handling and async applications | [link](./guide/jobs.md) |
| **GPU hub**: `@Gpu` / `gpu()` / `GpuArena` / workgroup barriers | [link](./guide/gpu/README.md) |
| **GPU quickstart**: probe, first kernel, launch, arena, then tool map | [link](./guide/gpu/quickstart.md) |
| Compact CPU-oriented API reference | [link](./API.md) |

## Contributor documentation

Implementation notes, Vulkan substrate tutorials, release packaging, and compiler internals live behind one page: [contributor.md](./contributor.md).
