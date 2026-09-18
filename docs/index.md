# CTHREADS GUIDES:

| description | link | 
|-|-|
| **Install**: Python, C++ compiler, CMake (including CMake/Ninja in a venv); GPU / Vulkan notes | [link](./install.md) |
| Important ``concepts for beginners``. CThreads ``tricks``, ``best practices`` and an ``introduction to multithreading`` | [link](./concepts.md) | 
| `@Thread` / `@Threadable`: types, kernel language subset, structs, methods, constructor | [link](./guide/thread_and_threadable.md) | 
| Indepth guide on ``Thread Pools`` for professional thread management | [link](./guide/pools.md) |
| `cthreads.sync`: Guides on how to use the native `sync` module (includes **`LOCKS`** and **`state synchronization`**). Explains how `C++` and `Python` interact in the threaded environment and how to ensure memory validity | [link](./guide/sync.md) |
| Guide to the internal **`linalg`** and **`math`** modules for high performance math tasks. Includes: `tensor (cthreads.Array)`, `cmath` and `python stdlib math` | [link](./guide/math_and_linalg.md) |
| How to use `cthreads.Job` for custom thread handling and `async` applications | [link](./guide/jobs.md) |
| **GPU (0.2.0):** `@Gpu` / `gpu()` / `GpuArena` / workgroup barriers - concepts, quickstart, best practices, examples | [link](./guide/gpu/README.md) |
| `cthreads documentation` | [link](./COMPILER.md) |
| **Release**: GitHub Actions, TestPyPI, PyPI trusted publishing | [link](./release.md) |
| End-to-end example (`@Thread` / `@Threadable` through codegen) | [link](./Example.md) |
| **Vulkan / GPU contributor guide** (substrate, not the first user path) | [link](./vk_guide/README.md) |
| **Internals:** GPU C++ modules (Context -> launch/join) | [link](./internals/gpu/README.md) |
| **Future:** CPU `@Thread` launching GPU | [link](./gpu_future_cpu_to_gpu.md) |
