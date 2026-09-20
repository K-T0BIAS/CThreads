# Contributor documentation

This page is aimed at contributers and those interested in the inner workings of the cthreads library. For normal usage we recommend sticking to [quickstart.md](./quickstart.md).

## Guides in this tree

| Document | What it covers |
|----------|----------------|
| [release.md](./release.md) | GitHub Actions, TestPyPI, PyPI trusted publishing |
| [COMPILER.md](./COMPILER.md) | How the Python-to-C++ translator works |
| [Example.md](./Example.md) | Long codegen walkthrough |
| [sync_state_docs.md](./sync_state_docs.md) | Mid-run writeback bridge and TLS details |
| [vk_guide/README.md](./vk_guide/README.md) | Vulkan compute substrate tutorial (ordered chapters) |
| [internals/gpu/README.md](./internals/gpu/README.md) | C++ GPU modules (Context through launch) |
| [gpu_future_cpu_to_gpu.md](./gpu_future_cpu_to_gpu.md) | Design note: CPU kernels launching GPU |
| [STYLE.md](./STYLE.md) | Writing and code style for this repo |

Start with [vk_guide/00-read-me-first.md](./vk_guide/00-read-me-first.md) if you are new to Vulkan but know the CPU pack model.
