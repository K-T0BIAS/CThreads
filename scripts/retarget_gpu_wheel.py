#!/usr/bin/env python3
"""Retarget this tree to build/publish the cthreads-gpu PyPI distribution.

GPU-enabled wheels are a separate PyPI project (``cthreads-gpu``) with the same
import path ``cthreads``. Prefer ``pip install cthreads-gpu`` for GPU; do not
install ``cthreads`` and ``cthreads-gpu`` together (they both ship ``_ext``).

Run from the repo root before cibuildwheel / ``python -m build`` for the GPU job.
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "pyproject.toml"
    text = path.read_text(encoding="utf-8")
    if 'name = "cthreads-gpu"' in text:
        print("pyproject.toml already retargeted to cthreads-gpu")
        return
    if 'name = "cthreads"' not in text:
        raise SystemExit("expected name = \"cthreads\" in pyproject.toml")
    text = text.replace('name = "cthreads"', 'name = "cthreads-gpu"', 1)
    desc = 'description = "Compile @Threadable / @Thread Python into native C++ kernels and run them off the GIL."'
    gpu_desc = (
        'description = "cthreads with Vulkan GPU (@Gpu) support built into _ext. '
        'Install via pip install cthreads-gpu (do not install alongside cthreads)."'
    )
    if desc in text:
        text = text.replace(desc, gpu_desc, 1)
    path.write_text(text, encoding="utf-8")
    print("retargeted pyproject.toml -> name = \"cthreads-gpu\"")


if __name__ == "__main__":
    main()
