#!/usr/bin/env python3
"""Retarget this tree to build/publish the cthreads-gpu PyPI distribution.

Pip extras cannot select a different binary for the same project name. GPU-enabled
wheels are therefore published as ``cthreads-gpu`` (same import path ``cthreads``),
and ``pip install cthreads[gpu]`` depends on that package.

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
    # Avoid a self-referential optional extra on the GPU distribution.
    old_extra = 'gpu = ["cthreads-gpu=='
    if old_extra in text:
        # Replace the gpu extra line with an empty marker list.
        lines: list[str] = []
        for line in text.splitlines(keepends=True):
            if line.startswith("gpu = ["):
                lines.append(
                    "gpu = []  "
                    "# GPU wheels are this distribution; extra is a no-op here\n"
                )
            else:
                lines.append(line)
        text = "".join(lines)
    desc = 'description = "Compile @Threadable / @Thread Python into native C++ kernels and run them off the GIL."'
    gpu_desc = (
        'description = "cthreads with Vulkan GPU (@Gpu) support built into _ext. '
        'Install via pip install cthreads[gpu] or pip install cthreads-gpu."'
    )
    if desc in text:
        text = text.replace(desc, gpu_desc, 1)
    path.write_text(text, encoding="utf-8")
    print("retargeted pyproject.toml -> name = \"cthreads-gpu\"")


if __name__ == "__main__":
    main()
