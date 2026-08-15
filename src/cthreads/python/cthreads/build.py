"""
Compile generated C++ sources into one shared library.

Requires `CompileSession.compile()` to have filled
`REGISTRY.threadable_units` / `thread_units` first.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import runtime
from .frontend.Registry import REGISTRY

BINARY_STEM = "cthreads_kernels"


def _locate_vs_cl() -> str | None:
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None

    probe = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-find",
            r"VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        ],
        capture_output=True,
        text=True,
    )
    lines = [ln.strip() for ln in probe.stdout.splitlines() if ln.strip()]
    return lines[0] if lines else None


def _vcvars_for_cl(cl_path: str) -> Path | None:
    p = Path(cl_path).resolve()
    for parent in p.parents:
        candidate = parent / "Auxiliary" / "Build" / "vcvars64.bat"
        if candidate.is_file():
            return candidate
    return None


def _detect_compiler() -> tuple[str, str]:
    env_cxx = os.environ.get("CXX")
    candidates: list[str] = []
    if env_cxx:
        candidates.append(env_cxx)
    if sys.platform == "win32":
        candidates.extend(["clang++", "g++", "c++", "cl"])
    else:
        candidates.extend(["c++", "clang++", "g++"])

    for name in candidates:
        path = shutil.which(name)
        if path:
            flavor = "msvc" if Path(path).stem.lower() == "cl" else "gnu"
            return path, flavor

    if sys.platform == "win32":
        cl = _locate_vs_cl()
        if cl:
            return cl, "msvc"

    raise RuntimeError(
        "No C++ compiler found. Set CXX or install clang/gcc "
        "(or MSVC Build Tools on Windows)."
    )


def _project_root_from_units() -> Path:
    roots: set[Path] = set()
    for unit in REGISTRY.threadable_units.values():
        roots.add(unit.hpp_path.resolve().parent.parent)
    for unit in REGISTRY.thread_units.values():
        if unit.owner is not None or unit.hpp_path is None:
            continue
        roots.add(unit.hpp_path.resolve().parent.parent)

    if not roots:
        raise RuntimeError(
            "No compiled units — call CompileSession.compile() before build()"
        )
    if len(roots) == 1:
        return next(iter(roots))
    try:
        return Path(os.path.commonpath([str(r) for r in roots]))
    except ValueError as e:
        raise RuntimeError(
            f"Cannot determine a single project root from unit paths: {roots}"
        ) from e


def _collect_sources_and_includes() -> tuple[list[Path], list[Path]]:
    sources: list[Path] = []
    include_dirs: set[Path] = set()
    thread_dirs: set[Path] = set()

    for unit in REGISTRY.threadable_units.values():
        cpp_path = unit.cpp_path.resolve()
        if not cpp_path.is_file():
            raise FileNotFoundError(f"Missing generated source: {cpp_path}")
        sources.append(cpp_path)
        include_dirs.add(unit.hpp_path.resolve().parent)
        thread_dirs.add(unit.hpp_path.resolve().parent.parent / "__Thread__")

    for unit in REGISTRY.thread_units.values():
        if unit.owner is not None:
            continue
        if unit.cpp_path is None or unit.hpp_path is None:
            raise RuntimeError(f"Free ThreadUnit {unit.handle.name} missing paths")
        cpp_path = unit.cpp_path.resolve()
        if not cpp_path.is_file():
            raise FileNotFoundError(f"Missing generated source: {cpp_path}")
        sources.append(cpp_path)
        include_dirs.add(unit.hpp_path.resolve().parent)
        thread_dirs.add(unit.hpp_path.resolve().parent)

    # Bundled runtime headers: .../python/cthreads/V2/build.py -> .../cpp/headers
    runtime_headers = (
        Path(__file__).resolve().parent.parent.parent / "cpp" / "headers"
    )
    if runtime_headers.is_dir():
        include_dirs.add(runtime_headers)

    sync_bridge = (
        Path(__file__).resolve().parent.parent.parent
        / "cpp"
        / "runtime"
        / "sync_bridge.cpp"
    )
    if sync_bridge.is_file():
        sources.append(sync_bridge)

    for thread_dir in thread_dirs:
        tbuf_cpp = thread_dir / "cthreads_tbuffer.cpp"
        if tbuf_cpp.is_file():
            sources.append(tbuf_cpp)
            include_dirs.add(thread_dir)

    return sorted(set(sources)), sorted(include_dirs)


def _binary_name() -> str:
    if sys.platform == "win32":
        return f"{BINARY_STEM}.dll"
    if sys.platform == "darwin":
        return f"lib{BINARY_STEM}.dylib"
    return f"lib{BINARY_STEM}.so"


def build(
    project_root: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Compile all unit-backed `.cpp` files into one shared library.

    Returns the path to the built shared library and sets `runtime.BINARY_PATH`.
    """
    from .cache import ensure_gitignore, hash_files, load_cache, save_cache

    compiler, flavor = _detect_compiler()
    sources, include_dirs = _collect_sources_and_includes()
    root = Path(project_root).resolve() if project_root else _project_root_from_units()
    root.mkdir(parents=True, exist_ok=True)
    ensure_gitignore(root)

    out = root / _binary_name()
    link_hash = hash_files(sources) + "|" + flavor
    cache = load_cache(root)

    if (
        not force
        and out.is_file()
        and cache.get("link_hash") == link_hash
        and cache.get("binary") == str(out)
    ):
        runtime.BINARY_PATH = str(out)
        return out

    if flavor == "msvc":
        cl_args = ["/nologo", "/O2", "/LD", "/EHsc", "/std:c++17"]
        for inc in include_dirs:
            cl_args.append(f"/I{inc}")
        cl_args.extend(str(s) for s in sources)
        cl_args.append(f"/Fe:{out}")

        vcvars = _vcvars_for_cl(compiler)
        if vcvars:
            quoted = subprocess.list2cmdline([compiler, *cl_args])
            bat = root / "_cthreads_build.bat"
            bat.write_text(
                f'@echo off\r\ncall "{vcvars}"\r\n{quoted}\r\n',
                encoding="utf-8",
            )
            cmd = ["cmd", "/c", str(bat)]
        else:
            cmd = [compiler, *cl_args]
    else:
        cmd = [compiler, "-shared", "-O2", "-std=c++17"]
        if sys.platform != "win32":
            cmd.append("-fPIC")
        for inc in include_dirs:
            cmd.extend(["-I", str(inc)])
        cmd.extend(str(s) for s in sources)
        cmd.extend(["-o", str(out)])

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root))

    if flavor == "msvc":
        bat = root / "_cthreads_build.bat"
        if bat.is_file():
            bat.unlink()
        for junk in root.glob("*.obj"):
            junk.unlink()
        for junk in root.glob("*.exp"):
            junk.unlink()

    if result.returncode != 0:
        raise RuntimeError(
            "C++ build failed.\n"
            f"Command: {cmd}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if not out.is_file():
        raise RuntimeError(f"Build reported success but missing output: {out}")

    runtime.BINARY_PATH = str(out)
    cache["link_hash"] = link_hash
    cache["binary"] = str(out)
    save_cache(root, cache)
    return out
