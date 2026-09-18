# Install

cthreads is a Python package with a **native extension**. Installing from PyPI gives you a wheel (or an sdist that compiles `_ext` with CMake). Later, the first `cthreads.thread(...)` compiles your `@Thread` kernels with a C++ compiler (no CMake for that step).

## Requirements

| | `pip install` (`_ext`) | First kernel `prepare` / `thread(...)` |
|---|---|---|
| Python **3.10+** | yes | yes |
| C++17 compiler | yes (sdist / editable); wheels ship a prebuilt `_ext` on Linux/Windows x86_64 | yes |
| CMake **3.18+** | yes for sdist / editable; not needed when using a wheel | no |
| pybind11 / scikit-build-core | pulled in by pip when building from source | no |

You always need a **C++ compiler** for kernel builds. CMake is only for building the installed `_ext` extension from source.

## From PyPI (recommended)

```bash
python -m venv .venv
# activate the venv, then:
python -m pip install -U pip
python -m pip install cthreads
# optional Vulkan GPU build (full package; do not also install cthreads):
# python -m pip install cthreads-gpu
```

Check:

```python
import cthreads
print(cthreads._ext)   # native module from the wheel (or built from sdist)
```

Package page: [https://pypi.org/project/cthreads/](https://pypi.org/project/cthreads/).  
Source and docs: [https://github.com/K-T0BIAS/CThreads](https://github.com/K-T0BIAS/CThreads).

Optional tests (clone the repo, or install the test extra if published):

```bash
python -m pip install "cthreads[test]"
# from a clone:
pytest
```

## From this repo (editable / contributors)

Use a virtualenv. You can put **CMake and Ninja in the venv** so you do not need a system CMake:

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install cmake ninja
python -m pip install -e ".[test]"
```

Linux / macOS:

```bash
source .venv/bin/activate
python -m pip install -U pip
python -m pip install cmake ninja
python -m pip install -e ".[test]"
```

`pip install cmake` installs the Kitware CMake wheel into the venv. With the venv **activated**, `cmake` is on `PATH`, so the isolated build can find it. `ninja` is optional; scikit-build-core uses it when present (faster than MSBuild / Make).

pybind11 and scikit-build-core are **not** something you install by hand for PyPI wheels. For editable installs, `pyproject.toml` lists them as build-system requires so pip fetches them when building.

`.[test]` adds pytest. For a runtime-only editable install:

```bash
python -m pip install -e .
```

### Check the install

```python
import cthreads
print(cthreads._ext)   # native module; fail here means the CMake/C++ build did not land
```

Optional tests:

```bash
pytest
```

## Compilers (per OS)

CMake in the venv does **not** replace a compiler. Install one of these, then keep using the venv `cmake` / `ninja` if you want.

### Windows

Install **[Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)** (or full Visual Studio) with the **Desktop development with C++** workload (MSVC, Windows SDK).

- `pip install -e .` uses CMake's Visual Studio / MSVC generator.
- Kernel builds look for `cl.exe` (via `PATH`, `CXX`, or `vswhere`). If `cl` is not on `PATH`, cthreads still finds a recent MSVC install and runs `vcvars64.bat` for the kernel link.

You can use clang or g++ on Windows if they are on `PATH` first (`CXX` overrides). Prefer **one toolchain** for `_ext` and kernels; MSVC for both is the usual Windows setup.

### Linux

```bash
# Debian / Ubuntu
sudo apt install build-essential python3-dev

# Fedora
sudo dnf install gcc-c++ python3-devel
```

`build-essential` / `gcc-c++` gives `g++`. `python3-dev` / `python3-devel` is required so CMake can compile against your Python.

A system CMake (`apt install cmake`) also works. The venv `pip install cmake ninja` path is enough if you do not want a distro CMake.

### macOS

```bash
xcode-select --install
```

That provides `clang++`. Then use venv CMake as above, or `brew install cmake` if you prefer a system binary.

Apple Silicon and Intel both work; the same C++17 + CMake flow applies.

## What gets built

1. **`pip install -e .`** - CMake configures `src/cthreads/cpp`, compiles `cthreads._ext` (Release, C++17). The module is copied next to the Python package so editable imports work (including on Windows).
2. **First `cthreads.thread(...)`** (or `prepare()` + `load_kernels()`) - codegen emits C++ for your `@Thread` / `@Threadable` types and links `cthreads_kernels` with `cl` / `g++` / `clang++`. Later launches reuse the cache until the annotated source changes.

The linalg extension is compiled with **AVX2** (`/arch:AVX2` on MSVC, `-mavx2 -mfma` elsewhere). That matches current x86_64 machines; very old CPUs without AVX2 are not a supported `_ext` target.

## Rebuilds

First `thread(...)` runs cache-checked `prepare` + `load_kernels`. After you change `@Thread` / `@Threadable` code:

```python
import cthreads

cthreads.unload_kernels()          # required on Windows before relinking a loaded DLL
cthreads.thread(fn, *args, force=True)
# or: cthreads.prepare(force=True) then cthreads.load_kernels()
```

Calling `thread(..., force=True)` while kernels are still loaded raises. Unload first.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `CMake was not found` / `cmake` missing | Activate the venv, `pip install cmake`, confirm `cmake --version` (needs **3.18+**). Or install a system CMake and keep it on `PATH`. |
| `No C++ compiler found` on first `thread(...)` | Install MSVC Build Tools / `build-essential` / Xcode CLT. Optionally set `CXX` to `cl`, `g++`, or `clang++`. |
| pip cannot compile `_ext` on Windows | Install the C++ workload. Retry from an **x64 Native Tools** prompt if CMake still cannot see MSVC. |
| `Python.h` / Development.Module missing (Linux) | Install `python3-dev` (or `python3.12-dev` matching the venv interpreter). |
| `thread(force=True)` errors about loaded kernels | `unload_kernels()` first, then force-rebuild. |
| Editable import finds Python but not `_ext` | Re-run `pip install -e .` with the venv active so the post-build copy lands beside `cthreads/`. |
| `LoadLibrary` error **4551** on Windows | Smart App Control blocking the unsigned `cthreads_kernels.dll` compiled in your project. Turn SAC off or use WSL/Linux. See [release.md](./release.md). |

## GPU (Vulkan compute)

From **0.2.0**, cthreads can run `@Gpu` kernels on a Vulkan compute device when the
native extension includes GPU support and the machine has a working Vulkan ICD
(normally installed with your GPU drivers).

### End users (PyPI)

| Install | `_ext` contents |
|---------|-----------------|
| `pip install cthreads` | CPU only (`CTHREADS_GPU` off) |
| `pip install cthreads-gpu` | Full package with GPU in `_ext` (same import: `cthreads`) |

These are **mutually exclusive**. Both ship `cthreads` / `_ext`; installing both
overwrites the extension. Prefer `cthreads-gpu` when you need `@Gpu`.

```python
from cthreads import gpu

if gpu.available():
    print(gpu.device_name())
else:
    print("GPU path not usable (no Vulkan ICD / device); CPU @Thread still works")
```

You need GPU **drivers** with a Vulkan ICD. You do **not** need the LunarG SDK
only to run. User guides: [guide/gpu/README.md](./guide/gpu/README.md).

### Contributors (editable / source)

Default editable builds leave `CTHREADS_GPU` **OFF**. To compile GPU into your
local `_ext`:

```powershell
# PowerShell
$env:CMAKE_ARGS="-DCTHREADS_GPU=ON"
pip install -e ".[test]"
```

```bash
export CMAKE_ARGS="-DCTHREADS_GPU=ON"
pip install -e ".[test]"
```

Or:

```bash
pip install -e . --config-settings=cmake.define.CTHREADS_GPU=ON
```

Building with `CTHREADS_GPU=ON` needs Vulkan **headers** (LunarG SDK or distro
`libvulkan-dev`). Runtime still loads the loader dynamically; end users need
drivers, not the SDK.

Release packaging (CPU + GPU wheels): [release.md](./release.md).
More detail: [vk_guide/03-sdk-runtime-drivers.md](./vk_guide/03-sdk-runtime-drivers.md)
and [guide/gpu/errors.md](./guide/gpu/errors.md).

## Publishing to PyPI

Wheels and the sdist are built on GitHub Actions when a GitHub Release is published.
End users: `pip install cthreads` or `pip install cthreads-gpu`. Maintainer
walkthrough: [release.md](./release.md).

## Next

- [README](../README.md) - `@Thread` / `@Threadable` and first `thread(...)` / `join` / `await`
- [concepts](./concepts.md) - GIL, pack / writeback, rules
- [GPU guides](./guide/gpu/README.md) - `@Gpu`, `gpu()`, arena, barriers
- [Guides](./index.md) - pools, sync, jobs, math
