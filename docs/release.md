# Releasing cthreads

This is a guide for the release pipeline. For more info or release rights contact [K-T0BIAS](https://github.com/K-T0BIAS)

Releases are built on **GitHub Actions**, not on your PC. Linux wheels are produced on Ubuntu runners. You only click Publish (and approve the `pypi` environment).

PyPI upload is **not** tied to merges into `main`. Broken PRs cannot publish.

## What gets published

| Artifact | Built on | Notes |
|----------|----------|--------|
| Source dist (sdist) | Ubuntu | Users with a compiler can build `_ext` themselves |
| `cthreads` wheels | Ubuntu + Windows | CPU `_ext` (`CTHREADS_GPU=OFF`), Python 3.10-3.13 x86_64 |
| `cthreads-gpu` wheels | Ubuntu + Windows | Same import path `cthreads`, `_ext` built with `CTHREADS_GPU=ON` |

macOS wheels are skipped for now (CMake enables AVX2 on non-MSVC; Apple Silicon would fail).

### CPU vs GPU install

Pip extras cannot swap binary contents of one project name. GPU builds are therefore a
**second PyPI project**:

```bash
pip install cthreads           # CPU wheel
pip install "cthreads[gpu]"    # installs cthreads + cthreads-gpu (GPU _ext)
pip install cthreads-gpu       # GPU wheel only (also provides import cthreads)
```

Keep `project.version` and the `gpu = ["cthreads-gpu==..."]` pin equal on every release
(the Release workflow checks this).

### Trusted Publishing (one-time, both projects)

Add a trusted publisher for **`cthreads`** and another for **`cthreads-gpu`**:

| Field | Value |
|-------|--------|
| Owner | your GitHub user or org |
| Repository | `CThreads` |
| Workflow | `release.yml` |
| Environment | `pypi` or `testpypi` |

Do this on both https://pypi.org and https://test.pypi.org. One `publish` job uploads
all wheels; each wheel goes to the project matching its metadata name.

## One-time setup

### 1. GitHub environments

In the repo: **Settings -> Environments**.

Create:

| Name | Purpose |
|------|---------|
| `pypi` | Production PyPI |
| `testpypi` | TestPyPI (manual workflow runs) |

On **`pypi`**, enable **Required reviewers** and add yourself. The publish job will wait for your approval even after you publish a GitHub Release.

Leave **Deployment branches** as “All” or restrict to tags if you prefer.

### 2. PyPI Trusted Publishing

No API token is stored in GitHub. PyPI trusts this repo + workflow.

**TestPyPI** (do this first):

1. Sign in at [https://test.pypi.org](https://test.pypi.org)
2. Account settings -> **Publishing** (or create the pending project)
3. Add a **trusted publisher** for **`cthreads`**:
   - Owner: your GitHub user or org
   - Repository: `CThreads` (the repo name on GitHub)
   - Workflow: `release.yml`
   - Environment: `testpypi`
4. Add the same trusted publisher for **`cthreads-gpu`**

**PyPI** (same fields, production):

1. Sign in at [https://pypi.org](https://pypi.org)
2. Add a trusted publisher with environment **`pypi`** and workflow **`release.yml`** for **`cthreads`**
3. Add the same trusted publisher again for the **`cthreads-gpu`** project

If the project name `cthreads` / `cthreads-gpu` is not registered yet, use PyPI's **pending publisher** / first-upload flow for that name.

Exact labels in the PyPI UI change occasionally; look for **Trusted publishers** / **Publishing**.

### 3. Branch protection (recommended)

**Settings -> Branches** -> protect `main`:

- Require a pull request
- Require status checks to pass: the **CI** workflow `test (ubuntu-latest, py3.12)` job

Then `main` cannot merge red tests.

## Dry run (TestPyPI)

1. Commit and push the workflows to `main` (or merge a PR). Confirm **CI** is green.
2. **Actions -> Release -> Run workflow**. This **never** uploads to production PyPI.
3. Approve the `testpypi` environment if GitHub asks.
4. Install:

```bash
python -m pip install -U pip
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ cthreads
# GPU build:
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "cthreads[gpu]"
```

`--extra-index-url` is only needed if TestPyPI cannot see some dependency (cthreads currently has none).

## Production release

1. Set `version` in `pyproject.toml` **and** the matching pin
   `gpu = ["cthreads-gpu==X.Y.Z"]` under `[project.optional-dependencies]`.
2. GitHub -> **Releases -> Draft a new release**.
3. Tag `v0.1.0` (or whatever matches `project.version`). Target `main`.
4. Click **Publish release**.
5. Watch **Actions -> Release**. Approve the `pypi` environment when asked.
6. If tests, wheels, or the version check fail, **nothing is uploaded**.

The version job compares `github.event.release.tag_name` (`v0.1.0`) to `project.version` (`0.1.0`). A mismatch fails the release before publish.

PyPI versions cannot be overwritten. If `0.1.0` is bad, yank it and ship `0.1.1`.

## After a release

```bash
python -m pip install cthreads
# or with Vulkan GPU support built into _ext:
python -m pip install "cthreads[gpu]"
```

Windows 11 with Smart App Control on may still fail to **load** `cthreads_kernels.dll` (`LoadLibrary` 4551). That is a Windows policy, not a missing wheel. See [install.md](./install.md).

## Workflow files

| File | Trigger | Publishes? |
|------|---------|------------|
| `.github/workflows/ci.yml` | PR and push to `main` | No (CPU tests + GPU-ON compile smoke) |
| `.github/workflows/release.yml` | GitHub Release published | PyPI (`cthreads` + `cthreads-gpu`) |
| `.github/workflows/release.yml` | Actions -> Run workflow | TestPyPI only |
