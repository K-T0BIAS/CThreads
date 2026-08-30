# Releasing cthreads

Releases are built on **GitHub Actions**, not on your PC. Linux wheels are produced on Ubuntu runners. You only click Publish (and optionally approve the `pypi` environment).

PyPI upload is **not** tied to merges into `main`. Broken PRs cannot publish.

## What gets published

| Artifact | Built on | Notes |
|----------|----------|--------|
| Source dist (sdist) | Ubuntu | Users with a compiler can build `_ext` themselves |
| `manylinux` wheels | Ubuntu | Python 3.10–3.13, x86_64 |
| Windows wheels | `windows-latest` | x86_64; kernel DLL is still compiled on the user’s machine |

macOS wheels are skipped for now (CMake enables AVX2 on non-MSVC; Apple Silicon would fail).

## One-time setup

### 1. GitHub environments

In the repo: **Settings → Environments**.

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
2. Account settings → **Publishing** (or create the pending project)
3. Add a **trusted publisher**:
   - Owner: your GitHub user or org
   - Repository: `Better_Threads` (the repo name on GitHub)
   - Workflow: `release.yml`
   - Environment: `testpypi`

**PyPI** (same fields, production):

1. Sign in at [https://pypi.org](https://pypi.org)
2. Add a trusted publisher with environment **`pypi`** and workflow **`release.yml`**

If the project name `cthreads` is not registered yet, use PyPI’s **pending publisher** / first-upload flow for that name.

Exact labels in the PyPI UI change occasionally; look for **Trusted publishers** / **Publishing**.

### 3. Branch protection (recommended)

**Settings → Branches** → protect `main`:

- Require a pull request
- Require status checks to pass: the **CI** workflow `test (ubuntu-latest, py3.12)` job

Then `main` cannot merge red tests.

## Dry run (TestPyPI)

1. Commit and push the workflows to `main` (or merge a PR). Confirm **CI** is green.
2. **Actions → Release → Run workflow**. This **never** uploads to production PyPI.
3. Approve the `testpypi` environment if GitHub asks.
4. Install:

```bash
python -m pip install -U pip
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ cthreads
```

`--extra-index-url` is only needed if TestPyPI cannot see some dependency (cthreads currently has none).

## Production release

1. Set `version` in `pyproject.toml` (must match the tag without the leading `v`).
2. GitHub → **Releases → Draft a new release**.
3. Tag `v0.1.0` (or whatever matches `project.version`). Target `main`.
4. Click **Publish release**.
5. Watch **Actions → Release**. Approve the `pypi` environment when asked.
6. If tests, wheels, or the version check fail, **nothing is uploaded**.

The version job compares `github.event.release.tag_name` (`v0.1.0`) to `project.version` (`0.1.0`). A mismatch fails the release before publish.

PyPI versions cannot be overwritten. If `0.1.0` is bad, yank it and ship `0.1.1`.

## After a release

```bash
python -m pip install cthreads
```

Windows 11 with Smart App Control on may still fail to **load** `cthreads_kernels.dll` (`LoadLibrary` 4551). That is a Windows policy, not a missing wheel. See [install.md](./install.md).

## Workflow files

| File | Trigger | Publishes? |
|------|---------|------------|
| `.github/workflows/ci.yml` | PR and push to `main` | No |
| `.github/workflows/release.yml` | GitHub Release published | PyPI |
| `.github/workflows/release.yml` | Actions → Run workflow | TestPyPI only |
