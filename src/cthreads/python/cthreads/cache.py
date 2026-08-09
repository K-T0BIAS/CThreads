"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Content-hash cache + write-if-changed helpers for smart compile/build.
"""

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from .CONFIG import VERSION

CACHE_FILENAME = ".cthreads_cache.json"

# Managed block written into the user project root `.gitignore` on compile/build.
_GITIGNORE_BEGIN = "# >>> cthreads (auto)"
_GITIGNORE_END = "# <<< cthreads (auto)"
_GITIGNORE_PATTERNS = (
    "__Thread__/",
    "__Threadable__/",
    ".cthreads_cache.json",
    "cthreads_kernels.dll",
    "cthreads_kernels.so",
    "libcthreads_kernels.so",
    "libcthreads_kernels.dylib",
)


def sha256_text(*parts: str) -> str:
    """
    Hash the text parts.
    """
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def source_fingerprint(*objs: Any) -> str:
    """Hash VERSION + getsource() for each object (class / function)."""
    chunks = [VERSION]
    for obj in objs:
        try:
            chunks.append(inspect.getsource(obj)) # collect the src code of the object
        except (OSError, TypeError):
            chunks.append(repr(obj))
        chunks.append(getattr(obj, "__qualname__", getattr(obj, "__name__", "")))
    return sha256_text(*chunks)


def write_if_changed(path: Path, content: str) -> bool:
    """
    Write content to path only if missing or different.
    Returns True if the file was written (changed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            if path.read_text(encoding="utf-8") == content:
                return False
        except OSError:
            pass
    path.write_text(content, encoding="utf-8")
    return True


def cache_path_for_root(root: Path) -> Path:
    return root / CACHE_FILENAME


def load_cache(root: Path) -> dict[str, Any]:
    path = cache_path_for_root(root)
    if not path.is_file():
        return {"version": VERSION, "units": {}, "link_hash": None, "binary": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": VERSION, "units": {}, "link_hash": None, "binary": None}
    if data.get("version") != VERSION:
        return {"version": VERSION, "units": {}, "link_hash": None, "binary": None}
    data.setdefault("units", {})
    return data


def save_cache(root: Path, data: dict[str, Any]) -> None:
    data = dict(data)
    data["version"] = VERSION
    path = cache_path_for_root(root)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hash_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    h.update(VERSION.encode("utf-8"))
    for path in sorted(paths, key=lambda p: str(p).lower()):
        h.update(str(path.name).encode("utf-8"))
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def ensure_gitignore(root: Path) -> bool:
    """
    Create or update ``root/.gitignore`` so codegen / link artifacts stay untracked.

    Idempotent: inserts a managed ``>>> cthreads (auto)`` block, or refreshes it
    if the patterns changed. Returns True if the file was created or modified.
    """
    root = Path(root)
    path = root / ".gitignore"
    block = (
        _GITIGNORE_BEGIN
        + "\n"
        + "\n".join(_GITIGNORE_PATTERNS)
        + "\n"
        + _GITIGNORE_END
        + "\n"
    )

    existing = ""
    if path.is_file():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError:
            existing = ""

    begin = existing.find(_GITIGNORE_BEGIN)
    end = existing.find(_GITIGNORE_END)
    if begin != -1 and end != -1 and end > begin:
        end_line = end + len(_GITIGNORE_END)
        if end_line < len(existing) and existing[end_line] == "\n":
            end_line += 1
        updated = existing[:begin] + block + existing[end_line:]
    elif existing.strip():
        sep = "" if existing.endswith("\n") else "\n"
        updated = existing + sep + "\n" + block
    else:
        updated = block

    if updated == existing:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")
    return True
