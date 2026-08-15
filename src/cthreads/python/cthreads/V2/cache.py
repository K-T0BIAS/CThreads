"""Content-hash cache + gitignore helpers for V2 compile/build."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from .frontend.Registry import REGISTRY


def sha256_text(*parts: str) -> str:
    """Hash the text parts."""
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def source_fingerprint(*objs: Any) -> str:
    """Hash VERSION + getsource() for each object (class / function)."""
    chunks = [REGISTRY.VERSION]
    for obj in objs:
        try:
            chunks.append(inspect.getsource(obj))
        except (OSError, TypeError):
            chunks.append(repr(obj))
        chunks.append(
            getattr(obj, "__qualname__", getattr(obj, "__name__", ""))
        )
    return sha256_text(*chunks)

CACHE_FILENAME = ".cthreads_cache.json"

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


def write_if_changed(path: Path, content: str) -> bool:
    """Write content to path only if missing or different."""
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
    version = REGISTRY.VERSION
    if not path.is_file():
        return {"version": version, "units": {}, "link_hash": None, "binary": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": version, "units": {}, "link_hash": None, "binary": None}
    if data.get("version") != version:
        return {"version": version, "units": {}, "link_hash": None, "binary": None}
    data.setdefault("units", {})
    return data


def save_cache(root: Path, data: dict[str, Any]) -> None:
    data = dict(data)
    data["version"] = REGISTRY.VERSION
    path = cache_path_for_root(root)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hash_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    h.update(REGISTRY.VERSION.encode("utf-8"))
    for path in sorted(paths, key=lambda p: str(p).lower()):
        h.update(str(path.name).encode("utf-8"))
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def ensure_gitignore(root: Path) -> bool:
    """Insert/refresh the managed cthreads block in ``root/.gitignore``."""
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
