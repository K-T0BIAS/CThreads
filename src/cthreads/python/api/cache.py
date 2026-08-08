"""
Content-hash cache + write-if-changed helpers for smart compile/build.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from .CONFIG import VERSION

CACHE_FILENAME = ".cthreads_cache.json"


def sha256_text(*parts: str) -> str:
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
            chunks.append(inspect.getsource(obj))
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
