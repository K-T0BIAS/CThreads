"""Unit tests for cthreads.cache."""

import json
from pathlib import Path

from cthreads import CONFIG
from cthreads.cache import (
    CACHE_FILENAME,
    cache_path_for_root,
    hash_files,
    load_cache,
    save_cache,
    sha256_text,
    source_fingerprint,
    write_if_changed,
)


def test_sha256_text_stable_and_distinct():
    a = sha256_text("hello")
    b = sha256_text("hello")
    c = sha256_text("hello", "world")
    assert a == b
    assert a != c
    assert len(a) == 64


def test_source_fingerprint_includes_version(monkeypatch):
    def f():
        return 1

    h1 = source_fingerprint(f)
    monkeypatch.setattr(CONFIG, "VERSION", CONFIG.VERSION + "-x")
    # cache module imported VERSION at load — fingerprint uses CONFIG.VERSION via import
    from cthreads import cache as cache_mod

    monkeypatch.setattr(cache_mod, "VERSION", "different")
    h2 = source_fingerprint(f)
    assert h1 != h2


def test_write_if_changed_writes_then_skips(tmp_path: Path):
    path = tmp_path / "a.txt"
    assert write_if_changed(path, "one") is True
    assert path.read_text(encoding="utf-8") == "one"
    assert write_if_changed(path, "one") is False
    assert write_if_changed(path, "two") is True
    assert path.read_text(encoding="utf-8") == "two"


def test_load_cache_missing_and_corrupt(tmp_path: Path):
    empty = load_cache(tmp_path)
    assert empty["units"] == {}
    assert empty["link_hash"] is None

    bad = tmp_path / CACHE_FILENAME
    bad.write_text("{not json", encoding="utf-8")
    recovered = load_cache(tmp_path)
    assert recovered["units"] == {}


def test_load_cache_wrong_version(tmp_path: Path):
    path = cache_path_for_root(tmp_path)
    path.write_text(
        json.dumps({"version": "0.0.0", "units": {"x": {}}}),
        encoding="utf-8",
    )
    data = load_cache(tmp_path)
    assert data["units"] == {}


def test_save_and_load_cache_roundtrip(tmp_path: Path):
    save_cache(tmp_path, {"units": {"move": {"src_hash": "abc"}}, "link_hash": "L"})
    data = load_cache(tmp_path)
    assert data["units"]["move"]["src_hash"] == "abc"
    assert data["link_hash"] == "L"
    assert data["version"] == CONFIG.VERSION


def test_hash_files_order_independent_by_name(tmp_path: Path):
    a = tmp_path / "a.cpp"
    b = tmp_path / "b.cpp"
    a.write_text("A", encoding="utf-8")
    b.write_text("B", encoding="utf-8")
    assert hash_files([a, b]) == hash_files([b, a])
    b.write_text("C", encoding="utf-8")
    assert hash_files([a, b]) != hash_files([a])
