"""Marshal promote/demote path for SharedHost-backed params."""

import ctypes
from unittest.mock import patch

import pytest

import cthreads.marshal as marshal


def test_promote_shared_calls_kernel_symbols(monkeypatch):
    captured: list[str] = []

    class FakeFn:
        pass

    promote0 = FakeFn()
    promote2 = FakeFn()

    class FakeLib:
        pass

    lib = FakeLib()

    def fake_fn(_lib, name):
        captured.append(name)
        if name == "worker__promote_a0_shared":
            return promote0
        if name == "worker__promote_a2_shared":
            return promote2
        raise AssertionError(name)

    def fake_call(fn, restype, argtypes, *args):
        assert fn in (promote0, promote2)
        assert argtypes == [ctypes.c_void_p, ctypes.c_void_p]
        assert args[0].value == 0x1000
        assert args[1].value == 0x2000

    monkeypatch.setattr(marshal, "_lib", lambda: lib)
    monkeypatch.setattr(marshal, "_pack_c", lambda p: ctypes.c_void_p(p))
    monkeypatch.setattr(marshal, "_fn", fake_fn)
    monkeypatch.setattr(marshal, "_call", fake_call)

    params = [
        {"name": "x", "pass_as": "shared", "schema": {"kind": "int"}},
        {"name": "y", "pass_as": "value", "schema": {"kind": "int"}},
        {
            "name": "head",
            "pass_as": "shared",
            "schema": {"kind": "list", "inner": {"kind": "int"}},
        },
    ]
    marshal.promote_shared_to_host("worker", params, 0x1000, 0x2000)
    assert captured == ["worker__promote_a0_shared", "worker__promote_a2_shared"]


def test_promote_shared_noop_when_host_null():
    marshal.promote_shared_to_host(
        "worker",
        [{"name": "x", "pass_as": "shared", "schema": {"kind": "int"}}],
        0x1000,
        0,
    )


def test_demote_shared_calls_kernel_symbols(monkeypatch):
    captured: list[str] = []

    class FakeFn:
        pass

    demote0 = FakeFn()
    demote_ret = FakeFn()

    class FakeLib:
        pass

    lib = FakeLib()

    def fake_fn(_lib, name):
        captured.append(name)
        if name == "step__demote_a0_shared":
            return demote0
        if name == "step__demote_return_shared":
            return demote_ret
        raise AssertionError(name)

    def fake_call(fn, restype, argtypes, *args):
        assert fn in (demote0, demote_ret)
        assert argtypes == [ctypes.c_void_p, ctypes.c_void_p]

    monkeypatch.setattr(marshal, "_lib", lambda: lib)
    monkeypatch.setattr(marshal, "_pack_c", lambda p: ctypes.c_void_p(p))
    monkeypatch.setattr(marshal, "_fn", fake_fn)
    monkeypatch.setattr(marshal, "_call", fake_call)

    params = [
        {
            "name": "head",
            "pass_as": "shared",
            "schema": {"kind": "list", "inner": {"kind": "int"}},
        },
    ]
    meta = {"return_pass_as": "shared", "symbol": "step", "params": params}
    marshal.demote_shared_from_host("step", params, 0x1000, 0x2000, meta)
    assert captured == ["step__demote_a0_shared", "step__demote_return_shared"]


def test_demote_shared_noop_when_host_null():
    marshal.demote_shared_from_host(
        "step",
        [{"name": "h", "pass_as": "shared", "schema": {"kind": "int"}}],
        0x1000,
        0,
    )


def test_writeback_job_state_demotes_then_writebacks(monkeypatch):
    captured: list[str] = []
    head = [1, 2]

    class FakeFn:
        pass

    demote0 = FakeFn()
    demote_ret = FakeFn()

    def fake_fn(_lib, name):
        captured.append(name)
        if name == "step__demote_a0_shared":
            return demote0
        if name == "step__demote_return_shared":
            return demote_ret
        raise AssertionError(name)

    def fake_call(fn, restype, argtypes, *args):
        captured.append("call")
        assert fn in (demote0, demote_ret)

    def fake_unpack(lib, symbol, prefix, schema, path, pack, **kw):
        captured.append(f"unpack:{prefix}")
        assert kw.get("into") is head

    monkeypatch.setattr(marshal, "_lib", lambda: object())
    monkeypatch.setattr(marshal, "_pack_c", lambda p: ctypes.c_void_p(p))
    monkeypatch.setattr(marshal, "_fn", fake_fn)
    monkeypatch.setattr(marshal, "_call", fake_call)
    monkeypatch.setattr(marshal, "unpack_value", fake_unpack)

    marshal.writeback_job_state(
        "step",
        [
            {
                "name": "head",
                "pass_as": "shared",
                "schema": {"kind": "list", "inner": {"kind": "int"}},
            }
        ],
        [head],
        0x1000,
        0xBEEF,
        meta={"return_pass_as": "shared", "symbol": "step", "params": []},
    )
    assert captured == [
        "step__demote_a0_shared",
        "call",
        "unpack:a0",
        "step__demote_return_shared",
        "call",
    ]
    assert marshal._wb_table == {}


def test_writeback_job_state_uses_original_param_index(monkeypatch):
    captured: list[str] = []
    head = [0, 0]

    def fake_fn(_lib, name):
        captured.append(name)
        return object()

    def fake_unpack(lib, symbol, prefix, schema, path, pack, **kw):
        captured.append(f"unpack:{prefix}")

    monkeypatch.setattr(marshal, "_lib", lambda: object())
    monkeypatch.setattr(marshal, "_pack_c", lambda p: ctypes.c_void_p(p))
    monkeypatch.setattr(marshal, "_fn", fake_fn)
    monkeypatch.setattr(marshal, "_call", lambda *a, **k: None)
    monkeypatch.setattr(marshal, "unpack_value", fake_unpack)

    marshal.writeback_job_state(
        "step",
        [
            {"name": "n", "pass_as": "value", "schema": {"kind": "int"}},
            {
                "name": "head",
                "pass_as": "shared",
                "schema": {"kind": "list", "inner": {"kind": "int"}},
            },
        ],
        [3, head],
        0x1000,
        0xBEEF,
    )
    assert captured == ["step__demote_a1_shared", "unpack:a1"]


def test_writeback_job_state_skips_scalars_and_threadable_dicts(monkeypatch):
    monkeypatch.setattr(marshal, "_lib", lambda: object())
    monkeypatch.setattr(marshal, "_pack_c", lambda p: ctypes.c_void_p(p))
    monkeypatch.setattr(
        marshal,
        "_fn",
        lambda *a, **k: pytest.fail("demote should not run"),
    )
    monkeypatch.setattr(
        marshal,
        "unpack_value",
        lambda *a, **k: pytest.fail("unpack should not run"),
    )

    marshal.writeback_job_state(
        "step",
        [
            {"name": "n", "pass_as": "value", "schema": {"kind": "int"}},
            {
                "name": "c",
                "pass_as": "ref",
                "schema": {"kind": "threadable", "type_name": "Counter"},
            },
        ],
        [1, {"n": 0}],
        0x1000,
        0,
    )


def test_unpack_return_demotes_shared_return_before_read(monkeypatch):
    order: list[str] = []

    def fake_demote(symbol, params, pack_ptr, host_ptr, meta=None):
        order.append("demote")
        assert meta and meta.get("return_pass_as") == "shared"

    def fake_unpack(lib, symbol, prefix, schema, path, pack, **kw):
        order.append("unpack")
        return 99

    monkeypatch.setattr(marshal, "demote_shared_from_host", fake_demote)
    monkeypatch.setattr(marshal, "_lib", lambda: object())
    monkeypatch.setattr(marshal, "_pack_c", lambda p: ctypes.c_void_p(p))
    monkeypatch.setattr(marshal, "unpack_value", fake_unpack)

    meta = {
        "symbol": "pick",
        "return_pass_as": "shared",
        "params": [],
        "return_schema": {"kind": "int", "cpp_type": "int"},
    }
    out = marshal.unpack_return(meta, 0x1000, 0xBEEF)
    assert out == 99
    assert order == ["demote", "unpack"]


def test_unpack_return_plain_return_skips_demote(monkeypatch):
    monkeypatch.setattr(
        marshal,
        "demote_shared_from_host",
        lambda *a, **k: pytest.fail("demote should not run"),
    )
    monkeypatch.setattr(marshal, "_lib", lambda: object())
    monkeypatch.setattr(marshal, "_pack_c", lambda p: ctypes.c_void_p(p))
    monkeypatch.setattr(marshal, "unpack_value", lambda *a, **k: 7)

    meta = {
        "symbol": "add",
        "params": [],
        "return_schema": {"kind": "int", "cpp_type": "int"},
    }
    assert marshal.unpack_return(meta, 0x1000, 0xBEEF) == 7


def test_writeback_params_still_skips_tbuffer(monkeypatch):
    monkeypatch.setattr(marshal, "_lib", lambda: object())
    monkeypatch.setattr(marshal, "_pack_c", lambda p: ctypes.c_void_p(p))
    with patch.object(marshal, "unpack_value") as mock_unpack:
        marshal.writeback_params(
            "step",
            [
                {
                    "kind": "tbuffer",
                    "pass_as": "tbuffer",
                    "schema": {"kind": "tbuffer", "cpp_type": "x"},
                }
            ],
            [object()],
            0x1000,
        )
        mock_unpack.assert_not_called()


def test_pack_params_still_packs_shared_staging_slot(monkeypatch):
    """Shared params are staged into a{i} before promote moves them to SharedHost."""
    calls: list[str] = []

    def fake_pack_value(lib, symbol, prefix, schema, value, path, pack, **kw):
        calls.append(prefix)

    monkeypatch.setattr(marshal, "_lib", lambda: object())
    monkeypatch.setattr(marshal, "_pack_c", lambda p: ctypes.c_void_p(p))
    monkeypatch.setattr(marshal, "pack_value", fake_pack_value)

    params = [
        {"name": "h", "pass_as": "shared", "schema": {"kind": "int"}},
        {"name": "n", "pass_as": "value", "schema": {"kind": "int"}},
    ]
    marshal.pack_params("w", params, [1, 2], 0x1000)
    assert calls == ["a0", "a1"]
