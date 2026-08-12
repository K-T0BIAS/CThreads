"""Marshal pack path for triple-buffer params."""

import ctypes
from unittest.mock import patch

import cthreads.marshal as marshal


def test_pack_tbuffer_calls_set_ptr(monkeypatch):
    captured: dict = {}

    class FakeFn:
        pass

    set_ptr = FakeFn()

    class FakeLib:
        pass

    lib = FakeLib()

    def fake_fn(lib_obj, name):
        if name == "step__set_a0_ptr":
            return set_ptr
        raise AssertionError(name)

    def fake_call(fn, restype, argtypes, *args):
        captured["fn"] = fn
        captured["args"] = args

    class FakeExt:
        @staticmethod
        def tbuffer_native_ptr(obj, type_name=None):
            del obj, type_name
            return 0xABCD

    import sys

    monkeypatch.setattr(marshal, "_fn", fake_fn)
    monkeypatch.setattr(marshal, "_call", fake_call)
    monkeypatch.setitem(sys.modules, "cthreads._ext", FakeExt)

    schema = {
        "kind": "tbuffer",
        "cpp_type": "cthreads::sync::tripple_buffer<double>",
        "inner": {"kind": "float", "cpp_type": "double"},
    }
    pack = ctypes.c_void_p(0x1000)
    marshal.pack_value(
        lib,
        "step",
        "a0",
        schema,
        object(),
        marshal._Path(),
        pack,
    )
    assert captured["fn"] is set_ptr
    assert captured["args"][0].value == 0x1000
    assert captured["args"][1].value == 0xABCD


def test_pack_tbuffer_handle_uses_ptr(monkeypatch):
    from cthreads.tbuffer_host import TBufferHandle

    captured: dict = {}

    class FakeFn:
        pass

    set_ptr = FakeFn()

    class FakeLib:
        pass

    lib = FakeLib()

    def fake_fn(lib_obj, name):
        if name == "step__set_a0_ptr":
            return set_ptr
        raise AssertionError(name)

    def fake_call(fn, restype, argtypes, *args):
        captured["args"] = args

    monkeypatch.setattr(marshal, "_fn", fake_fn)
    monkeypatch.setattr(marshal, "_call", fake_call)

    handle = TBufferHandle("Particle", 0xBEEF)
    schema = {
        "kind": "tbuffer",
        "cpp_type": "cthreads::sync::tripple_buffer<Particle>",
        "inner": {"kind": "threadable", "type_name": "Particle", "cpp_type": "Particle"},
    }
    pack = ctypes.c_void_p(0x1000)
    marshal.pack_value(
        lib,
        "step",
        "a0",
        schema,
        handle,
        marshal._Path(),
        pack,
    )
    assert captured["args"][1].value == 0xBEEF


def test_writeback_skips_tbuffer(monkeypatch):
    monkeypatch.setattr(marshal, "_lib", lambda: object())
    monkeypatch.setattr(marshal, "_pack_c", lambda p: ctypes.c_void_p(p))
    with patch.object(marshal, "unpack_value") as mock_unpack:
        marshal.writeback_params(
            "step",
            [
                {
                    "kind": "tbuffer",
                    "cpp_type": "cthreads::sync::tripple_buffer<double>",
                    "schema": {
                        "kind": "tbuffer",
                        "cpp_type": "cthreads::sync::tripple_buffer<double>",
                    },
                }
            ],
            [object()],
            0x1000,
        )
        mock_unpack.assert_not_called()
