"""Marshal pack-pointer threading / no global state."""

import ctypes
import threading

import cthreads.marshal as marshal


def test_no_global_pack_slot():
    assert not hasattr(marshal, "_PACK")
    assert not hasattr(marshal, "path_pack")


def test_pack_c_rejects_null():
    try:
        marshal._pack_c(0)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "null pack" in str(e)


def test_pack_c_accepts_int_and_c_void_p():
    p = marshal._pack_c(0xDEAD)
    assert isinstance(p, ctypes.c_void_p)
    assert p.value == 0xDEAD
    p2 = marshal._pack_c(ctypes.c_void_p(0xBEEF))
    assert p2.value == 0xBEEF


def test_concurrent_pack_c_independent():
    """Simulates concurrent Jobs each holding their own pack pointer."""
    results: list[int] = []
    lock = threading.Lock()

    def worker(addr: int) -> None:
        pack = marshal._pack_c(addr)
        # ctypes may release the GIL on real DLL calls; yield to amplify races
        # if a global were still in use.
        for _ in range(200):
            assert pack.value == addr
            threading.Event().wait(0)
        with lock:
            results.append(int(pack.value))

    threads = [
        threading.Thread(target=worker, args=(0x1000 + i,))
        for i in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [0x1000 + i for i in range(16)]


def test_lib_caches_single_cdll(monkeypatch):
    created: list[str] = []

    class FakeDLL:
        def __init__(self, path: str):
            created.append(path)

    class FakeExt:
        @staticmethod
        def kernel_path():
            return r"C:\fake\kernels.dll"

    import cthreads

    monkeypatch.setattr(marshal.ctypes, "CDLL", FakeDLL)
    monkeypatch.setattr(cthreads, "_ext", FakeExt)
    monkeypatch.setattr(marshal, "_cached_lib", None)
    monkeypatch.setattr(marshal, "_cached_path", None)

    a = marshal._lib()
    b = marshal._lib()
    assert a is b
    assert created == [r"C:\fake\kernels.dll"]


def test_call_does_not_mutate_shared_argtypes():
    """_call must not touch fn.argtypes (races under concurrent Jobs)."""
    try:
        libc = ctypes.CDLL("msvcrt")
    except OSError:
        libc = ctypes.CDLL(None)
    strlen = libc.strlen
    strlen.argtypes = None
    strlen.restype = None
    before_at = strlen.argtypes
    before_rt = strlen.restype
    n = marshal._call(
        strlen,
        ctypes.c_size_t,
        [ctypes.c_char_p],
        b"abcd",
    )
    assert n == 4
    assert strlen.argtypes is before_at
    assert strlen.restype is before_rt
