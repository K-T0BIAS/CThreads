"""Unit tests for cthreads.CONFIG."""

from cthreads.CONFIG import KERNELS, REGISTRY, STORE, VERSION, _Registry


def test_version_is_string():
    assert isinstance(VERSION, str)
    assert VERSION


def test_registry_register_threadable_and_thread():
    reg = _Registry()

    class T:
        pass

    def f():
        pass

    f.__qualname__ = "f"
    reg.register_threadable(T)
    reg.register_thread(f)
    assert reg.threadables["T"] is T
    assert reg.threads["f"] is f


def test_registry_overwrite_is_idempotent():
    reg = _Registry()

    class T:
        pass

    class T2:
        __name__ = "T"

    # simulate same name overwrite
    T2.__name__ = "T"
    reg.register_threadable(T)
    reg.register_threadable(T2)
    assert reg.threadables["T"] is T2


def test_registry_clear():
    REGISTRY.register_threadable(type("A", (), {}))
    STORE["A"] = "x"
    KERNELS["a"] = object()
    REGISTRY.clear()
    assert REGISTRY.threadables == {}
    assert REGISTRY.threads == {}
