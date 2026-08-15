"""Unit tests for registry / version globals (replaces CONFIG tests)."""

from cthreads.frontend.Registry.registry import REGISTRY, Registry
from cthreads.kernel_meta import KERNELS


def test_version_is_string():
    assert isinstance(REGISTRY.VERSION, str)
    assert REGISTRY.VERSION


def test_registry_register_threadable_and_thread():
    reg = Registry()

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
    reg = Registry()

    class T:
        pass

    class T2:
        pass

    T2.__name__ = "T"
    reg.register_threadable(T)
    reg.register_threadable(T2)
    assert reg.threadables["T"] is T2


def test_registry_clear():
    REGISTRY.register_threadable(type("A", (), {}))
    KERNELS["a"] = object()
    REGISTRY.clear()
    assert REGISTRY.threadables == {}
    assert REGISTRY.threads == {}
    assert KERNELS == {}
