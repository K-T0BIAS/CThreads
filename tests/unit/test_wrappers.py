"""Unit tests for @Thread / @Threadable wrappers."""

import pytest

from cthreads import CONFIG
from cthreads.Thread.wrapper import Thread
from cthreads.Threadable.wrapper import Threadable


def test_thread_registers_and_marks():
    @Thread
    def add(a: int, b: int) -> int:
        return a + b

    assert add.__threaded is True
    assert add.__thread_version == CONFIG.VERSION
    assert CONFIG.REGISTRY.threads[add.__qualname__] is add


def test_thread_rejects_bad_type():
    class Bad:
        pass

    with pytest.raises(TypeError, match="invalid type"):

        @Thread
        def f(x: Bad) -> None:
            pass


def test_threadable_registers_fields_and_methods():
    @Threadable
    class Particle:
        x: float
        y: float

        @Thread
        def step(self, dt: float) -> None:
            self.x += dt

    assert Particle.__threadable is True
    assert CONFIG.REGISTRY.threadables["Particle"] is Particle
    assert Particle.step.__threaded is True


def test_threadable_rejects_bad_field():
    class Bad:
        pass

    with pytest.raises(TypeError, match="invalid type"):

        @Threadable
        class Box:
            item: Bad
