"""Unit tests for @Thread / @Threadable wrappers."""

import pytest

from cthreads import Thread, Threadable
from cthreads.frontend.Registry import REGISTRY


def test_thread_registers_and_marks():
    @Thread
    def add(a: int, b: int) -> int:
        return a + b

    assert add.__threaded is True
    assert add.__thread_version == REGISTRY.VERSION
    assert REGISTRY.threads[add.__qualname__] is add


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
    assert REGISTRY.threadables["Particle"] is Particle
    assert Particle.step.__threaded is True


def test_threadable_default_init_zeros_fields():
    @Threadable
    class Vec2:
        x: float
        y: float

    @Threadable
    class Body:
        pos: Vec2
        n: int
        tags: list[str]
        ok: bool
        name: str

    v = Vec2()
    assert v.x == 0.0 and v.y == 0.0
    b = Body()
    assert b.n == 0
    assert b.ok is False
    assert b.name == ""
    assert b.tags == []
    assert isinstance(b.pos, Vec2)
    assert b.pos.x == 0.0


def test_threadable_rejects_user_init():
    with pytest.raises(TypeError, match="must not define __init__"):

        @Threadable
        class Bad:
            x: float

            def __init__(self, x: float):
                self.x = x


def test_threadable_rejects_bad_field():
    class Bad:
        pass

    with pytest.raises(TypeError, match="invalid type"):

        @Threadable
        class Box:
            item: Bad
