"""Unit tests for free-thread / Threadable C++ codegen paths."""

from pathlib import Path

import pytest

from cthreads.CONFIG import KERNELS, STORE, VERSION
from cthreads.Thread.compile.compile import compile_free_thread, translate_thread
from cthreads.Thread.wrapper import Thread
from cthreads.Threadable.compile import compile_threadable
from cthreads.Threadable.wrapper import Threadable


def test_translate_thread_requires_decorator_and_version():
    def plain(a: int) -> int:
        return a

    with pytest.raises(TypeError, match="not a Thread"):
        translate_thread(plain)

    @Thread
    def ok(a: int) -> int:
        return a

    ok.__thread_version = "nope"
    with pytest.raises(TypeError, match="invalid version"):
        translate_thread(ok)


def test_compile_free_thread_emits_hpp_cpp_and_trampolines(tmp_module):
    mod = tmp_module(
        """
        from cthreads import Thread

        @Thread
        def add(a: int, b: int) -> int:
            return a + b
        """
    )
    changed = compile_free_thread(mod.add, force=True, cache={})
    assert changed is True
    root = Path(mod.__file__).parent
    hpp = root / "__Thread__" / "add.hpp"
    cpp = root / "__Thread__" / "add.cpp"
    export = root / "__Thread__" / "cthreads_export.hpp"
    assert hpp.is_file() and cpp.is_file() and export.is_file()
    hpp_txt = hpp.read_text(encoding="utf-8")
    cpp_txt = cpp.read_text(encoding="utf-8")
    assert "CTHREADS_API int add(int a, int b);" in hpp_txt
    assert "return (a + b);" in cpp_txt
    assert "add__call" in cpp_txt
    assert STORE["add"].endswith("add.hpp")
    assert "add" in KERNELS

    # second compile with same cache → no rewrite
    cache = {
        "units": {
            "add": {
                "src_hash": __import__(
                    "cthreads.cache", fromlist=["source_fingerprint"]
                ).source_fingerprint(mod.add),
                "hpp": str(hpp),
                "cpp": str(cpp),
            }
        }
    }
    assert compile_free_thread(mod.add, force=False, cache=cache) is False


def test_compile_threadable_emits_struct_and_c_wrapper(tmp_module):
    mod = tmp_module(
        """
        from cthreads import Threadable, Thread

        @Threadable
        class Particle:
            x: float
            y: float
            velocity: float

            @Thread
            def step(self, dt: float) -> None:
                self.x += self.velocity * dt
        """
    )
    methods = [mod.Particle.step]
    changed = compile_threadable(mod.Particle, methods, force=True, cache={})
    assert changed is True
    root = Path(mod.__file__).parent
    hpp = (root / "__Threadable__" / "Particle.hpp").read_text(encoding="utf-8")
    cpp = (root / "__Threadable__" / "Particle.cpp").read_text(encoding="utf-8")
    assert "struct Particle {" in hpp
    assert "double x;" in hpp
    assert "void step(double dt);" in hpp
    assert "Particle_step" in hpp
    assert "this->x +=" in cpp or "this->x" in cpp
    assert "self->step(" in cpp
    assert "Particle_step" in KERNELS


def test_compile_threadable_rejects_non_threadable():
    class Plain:
        pass

    with pytest.raises(TypeError, match="not a Threadable"):
        compile_threadable(Plain, methods=[], force=True)
