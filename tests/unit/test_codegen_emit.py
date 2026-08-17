"""Unit tests for free-thread / Threadable C++ codegen paths."""

from pathlib import Path

from cthreads import Thread, Threadable, compile
from cthreads.cache import source_fingerprint
from cthreads.compiler.translation import translate_function
from cthreads.frontend.Registry import REGISTRY
from cthreads.kernel_meta import KERNELS


def test_translate_function_on_thread():
    @Thread
    def ok(a: int) -> int:
        return a

    result = translate_function(ok, this_file=Path("ok.hpp"))
    assert "ok" in result.free_signature()


def test_compile_free_thread_emits_hpp_cpp_and_trampolines(tmp_module):
    mod = tmp_module(
        """
        from cthreads import Thread

        @Thread
        def add(a: int, b: int) -> int:
            return a + b
        """
    )
    info = compile(force=True)
    assert "add" in info["rewritten"]
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
    assert REGISTRY.thread_units[mod.add.__qualname__].hpp_path == hpp
    assert "add" in KERNELS

    info2 = compile(force=False)
    assert info2["rewritten"] == []


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
    info = compile(force=True)
    assert "Particle" in info["rewritten"]
    root = Path(mod.__file__).parent
    hpp = (root / "__Threadable__" / "Particle.hpp").read_text(encoding="utf-8")
    cpp = (root / "__Threadable__" / "Particle.cpp").read_text(encoding="utf-8")
    assert "struct Particle {" in hpp
    assert "double x{};" in hpp
    assert "Particle() = default;" in hpp
    assert "void step(double dt);" in hpp
    assert "Particle_step" in hpp
    assert "this->x" in cpp
    assert "self->step(" in cpp
    assert "Particle_step" in KERNELS


def test_compile_free_thread_calls_threadable_method(tmp_module):
    mod = tmp_module(
        """
        from cthreads import Threadable, Thread

        @Threadable
        class Particle:
            x: float

            @Thread
            def step(self, dt: float) -> None:
                self.x += dt

        @Thread
        def run(ps: list[Particle], n: int, dt: float) -> None:
            for i in range(n):
                ps[i].step(dt)
        """
    )
    info = compile(force=True)
    assert "run" in info["rewritten"]
    cpp = (Path(mod.__file__).parent / "__Thread__" / "run.cpp").read_text(
        encoding="utf-8"
    )
    assert ".step(dt)" in cpp


def test_source_fingerprint_used_in_cache(tmp_module):
    mod = tmp_module(
        """
        from cthreads import Thread

        @Thread
        def add(a: int, b: int) -> int:
            return a + b
        """
    )
    compile(force=True)
    fp = source_fingerprint(mod.add)
    assert isinstance(fp, str) and len(fp) == 64
