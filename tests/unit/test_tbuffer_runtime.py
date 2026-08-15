"""Codegen for threadable triple-buffer allocators in kernels.dll."""

from pathlib import Path

from cthreads import Thread, Threadable
from cthreads.compiler.orchestrator.units import Handle, ThreadableUnit
from cthreads.frontend.Registry import REGISTRY
from cthreads.kernel_meta import (
    KERNELS,
    KernelMeta,
    ParamMeta,
    TypeSchema,
    build_kernel_meta,
    collect_tbuffer_threadables,
    emit_tbuffer_runtime_files,
    write_tbuffer_runtime,
)
from cthreads.types import TBuffer


def test_collect_tbuffer_threadables_from_kernels():
    KERNELS.clear()
    particle = TypeSchema(
        "threadable",
        "Particle",
        type_name="Particle",
        fields=[("x", TypeSchema("float", "double"))],
    )
    tbuf = TypeSchema(
        "tbuffer",
        "cthreads::sync::tripple_buffer<Particle>",
        inner=particle,
    )
    KERNELS["move"] = KernelMeta(
        symbol="move",
        call_symbol="move__call",
        args_new_symbol="move__new",
        args_free_symbol="move__free",
        params=[ParamMeta("buf", "tbuffer", tbuf)],
    )
    assert collect_tbuffer_threadables() == {"Particle"}


def test_emit_tbuffer_runtime_files(tmp_path):
    thread_dir = tmp_path / "__Thread__"
    thread_dir.mkdir()
    threadable_dir = tmp_path / "__Threadable__"
    threadable_dir.mkdir()
    hpp = threadable_dir / "Particle.hpp"
    hpp.write_text("struct Particle {};\n", encoding="utf-8")

    Particle = type("Particle", (), {"__threadable": True, "__annotations__": {}})
    REGISTRY.threadable_units["Particle"] = ThreadableUnit(
        handle=Handle(name="Particle", path="dummy.py", target=Particle),
        fields={},
        hpp_path=hpp,
        cpp_path=threadable_dir / "Particle.cpp",
    )

    hpp_src, cpp = emit_tbuffer_runtime_files({"Particle"}, thread_dir)

    assert "cthreads_create_tbuffer" in hpp_src
    assert "tripple_buffer<Particle>" in cpp
    assert "../__Threadable__/Particle.hpp" in cpp.replace("\\", "/")
    assert "delete static_cast<cthreads::sync::tripple_buffer<Particle>*>" in cpp


def test_write_tbuffer_runtime_end_to_end(tmp_path):
    @Threadable
    class Particle:
        x: float

    @Thread
    def step(buf: TBuffer[Particle]) -> None:
        pass

    build_kernel_meta(step, symbol="step")
    threadable_dir = tmp_path / "__Threadable__"
    threadable_dir.mkdir(parents=True, exist_ok=True)
    hpp = threadable_dir / "Particle.hpp"
    hpp.write_text("struct Particle { double x; };\n", encoding="utf-8")
    REGISTRY.threadable_units["Particle"] = ThreadableUnit(
        handle=Handle(name="Particle", path="dummy.py", target=Particle),
        fields={},
        hpp_path=hpp,
        cpp_path=threadable_dir / "Particle.cpp",
    )

    assert write_tbuffer_runtime(tmp_path) is True
    cpp = (tmp_path / "__Thread__" / "cthreads_tbuffer.cpp").read_text(encoding="utf-8")
    assert "tripple_buffer<Particle>" in cpp

    KERNELS.clear()
    assert write_tbuffer_runtime(tmp_path) is False
    assert not (tmp_path / "__Thread__" / "cthreads_tbuffer.cpp").is_file()
