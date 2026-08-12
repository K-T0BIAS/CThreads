"""Codegen for threadable triple-buffer allocators in kernels.dll."""

from pathlib import Path

from cthreads.CONFIG import KERNELS, STORE
from cthreads.kernel_meta import (
    KernelMeta,
    ParamMeta,
    TypeSchema,
    collect_tbuffer_threadables,
    emit_tbuffer_runtime_files,
    write_tbuffer_runtime,
)
from cthreads.pyTypes import TBuffer
from cthreads.Thread.wrapper import Thread
from cthreads.Threadable.wrapper import Threadable


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
    (threadable_dir / "Particle.hpp").write_text("struct Particle {};\n", encoding="utf-8")
    STORE["Particle"] = str(threadable_dir / "Particle.hpp")

    hpp, cpp = emit_tbuffer_runtime_files({"Particle"}, thread_dir)

    assert "cthreads_create_tbuffer" in hpp
    assert "tripple_buffer<Particle>" in cpp
    assert "../__Threadable__/Particle.hpp" in cpp
    assert "delete static_cast<cthreads::sync::tripple_buffer<Particle>*>" in cpp


def test_write_tbuffer_runtime_end_to_end(tmp_path):
    @Threadable
    class Particle:
        x: float

    @Thread
    def step(buf: TBuffer[Particle]) -> None:
        pass

    from cthreads.kernel_meta import build_kernel_meta

    build_kernel_meta(step, symbol="step")
    STORE["Particle"] = str(tmp_path / "__Threadable__" / "Particle.hpp")
    (tmp_path / "__Threadable__").mkdir(parents=True, exist_ok=True)
    (tmp_path / "__Threadable__" / "Particle.hpp").write_text(
        "struct Particle { double x; };\n", encoding="utf-8"
    )

    assert write_tbuffer_runtime(tmp_path) is True
    cpp = (tmp_path / "__Thread__" / "cthreads_tbuffer.cpp").read_text(encoding="utf-8")
    assert "tripple_buffer<Particle>" in cpp

    KERNELS.clear()
    assert write_tbuffer_runtime(tmp_path) is False
    assert not (tmp_path / "__Thread__" / "cthreads_tbuffer.cpp").is_file()
