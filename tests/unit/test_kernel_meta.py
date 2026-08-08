"""Unit tests for kernel_meta schema + trampoline emission."""

import pytest

from cthreads.CONFIG import KERNELS, REGISTRY, VERSION
from cthreads.kernel_meta import (
    KernelMeta,
    ParamMeta,
    build_kernel_meta,
    emit_trampoline_cpp,
    emit_trampoline_decls,
)
from cthreads.Thread.wrapper import Thread
from cthreads.Threadable.wrapper import Threadable


def test_build_kernel_meta_free_function():
    @Thread
    def add(a: int, b: float) -> int:
        return a

    meta = build_kernel_meta(add, symbol="add")
    assert meta.symbol == "add"
    assert meta.call_symbol == "add__call"
    assert meta.return_kind == "int"
    assert [p.kind for p in meta.params] == ["int", "float"]
    assert meta.params[0].pass_as == "value"
    assert KERNELS["add"] is meta
    assert add.__kernel_symbol__ == "add"
    assert isinstance(add.__kernel_meta__, dict)


def test_build_kernel_meta_threadable_ref_and_method():
    @Threadable
    class Particle:
        x: float
        y: float
        velocity: float

        @Thread
        def step(self, dt: float) -> None:
            self.x += dt

    @Thread
    def move(p: Particle, dt: float) -> None:
        p.x += dt

    free = build_kernel_meta(move, symbol="move")
    assert free.params[0].kind == "threadable"
    assert free.params[0].pass_as == "ref"
    assert {f.name for f in free.params[0].fields} == {"x", "y", "velocity"}

    method = build_kernel_meta(
        Particle.step, symbol="Particle_step", owner_name="Particle", owner_cls=Particle
    )
    assert method.is_method is True
    assert method.params[0].name == "self"
    assert method.params[0].pass_as == "ptr"
    assert method.return_kind == "void"


def test_build_kernel_meta_list_and_bad_list():
    @Thread
    def sum_xs(xs: list[float]) -> float:
        return 0.0

    meta = build_kernel_meta(sum_xs, symbol="sum_xs")
    assert meta.params[0].kind == "list"
    assert meta.params[0].list_inner == "float"

    @Thread
    def bad(xs: list[str]) -> None:
        pass

    with pytest.raises(TypeError, match="list element"):
        build_kernel_meta(bad, symbol="bad")


def test_build_kernel_meta_missing_owner():
    @Thread
    def step(self, dt: float) -> None:
        pass

    with pytest.raises(TypeError, match="Cannot resolve Threadable"):
        build_kernel_meta(step, symbol="X_step", owner_name="Missing")


def test_emit_trampoline_cpp_primitives():
    meta = KernelMeta(
        symbol="add",
        call_symbol="add__call",
        args_new_symbol="add__args_new",
        args_free_symbol="add__args_free",
        params=[
            ParamMeta("a", "int", "int", "value"),
            ParamMeta("b", "float", "double", "value"),
        ],
        return_kind="int",
    )
    cpp = emit_trampoline_cpp(meta, real_call="add")
    assert "struct add__args" in cpp
    assert "CTHREADS_API void* add__args_new()" in cpp
    assert "add__set_a0" in cpp
    assert "add__set_a1" in cpp
    assert "a->ret = add(a->a0, a->a1);" in cpp
    assert "add__get_ret" in cpp
    decls = emit_trampoline_decls(meta)
    assert "add__call" in decls


def test_emit_trampoline_list_includes_cstddef():
    meta = KernelMeta(
        symbol="sum_xs",
        call_symbol="sum_xs__call",
        args_new_symbol="sum_xs__args_new",
        args_free_symbol="sum_xs__args_free",
        params=[
            ParamMeta(
                "xs", "list", "std::vector<double>", "value", list_inner="float"
            )
        ],
        return_kind="void",
    )
    cpp = emit_trampoline_cpp(meta, real_call="sum_xs")
    assert cpp.startswith("#include <cstddef>")
    assert "const double* data, size_t n" in cpp
