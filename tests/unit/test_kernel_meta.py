"""Unit tests for kernel_meta schema + trampoline emission."""

import pytest

from cthreads.CONFIG import KERNELS
from cthreads.kernel_meta import (
    KernelMeta,
    ParamMeta,
    TypeSchema,
    build_kernel_meta,
    emit_trampoline_cpp,
    emit_trampoline_decls,
)
from cthreads.Thread.wrapper import Thread
from cthreads.Threadable.wrapper import Threadable


def _prim(kind: str) -> TypeSchema:
    cpp = {"int": "int", "float": "double", "bool": "bool", "str": "std::string"}[kind]
    return TypeSchema(kind, cpp)


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
    assert add.__kernel_meta__["params"][0]["schema"]["kind"] == "int"


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
    assert {n for n, _ in free.params[0].schema.fields} == {"x", "y", "velocity"}

    method = build_kernel_meta(
        Particle.step, symbol="Particle_step", owner_name="Particle", owner_cls=Particle
    )
    assert method.is_method is True
    assert method.params[0].name == "self"
    assert method.params[0].pass_as == "ptr"
    assert method.return_kind == "void"


def test_build_kernel_meta_threadable_return():
    @Threadable
    class Particle:
        x: float
        y: float
        velocity: float

    @Thread
    def make_particle(x: float, y: float, v: float) -> Particle:
        raise NotImplementedError

    meta = build_kernel_meta(make_particle, symbol="make_particle")
    assert meta.return_kind == "threadable"
    assert meta.return_cpp_type == "Particle"
    assert {n for n, _ in meta.return_schema.fields} == {"x", "y", "velocity"}
    assert make_particle.__kernel_meta__["return_cls"] is Particle
    assert make_particle.__kernel_meta__["types"]["Particle"] is Particle


def test_build_kernel_meta_nested_list_dict_fields():
    @Threadable
    class Vec:
        x: float
        y: float

    @Threadable
    class Body:
        pos: Vec
        tags: list[str]
        scores: dict[str, float]
        flock: list[Vec]

        @Thread
        def step(self, dt: float) -> None:
            self.pos.x += dt

    meta = build_kernel_meta(
        Body.step, symbol="Body_step", owner_name="Body", owner_cls=Body
    )
    fields = {n: s for n, s in meta.params[0].schema.fields}
    assert fields["pos"].kind == "threadable"
    assert fields["tags"].kind == "list"
    assert fields["tags"].inner.kind == "str"
    assert fields["scores"].kind == "dict"
    assert fields["flock"].inner.kind == "threadable"


def test_hint_to_schema_list_threadable_return():
    @Threadable
    class Particle:
        x: float
        y: float

    @Thread
    def all_particles() -> list[Particle]:
        raise NotImplementedError

    meta = build_kernel_meta(all_particles, symbol="all_particles")
    assert meta.return_kind == "list"
    assert meta.return_schema.inner.kind == "threadable"
    assert meta.return_schema.inner.type_name == "Particle"


def test_build_kernel_meta_method_threadable_return():
    @Threadable
    class Particle:
        x: float
        y: float
        velocity: float

    @Thread
    def clone(self) -> Particle:
        raise NotImplementedError

    meta = build_kernel_meta(
        clone, symbol="Particle_clone", owner_name="Particle", owner_cls=Particle
    )
    assert meta.return_kind == "threadable"
    assert meta.is_method is True
    assert clone.__kernel_meta__["return_cls"] is Particle


def test_build_kernel_meta_list_str_ok():
    @Thread
    def join_xs(xs: list[str]) -> None:
        pass

    meta = build_kernel_meta(join_xs, symbol="join_xs")
    assert meta.params[0].kind == "list"
    assert meta.params[0].list_inner == "str"
    assert meta.params[0].pass_as == "ref"


def test_build_kernel_meta_list_and_dict_pass_as_ref():
    @Thread
    def mut_list(xs: list[int]) -> None:
        pass

    @Thread
    def mut_dict(d: dict[str, float]) -> None:
        pass

    assert build_kernel_meta(mut_list, symbol="mut_list").params[0].pass_as == "ref"
    assert build_kernel_meta(mut_dict, symbol="mut_dict").params[0].pass_as == "ref"


def test_build_kernel_meta_missing_owner():
    @Thread
    def step(self, dt: float) -> None:
        pass

    with pytest.raises(TypeError, match="Cannot resolve Threadable"):
        build_kernel_meta(step, symbol="X_step", owner_name="Missing")


def test_emit_trampoline_nested_accessors():
    vec = TypeSchema(
        "threadable",
        "Vec",
        type_name="Vec",
        fields=[("x", _prim("float")), ("y", _prim("float"))],
    )
    body = TypeSchema(
        "threadable",
        "Body",
        type_name="Body",
        fields=[
            ("pos", vec),
            ("tags", TypeSchema("list", "std::vector<std::string>", inner=_prim("str"))),
            (
                "meta",
                TypeSchema(
                    "dict",
                    "std::unordered_map<std::string, double>",
                    key=_prim("str"),
                    value=_prim("float"),
                ),
            ),
        ],
    )
    meta = KernelMeta(
        symbol="t",
        call_symbol="t__call",
        args_new_symbol="t__new",
        args_free_symbol="t__free",
        params=[ParamMeta("b", "ref", body)],
        return_schema=None,
    )
    cpp = emit_trampoline_cpp(meta, real_call="t")
    assert "t__set_a0_pos_x(void* p, double v)" in cpp
    assert "t__a0_tags_resize(void* p, size_t n)" in cpp
    assert "t__a0_meta_insert(void* p, const char* k0, double v)" in cpp
    assert "#include <iterator>" in cpp


def test_emit_trampoline_cpp_primitives():
    meta = KernelMeta(
        symbol="add",
        call_symbol="add__call",
        args_new_symbol="add__args_new",
        args_free_symbol="add__args_free",
        params=[
            ParamMeta("a", "value", _prim("int")),
            ParamMeta("b", "value", _prim("float")),
        ],
        return_schema=_prim("int"),
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


def test_emit_trampoline_list_return():
    particle = TypeSchema(
        "threadable",
        "Particle",
        type_name="Particle",
        fields=[("x", _prim("float")), ("y", _prim("float"))],
    )
    meta = KernelMeta(
        symbol="all",
        call_symbol="all__call",
        args_new_symbol="all__new",
        args_free_symbol="all__free",
        params=[],
        return_schema=TypeSchema(
            "list", "std::vector<Particle>", inner=particle
        ),
    )
    cpp = emit_trampoline_cpp(meta, real_call="all")
    assert "std::vector<Particle> ret;" in cpp
    assert "all__ret_resize" in cpp
    assert "all__set_ret_elem_x" in cpp
