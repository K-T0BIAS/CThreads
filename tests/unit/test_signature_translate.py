"""Unit tests for signature + full function translate."""

from pathlib import Path

import pytest

from cthreads.compiler.translation import (
    Signature,
    Source,
    TranslationResult,
    translate_function,
)
from cthreads.compiler.translation.context import TranslationContext
from cthreads.frontend.Registry import REGISTRY
from cthreads.compiler.orchestrator.units import Handle, ThreadableUnit
from helpers import make_threadable_type


def test_signature_params_and_void_return():
    def move(a: int, b: float) -> None:
        pass

    ctx = TranslationContext(fn=move, this_file=Path("x.hpp"))
    parts = Signature.translate(Source.parse_function(move), ctx)
    assert parts.return_type is None
    assert parts.params_csv == "int a, double b"
    assert ctx.symbols["a"].cpp_name == "int"


def test_signature_mutables_pass_by_ref():
    def mut(xs: list[int], d: dict[str, int]) -> None:
        pass

    ctx = TranslationContext(fn=mut, this_file=Path("x.hpp"))
    parts = Signature.translate(Source.parse_function(mut), ctx)
    assert parts.params_csv == (
        "std::vector<int>& xs, "
        "std::unordered_map<std::string, int>& d"
    )


def test_signature_missing_annotation():
    def f(a):
        pass

    f.__annotations__.clear()
    ctx = TranslationContext(fn=f, this_file=Path("x.hpp"))
    with pytest.raises(TypeError, match="needs a type annotation"):
        Signature.translate(Source.parse_function(f), ctx)


def test_signature_rejects_varargs():
    def f(*xs: int) -> None:
        pass

    ctx = TranslationContext(fn=f, this_file=Path("x.hpp"))
    with pytest.raises(TypeError, match="not supported"):
        Signature.translate(Source.parse_function(f), ctx)


def test_signature_method_drops_self():
    Particle = make_threadable_type("Particle")
    REGISTRY.register_threadable(Particle)
    unit = ThreadableUnit(
        handle=Handle(name="Particle", path="p.py", target=Particle),
        fields={},
        hpp_path=Path("__Threadable__/Particle.hpp"),
        cpp_path=Path("__Threadable__/Particle.cpp"),
    )
    REGISTRY.threadable_units["Particle"] = unit

    def step(self, dt: float) -> None:
        pass

    step.__qualname__ = "Particle.step"
    ctx = TranslationContext(
        fn=step, this_file=unit.hpp_path, owner=unit
    )
    parts = Signature.translate(Source.parse_function(step), ctx)
    assert parts.params_csv == "double dt"
    assert "self" in ctx.symbols


def test_translate_function_body_and_result_helpers():
    def move(a: int) -> int:
        b: int = a + 1
        return b

    result = translate_function(move, this_file=Path("move.hpp"))
    assert isinstance(result, TranslationResult)
    assert result.return_type is not None
    assert result.return_type.cpp_name == "int"
    assert "int b = (a + 1);" in result.body
    assert "return b;" in result.body
    assert result.free_signature().startswith("CTHREADS_API int move(")
    assert "Particle::move" in result.method_def_signature("Particle")
    assert result.method_decl().strip().endswith(");")
