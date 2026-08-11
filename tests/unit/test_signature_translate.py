"""Unit tests for signature + full function translate."""

import ast
import textwrap

import pytest

from cthreads.CONFIG import STORE, VERSION
from cthreads.pyTypes import PyInt
from cthreads.Thread.compile.AstTranslators.signature import translate_signature
from cthreads.Thread.compile.AstTranslators.translate import (
    TranslateResult,
    translate_function,
)
from helpers import make_ctx


def _func_def(src: str) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(src))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef))


def test_signature_params_and_void_return():
    ctx = make_ctx("move", hints={"p": int, "dt": float, "return": None})
    # use float/int properly via fake threadable? int is fine
    ctx.hints = {"a": int, "b": float, "return": type(None)}
    fd = _func_def(
        """
        def move(a: int, b: float) -> None:
            pass
        """
    )
    parts = translate_signature(fd, ctx)
    assert parts.return_type == "void"
    assert parts.params_csv == "int a, double b"
    assert ctx.symbols["a"].cpp_name == "int"


def test_signature_mutables_pass_by_ref():
    ctx = make_ctx(
        "mut",
        hints={"xs": list[int], "d": dict[str, int], "return": type(None)},
    )
    fd = _func_def(
        """
        def mut(xs: list[int], d: dict[str, int]) -> None:
            pass
        """
    )
    parts = translate_signature(fd, ctx)
    assert parts.params_csv == (
        "std::vector<int>& xs, "
        "std::unordered_map<std::string, int>& d"
    )


def test_signature_missing_annotation():
    ctx = make_ctx("f", hints={})
    fd = _func_def("def f(a):\n    pass")
    with pytest.raises(TypeError, match="needs a type annotation"):
        translate_signature(fd, ctx)


def test_signature_rejects_varargs():
    ctx = make_ctx("f", hints={"xs": int, "return": None})
    fd = _func_def("def f(*xs: int) -> None:\n    pass")
    with pytest.raises(TypeError, match="not supported"):
        translate_signature(fd, ctx)


def test_signature_method_drops_self():
    STORE["Particle"] = "__Threadable__/Particle.hpp"
    ctx = make_ctx("step", owner_name="Particle", hints={"dt": float, "return": None})
    fd = _func_def(
        """
        def step(self, dt: float) -> None:
            pass
        """
    )
    parts = translate_signature(fd, ctx)
    assert parts.params_csv == "double dt"
    assert "self" in ctx.symbols


def test_translate_function_body_and_result_helpers():
    def move(a: int) -> int:
        b: int = a + 1
        return b

    fd = _func_def(
        """
        def move(a: int) -> int:
            b: int = a + 1
            return b
        """
    )
    result = translate_function(move, fd, {"a": int, "return": int})
    assert isinstance(result, TranslateResult)
    assert result.return_type == "int"
    assert "int b = (a + 1);" in result.body
    assert "return b;" in result.body
    assert result.free_signature().startswith("CTHREADS_API int move(")
    assert "Particle::move" in result.method_def_signature("Particle")
    assert result.method_decl().strip().endswith(");")
