"""Unit tests for GPU Syntax leaf translators and index AttrPlugin."""

from __future__ import annotations

import ast

import pytest

from cthreads.compiler.translation.Source import Source
from cthreads.gpu import BlockDim, BlockIdx, GlobalIdx, GridDim, ThreadIdx
from cthreads.gpu.compiler.translation.Signature import GpuSignature
from cthreads.gpu.compiler.translation.context import GpuTranslationContext
from cthreads.gpu.compiler.translation.syntax.Syntax import GpuSyntax
from cthreads.types import PyFloat, PyInt, PyList


def _ctx_for(fn):
    ctx = GpuTranslationContext(fn=fn)
    GpuSignature.translate(Source.parse_function(fn), ctx)
    return ctx


def _expr(src: str, ctx: GpuTranslationContext) -> str:
    tree = ast.parse(src, mode="eval")
    assert isinstance(tree, ast.Expression)
    return GpuSyntax.expr(tree.body, ctx)


def _stmt(src: str, ctx: GpuTranslationContext) -> list[str]:
    tree = ast.parse(src)
    assert len(tree.body) == 1
    return GpuSyntax.stmt(tree.body[0], ctx)


def _inject_indexes(fn):
    fn.__globals__.update(
        {
            "GlobalIdx": GlobalIdx,
            "ThreadIdx": ThreadIdx,
            "BlockIdx": BlockIdx,
            "BlockDim": BlockDim,
            "GridDim": GridDim,
        }
    )


# --- Name -------------------------------------------------------------------


def test_name_scalar_rewrites_to_scalars_block():
    def k(n: int, a: float, x: list[float]) -> None:
        pass

    ctx = _ctx_for(k)
    assert _expr("n", ctx) == "scalars.n"
    assert _expr("a", ctx) == "scalars.a"


def test_name_list_stays_bare():
    def k(x: list[float]) -> None:
        pass

    ctx = _ctx_for(k)
    assert _expr("x", ctx) == "x"


def test_name_local_bare():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    ctx.symbols["i"] = PyInt()
    assert _expr("i", ctx) == "i"


def test_name_unknown_raises():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match="unknown name"):
        _expr("missing", ctx)


def test_name_self_raises():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match="self"):
        _expr("self", ctx)


# --- Index ------------------------------------------------------------------


def test_index_list_to_data():
    def k(x: list[float]) -> None:
        pass

    ctx = _ctx_for(k)
    ctx.symbols["i"] = PyInt()
    assert _expr("x[i]", ctx) == "(x.data[i])"


def test_index_nested_expr_index():
    def k(x: list[float], n: int) -> None:
        pass

    ctx = _ctx_for(k)
    assert _expr("x[n]", ctx) == "(x.data[scalars.n])"


def test_index_rejects_slice():
    def k(x: list[float]) -> None:
        pass

    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match="slice"):
        _expr("x[1:2]", ctx)


def test_index_rejects_non_list():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match="list parameters"):
        _expr("n[0]", ctx)


def test_index_literal():
    def k(x: list[float]) -> None:
        pass

    ctx = _ctx_for(k)
    assert _expr("x[0]", ctx) == "(x.data[0])"


# --- Op ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "src, needle",
    [
        ("a + b", "+"),
        ("a - b", "-"),
        ("a * b", "*"),
        ("a / b", "/"),
        ("a // b", "/"),
        ("a % b", "%"),
        ("a << b", "<<"),
        ("a >> b", ">>"),
        ("a | b", "|"),
        ("a ^ b", "^"),
        ("a & b", "&"),
    ],
)
def test_binop_table(src, needle):
    def k(a: int, b: int) -> None:
        pass

    ctx = _ctx_for(k)
    out = _expr(src, ctx)
    assert needle in out
    assert "scalars.a" in out and "scalars.b" in out


def test_binop_mul_add():
    def k(a: float, x: list[float]) -> None:
        pass

    ctx = _ctx_for(k)
    ctx.symbols["i"] = PyInt()
    out = _expr("a * x[i] + x[i]", ctx)
    assert "scalars.a" in out
    assert "x.data[i]" in out
    assert "*" in out and "+" in out


@pytest.mark.parametrize(
    "src, op",
    [
        ("a == b", "=="),
        ("a != b", "!="),
        ("a < b", "<"),
        ("a <= b", "<="),
        ("a > b", ">"),
        ("a >= b", ">="),
    ],
)
def test_compare_ops(src, op):
    def k(a: int, b: int) -> None:
        pass

    ctx = _ctx_for(k)
    assert op in _expr(src, ctx)


def test_compare_and_bool():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    ctx.symbols["i"] = PyInt()
    out = _expr("i >= n and True", ctx)
    assert ">=" in out and "&&" in out
    assert "scalars.n" in out


def test_bool_or():
    def k(a: bool, b: bool) -> None:
        pass

    ctx = _ctx_for(k)
    assert "||" in _expr("a or b", ctx)


def test_unary_not_neg():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    assert "!" in _expr("not n", ctx)
    assert "-" in _expr("-n", ctx)
    assert "+" in _expr("+n", ctx)
    assert "~" in _expr("~n", ctx)


def test_pow_rejected():
    def k(a: float) -> None:
        pass

    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match=r"\*\*|pow"):
        _expr("a ** 2", ctx)


def test_chained_compare():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    ctx.symbols["i"] = PyInt()
    out = _expr("0 <= i < n", ctx)
    assert "&&" in out


def test_unsupported_call_raises():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match="unsupported call"):
        _expr("abs(n)", ctx)


def test_float_call_rejected():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match="unsupported call"):
        _expr("float(n)", ctx)


# --- Assign -----------------------------------------------------------------


def test_assign_list_element():
    def k(a: float, x: list[float], y: list[float]) -> None:
        pass

    ctx = _ctx_for(k)
    ctx.symbols["i"] = PyInt()
    lines = _stmt("y[i] = a * x[i] + y[i]", ctx)
    assert len(lines) == 1
    assert "y.data[i]" in lines[0]
    assert "scalars.a" in lines[0]


def test_ann_assign_local():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    lines = _stmt("i: int = n", ctx)
    assert lines == ["    int i = scalars.n;"]
    assert isinstance(ctx.symbols["i"], PyInt)
    assert "i" not in ctx.scalar_params


def test_ann_assign_from_global_idx():
    def k(n: int) -> None:
        pass

    _inject_indexes(k)
    ctx = _ctx_for(k)
    lines = _stmt("i: int = GlobalIdx.x", ctx)
    assert "int i =" in lines[0]
    assert "gl_GlobalInvocationID.x" in lines[0]


def test_ann_assign_uninitialized():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    # Python allows `i: int` without value via AnnAssign value=None in AST —
    # constructing via exec-style source needs a value; use explicit None-ish
    # by building AST.
    tree = ast.parse("i: int")
    assert isinstance(tree.body[0], ast.AnnAssign)
    lines = GpuSyntax.stmt(tree.body[0], ctx)
    assert lines == ["    int i;"]


def test_ann_assign_redeclare_raises():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    _stmt("i: int = 0", ctx)
    with pytest.raises(TypeError, match="redeclaration"):
        _stmt("i: int = 1", ctx)


def test_assign_unknown_name_raises():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match="unknown name"):
        _stmt("i = n", ctx)


def test_assign_bare_list_rejected():
    def k(x: list[float]) -> None:
        pass

    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match="list parameter"):
        _stmt("x = x", ctx)


def test_assign_multi_target_rejected():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    _stmt("i: int = 0", ctx)
    _stmt("j: int = 0", ctx)
    with pytest.raises(TypeError, match="single-target"):
        _stmt("i = j = n", ctx)


def test_aug_assign():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    _stmt("i: int = 0", ctx)
    lines = _stmt("i += n", ctx)
    assert lines == ["    i += scalars.n;"]


@pytest.mark.parametrize(
    "src, op",
    [
        ("i += n", "+="),
        ("i -= n", "-="),
        ("i *= n", "*="),
        ("i %= n", "%="),
        ("i &= n", "&="),
        ("i |= n", "|="),
        ("i ^= n", "^="),
        ("i <<= n", "<<="),
        ("i >>= n", ">>="),
    ],
)
def test_aug_assign_ops(src, op):
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    _stmt("i: int = 0", ctx)
    assert op in _stmt(src, ctx)[0]


def test_aug_assign_pow_rejected():
    def k(a: float) -> None:
        pass

    ctx = _ctx_for(k)
    _stmt("t: float = a", ctx)
    with pytest.raises(TypeError, match=r"\*\*|pow"):
        _stmt("t **= 2", ctx)


def test_aug_assign_list_elem():
    def k(y: list[float]) -> None:
        pass

    ctx = _ctx_for(k)
    ctx.symbols["i"] = PyInt()
    lines = _stmt("y[i] += 1.0", ctx)
    assert "y.data[i]" in lines[0]
    assert "+=" in lines[0]


def test_ann_assign_local_list_rejected():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match="local list"):
        _stmt("xs: list[float] = []", ctx)


def test_slice_assign_rejected():
    def k(x: list[float]) -> None:
        pass

    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match="slice"):
        _stmt("x[1:2] = x[0:1]", ctx)


# --- Flow -------------------------------------------------------------------


def test_if_return():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    ctx.symbols["i"] = PyInt()
    lines = _stmt("if i >= n:\n    return", ctx)
    assert lines[0].startswith("    if ")
    assert any("return;" in L for L in lines)


def test_return_always_bare():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    assert _stmt("return", ctx) == ["    return;"]
    assert _stmt("return 1", ctx) == ["    return;"]


def test_pass_break_continue():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    assert _stmt("pass", ctx) == []
    assert _stmt("break", ctx) == ["    break;"]
    assert _stmt("continue", ctx) == ["    continue;"]


@pytest.mark.parametrize(
    "src, needles",
    [
        ("for i in range(n):\n    pass", ["for (int i = 0;", "scalars.n", "i += 1"]),
        (
            "for i in range(1, n):\n    pass",
            ["for (int i = 1;", "scalars.n", "i += 1"],
        ),
        (
            "for i in range(0, n, 2):\n    pass",
            ["for (int i = 0;", "scalars.n", "i += 2"],
        ),
    ],
)
def test_for_range_forms(src, needles):
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    lines = _stmt(src, ctx)
    joined = "\n".join(lines)
    for needle in needles:
        assert needle in joined
    assert "i" not in ctx.symbols


def test_for_range():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    lines = _stmt("for i in range(n):\n    pass", ctx)
    assert "for (int i = 0;" in lines[0]
    assert "scalars.n" in lines[0]
    assert "i" not in ctx.symbols


def test_for_list_rejected():
    def k(x: list[float]) -> None:
        pass

    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match="range"):
        _stmt("for v in x:\n    pass", ctx)


def test_for_else_rejected():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match="for/else"):
        _stmt("for i in range(n):\n    pass\nelse:\n    pass", ctx)


def test_while_else_rejected():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match="while/else"):
        _stmt("while n > 0:\n    break\nelse:\n    pass", ctx)


def test_for_rebind_rejected():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    _stmt("i: int = 0", ctx)
    with pytest.raises(TypeError, match="rebinds"):
        _stmt("for i in range(n):\n    pass", ctx)


def test_for_range_zero_args_rejected():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match="range"):
        _stmt("for i in range():\n    pass", ctx)


def test_for_range_four_args_rejected():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match="range"):
        _stmt("for i in range(0, n, 1, 2):\n    pass", ctx)


def test_while():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    lines = _stmt("while n > 0:\n    break", ctx)
    assert lines[0].startswith("    while ")


def test_if_else():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    lines = _stmt("if n > 0:\n    pass\nelse:\n    return", ctx)
    assert any("else" in L for L in lines)


def test_if_elif_lowers_as_nested_else():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    lines = _stmt(
        "if n > 0:\n    return\nelif n < 0:\n    return\nelse:\n    pass",
        ctx,
    )
    joined = "\n".join(lines)
    assert "if (" in joined
    assert "else" in joined


def test_docstring_expr_ignored():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    assert _stmt('"doc"', ctx) == []


def test_unsupported_expr_stmt_comment():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    lines = _stmt("n", ctx)
    assert lines[0].startswith("    // unsupported")


def test_unsupported_stmt_comment():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    tree = ast.parse("raise RuntimeError()")
    lines = GpuSyntax.stmt(tree.body[0], ctx)
    assert "unsupported statement" in lines[0]


# --- Index builtins / AttrPlugin --------------------------------------------


@pytest.mark.parametrize(
    "expr, glsl",
    [
        ("GlobalIdx.x", "int(gl_GlobalInvocationID.x)"),
        ("GlobalIdx.y", "int(gl_GlobalInvocationID.y)"),
        ("GlobalIdx.z", "int(gl_GlobalInvocationID.z)"),
        ("ThreadIdx.x", "int(gl_LocalInvocationID.x)"),
        ("ThreadIdx.y", "int(gl_LocalInvocationID.y)"),
        ("ThreadIdx.z", "int(gl_LocalInvocationID.z)"),
        ("BlockIdx.x", "int(gl_WorkGroupID.x)"),
        ("BlockIdx.y", "int(gl_WorkGroupID.y)"),
        ("BlockIdx.z", "int(gl_WorkGroupID.z)"),
        ("BlockDim.x", "int(gl_WorkGroupSize.x)"),
        ("BlockDim.y", "int(gl_WorkGroupSize.y)"),
        ("BlockDim.z", "int(gl_WorkGroupSize.z)"),
        ("GridDim.x", "int(gl_NumWorkGroups.x)"),
        ("GridDim.y", "int(gl_NumWorkGroups.y)"),
        ("GridDim.z", "int(gl_NumWorkGroups.z)"),
    ],
)
def test_index_builtins(expr, glsl):
    def k(n: int) -> None:
        pass

    _inject_indexes(k)
    ctx = _ctx_for(k)
    assert _expr(expr, ctx) == glsl


def test_module_qualified_global_idx():
    import cthreads.gpu as gpu_mod

    def k(n: int) -> None:
        pass

    k.__globals__["gpu"] = gpu_mod
    ctx = _ctx_for(k)
    assert _expr("gpu.GlobalIdx.x", ctx) == "int(gl_GlobalInvocationID.x)"


def test_unknown_attr_raises():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match="unsupported attribute"):
        _expr("n.imag", ctx)


def test_index_builtin_bad_axis_raises():
    def k(n: int) -> None:
        pass

    _inject_indexes(k)
    ctx = _ctx_for(k)
    with pytest.raises(TypeError, match="unsupported attribute"):
        _expr("GlobalIdx.w", ctx)


def test_literal_constants():
    def k(n: int) -> None:
        pass

    ctx = _ctx_for(k)
    assert _expr("2", ctx) == "2"
    assert _expr("True", ctx) == "true"
    assert _expr("False", ctx) == "false"
    assert _expr("1.5", ctx) == "1.5"
    assert _expr("0.0", ctx) == "0.0"


def test_nested_if_in_for():
    def k(n: int, y: list[int]) -> None:
        pass

    ctx = _ctx_for(k)
    lines = _stmt(
        "for i in range(n):\n"
        "    if i > 0:\n"
        "        y[i] = i\n",
        ctx,
    )
    joined = "\n".join(lines)
    assert "for (int i = 0;" in joined
    assert "if (" in joined
    assert "y.data[i]" in joined
