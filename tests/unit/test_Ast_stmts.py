"""Unit tests for AST statement translation (Syntax.stmt)."""

import ast

import pytest

from cthreads.compiler.translation.syntax import Syntax
from cthreads.types import PyDict, PyFloat, PyInt, PyList, PyString, PyThreadable
from helpers import make_ctx, parse_stmt


def test_ann_assign_decl_and_init():
    ctx = make_ctx()
    assert Syntax.stmt(parse_stmt("x: int"), ctx) == ["    int x;"]
    assert "x" in ctx.symbols
    ctx2 = make_ctx(symbols={"a": PyInt()})
    lines = Syntax.stmt(parse_stmt("b: int = a + 10"), ctx2)
    assert lines == ["    int b = (a + 10);"]
    ctx3 = make_ctx()
    lines = Syntax.stmt(parse_stmt("xs: list[int] = [1, 2, 3]"), ctx3)
    assert lines == ["    std::vector<int> xs = std::vector<int>{1, 2, 3};"]
    ctx4 = make_ctx()
    lines = Syntax.stmt(parse_stmt("ys: list[int] = []"), ctx4)
    assert lines == ["    std::vector<int> ys = {};"]


def test_ann_assign_redeclaration():
    ctx = make_ctx(symbols={"x": PyInt()})
    with pytest.raises(TypeError, match="redeclaration"):
        Syntax.stmt(parse_stmt("x: int = 1"), ctx)


def test_assign_known_name_and_attr():
    ctx = make_ctx(
        symbols={"x": PyInt(), "p": PyThreadable("Particle")}
    )
    assert Syntax.stmt(parse_stmt("x = 3"), ctx) == ["    x = 3;"]
    assert Syntax.stmt(parse_stmt("p.velocity = 1.0"), ctx) == [
        "    p.velocity = 1.0;"
    ]


def test_assign_subscript_list_and_dict():
    ctx = make_ctx(
        symbols={
            "xs": PyList(PyFloat()),
            "d": PyDict(PyString(), PyInt()),
            "i": PyInt(),
            "k": PyString(),
            "v": PyFloat(),
            "n": PyInt(),
        }
    )
    assert Syntax.stmt(parse_stmt("xs[i] = v"), ctx) == ["    (xs[i]) = v;"]
    assert Syntax.stmt(parse_stmt("d[k] = n"), ctx) == ["    (d[k]) = n;"]


def test_assign_unknown_and_multi_target():
    ctx = make_ctx()
    with pytest.raises(TypeError, match="unknown name"):
        Syntax.stmt(parse_stmt("x = 1"), ctx)
    with pytest.raises(TypeError, match="single-target"):
        Syntax.stmt(parse_stmt("a = b = 1"), ctx)


def test_aug_assign():
    ctx = make_ctx(
        symbols={"x": PyInt(), "p": PyThreadable("Particle"), "dt": PyFloat()}
    )
    assert Syntax.stmt(parse_stmt("x += 1"), ctx) == ["    x += 1;"]
    assert Syntax.stmt(parse_stmt("p.x += dt"), ctx) == ["    p.x += dt;"]


def test_return_pass_break_continue():
    ctx = make_ctx(symbols={"x": PyInt()})
    assert Syntax.stmt(parse_stmt("return"), ctx) == ["    return;"]
    assert Syntax.stmt(parse_stmt("return x"), ctx) == ["    return x;"]
    assert Syntax.stmt(parse_stmt("pass"), ctx) == []
    assert Syntax.stmt(parse_stmt("break"), ctx) == ["    break;"]
    assert Syntax.stmt(parse_stmt("continue"), ctx) == ["    continue;"]


def test_if_else():
    ctx = make_ctx(symbols={"a": PyInt(), "b": PyInt()})
    lines = Syntax.stmt(
        parse_stmt(
            """
            if a > b:
                return a
            else:
                return b
            """
        ),
        ctx,
    )
    joined = "\n".join(lines)
    assert "if ((a > b)) {" in joined
    assert "else {" in joined
    assert "return a;" in joined


def test_while_and_while_else_rejected():
    ctx = make_ctx(symbols={"n": PyInt()})
    lines = Syntax.stmt(
        parse_stmt(
            """
            while n > 0:
                n -= 1
            """
        ),
        ctx,
    )
    assert lines[0].startswith("    while ((n > 0))")
    with pytest.raises(TypeError, match="while/else"):
        Syntax.stmt(
            parse_stmt(
                """
                while n > 0:
                    n -= 1
                else:
                    pass
                """
            ),
            ctx,
        )


def test_for_range_shapes():
    ctx = make_ctx(symbols={"n": PyInt(), "a": PyInt(), "b": PyInt(), "s": PyInt()})
    one = "\n".join(Syntax.stmt(parse_stmt("for i in range(n):\n    pass"), ctx))
    assert "for (int i = 0; i < n; i += 1)" in one

    two = "\n".join(
        Syntax.stmt(parse_stmt("for i in range(a, b):\n    pass"), ctx)
    )
    assert "for (int i = a; i < b; i += 1)" in two

    three = "\n".join(
        Syntax.stmt(parse_stmt("for i in range(a, b, s):\n    pass"), ctx)
    )
    assert "for (int i = a; i < b; i += s)" in three


def test_for_list_foreach():
    ctx = make_ctx(symbols={"xs": PyList(PyFloat())})
    lines = Syntax.stmt(
        parse_stmt(
            """
            for x in xs:
                pass
            """
        ),
        ctx,
    )
    assert lines[0] == "    for (auto& x : xs) {"
    assert "x" not in ctx.symbols


def test_for_errors():
    ctx = make_ctx(symbols={"xs": PyInt(), "i": PyInt()})
    with pytest.raises(TypeError, match="for/else"):
        Syntax.stmt(
            parse_stmt(
                """
                for i in range(3):
                    pass
                else:
                    pass
                """
            ),
            make_ctx(),
        )
    with pytest.raises(TypeError, match="rebinds"):
        Syntax.stmt(parse_stmt("for i in range(3):\n    pass"), ctx)
    with pytest.raises(TypeError, match="must be a list"):
        Syntax.stmt(
            parse_stmt("for x in xs:\n    pass"),
            make_ctx(symbols={"xs": PyInt()}),
        )


def test_unsupported_stmt_comment():
    ctx = make_ctx()
    lines = Syntax.stmt(ast.Import(names=[]), ctx)
    assert lines[0].startswith("    // unsupported statement")
