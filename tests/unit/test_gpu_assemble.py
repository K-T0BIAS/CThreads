"""Unit tests for assemble_comp."""

from __future__ import annotations

import pytest

from cthreads.gpu.compiler.translation.assemble import assemble_comp


def test_assemble_wraps_version_and_main():
    pre = "layout(local_size_x = 64) in;\n"
    body = "    int i = 0;\n"
    src = assemble_comp(pre, body)
    assert src.startswith("#version 450\n")
    assert "layout(local_size_x = 64) in;" in src
    assert "void main() {" in src
    assert "    int i = 0;" in src
    assert src.rstrip().endswith("}")


@pytest.mark.parametrize("ver", [450, 460, 430])
def test_assemble_custom_version(ver):
    src = assemble_comp("", "    return;\n", version=ver)
    assert src.startswith(f"#version {ver}\n")


def test_assemble_empty_body():
    src = assemble_comp("layout(local_size_x = 1) in;\n", "")
    assert "void main() {" in src
    assert src.rstrip().endswith("}")


def test_assemble_strips_trailing_whitespace_on_parts():
    src = assemble_comp("preamble\n\n", "    body;\n\n")
    assert "\n\n\nvoid main" not in src
    assert "void main() {" in src


def test_assemble_multiline_body_preserved():
    body = "    int i = 0;\n    i += 1;\n    return;\n"
    src = assemble_comp("layout(local_size_x = 1) in;", body)
    assert "int i = 0;" in src
    assert "i += 1;" in src
    assert "return;" in src


def test_assemble_order_version_preamble_main():
    src = assemble_comp("AAA\n", "    BBB;\n")
    assert src.index("#version") < src.index("AAA") < src.index("void main")
    assert src.index("void main") < src.index("BBB")
