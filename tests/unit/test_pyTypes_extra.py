"""Extra edge-case coverage for types / hint mapping."""

import pytest

from cthreads.types import PyDict, PyList, hint_to_pytype


def test_nested_list_hint():
    py = hint_to_pytype(list[list[int]])
    assert isinstance(py, PyList)
    assert isinstance(py.inner_type, PyList)


def test_dict_requires_two_args():
    with pytest.raises(TypeError):
        hint_to_pytype(dict)
