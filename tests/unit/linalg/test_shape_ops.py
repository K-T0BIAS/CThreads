"""view / reshape / flatten / transpose / permute / squeeze / unsqueeze + inplace."""

from __future__ import annotations

import pytest

from unit.linalg.helpers import ARRAY_PARAMS, assert_close, filled, flatten, nest


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_reshape_same_numel(Array):
    a, data = filled(Array, [2, 3])
    r = a.reshape([3, 2])
    assert r.shape == [3, 2]
    assert r.is_contiguous() is True
    assert_close(r, nest(flatten(data), [3, 2]), Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_reshape_to_1d_and_3d(Array):
    a, data = filled(Array, [2, 3])
    f = a.reshape([6])
    assert f.shape == [6]
    assert_close(f, flatten(data), Array)
    c = a.reshape([1, 2, 3])
    assert c.shape == [1, 2, 3]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_reshape_numel_mismatch(Array):
    a, _ = filled(Array, [2, 3])
    with pytest.raises(RuntimeError, match="reshape"):
        a.reshape([5])
    with pytest.raises(RuntimeError, match="view"):
        a.view([4])


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_view_matches_reshape_on_contiguous(Array):
    a, data = filled(Array, [2, 3])
    assert_close(a.view([3, 2]), a.reshape([3, 2]), Array)
    assert_close(a.view([6]), flatten(data), Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_flatten(Array):
    a, data = filled(Array, [2, 3, 2])
    f = a.flatten()
    assert f.shape == [12]
    assert f.is_contiguous() is True
    assert_close(f, flatten(data), Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_flatten_1d_is_same_values(Array):
    a, data = filled(Array, [5])
    assert_close(a.flatten(), data, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_flatten_empty_raises(Array):
    a = Array([0])
    with pytest.raises(RuntimeError, match="flatten"):
        a.flatten()


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_transpose_2d(Array):
    a, data = filled(Array, [2, 3])
    t = a.transpose()
    assert t.shape == [3, 2]
    assert t.strides == [1, 3]
    assert t.is_contiguous() is False
    expected = [[data[i][j] for i in range(2)] for j in range(3)]
    assert_close(t, expected, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_transpose_1d_noop_shape(Array):
    a, data = filled(Array, [4])
    t = a.transpose()
    assert t.shape == [4]
    assert_close(t, data, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_transpose_3d_reverses_axes(Array):
    a, data = filled(Array, [2, 3, 4])
    t = a.transpose()
    assert t.shape == [4, 3, 2]
    assert t[0, 0, 0] == data[0][0][0]
    assert t[3, 2, 1] == data[1][2][3]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_permute_identity_and_swap(Array):
    a, data = filled(Array, [2, 3])
    assert_close(a.permute([0, 1]), data, Array)
    assert_close(a.permute([1, 0]), a.transpose(), Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_permute_3d(Array):
    a, data = filled(Array, [2, 3, 4])
    p = a.permute([1, 2, 0])
    assert p.shape == [3, 4, 2]
    assert p[0, 0, 0] == data[0][0][0]
    assert p[2, 3, 1] == data[1][2][3]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_permute_errors(Array):
    a, _ = filled(Array, [2, 3, 4])
    with pytest.raises(RuntimeError, match="axes rank"):
        a.permute([0, 1])
    with pytest.raises(RuntimeError, match="out of bounds"):
        a.permute([0, 1, 3])
    with pytest.raises(RuntimeError, match="duplicate"):
        a.permute([0, 0, 1])


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_squeeze_unsqueeze(Array):
    a, data = filled(Array, [2, 3])
    u = a.unsqueeze(0)
    assert u.shape == [1, 2, 3]
    assert_close(u.squeeze(0), data, Array)
    u1 = a.unsqueeze(1)
    assert u1.shape == [2, 1, 3]
    assert_close(u1.squeeze(1), data, Array)
    u2 = a.unsqueeze(2)
    assert u2.shape == [2, 3, 1]
    assert_close(u2.squeeze(2), data, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_unsqueeze_at_end_equals_ndim(Array):
    a, _ = filled(Array, [3])
    u = a.unsqueeze(1)
    assert u.shape == [3, 1]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_squeeze_errors(Array):
    a, _ = filled(Array, [2, 3])
    with pytest.raises(RuntimeError, match="singleton"):
        a.squeeze(0)
    with pytest.raises(RuntimeError, match="out of bounds"):
        a.squeeze(2)
    u = a.unsqueeze(0)
    with pytest.raises(RuntimeError, match="unsqueeze"):
        u.unsqueeze(4)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_reshape_of_transpose_reinterprets_buffer(Array):
    a, data = filled(Array, [2, 3])
    t = a.transpose()
    r = t.reshape([6])
    assert_close(r, flatten(data), Array)
    logical = t.contiguous().flatten()
    assert_close(logical, [data[i][j] for j in range(3) for i in range(2)], Array)
    assert logical.to_list() != r.to_list()


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_inplace_reshape_flatten_transpose(Array):
    a, data = filled(Array, [2, 3])
    a._reshape([3, 2])
    assert a.shape == [3, 2]
    assert_close(a, nest(flatten(data), [3, 2]), Array)
    a._flatten()
    assert a.shape == [6]
    a._reshape([2, 3])
    a._transpose()
    assert a.shape == [3, 2]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_inplace_permute_squeeze_unsqueeze(Array):
    a, data = filled(Array, [2, 3])
    a._permute([1, 0])
    assert a.shape == [3, 2]
    a._permute([1, 0])
    assert_close(a, data, Array)
    a._unsqueeze(0)
    assert a.shape == [1, 2, 3]
    a._squeeze(0)
    assert a.shape == [2, 3]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_inplace_view(Array):
    a, data = filled(Array, [2, 3])
    a._view([6])
    assert a.shape == [6]
    assert_close(a, flatten(data), Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_out_of_place_shape_ops_do_not_mutate(Array):
    a, data = filled(Array, [2, 3])
    _ = a.reshape([3, 2])
    _ = a.transpose()
    _ = a.flatten()
    _ = a.unsqueeze(0)
    assert a.shape == [2, 3]
    assert_close(a, data, Array)
