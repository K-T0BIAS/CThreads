"""Boolean prefix masks: gather/scatter, comparisons, combinators. No broadcast."""

from __future__ import annotations

import pytest

from cthreads import linalg

from unit.linalg.helpers import ARRAY_PARAMS, assert_close, cast, filled


def test_bool_from_list_roundtrip():
    m = linalg.ArrayBool.from_list([True, False, True])
    assert m.shape == [3]
    assert m.to_list() == [True, False, True]
    assert m.count() == 2
    assert m.any() is True
    assert m.all() is False


def test_bool_2d_from_list():
    m = linalg.ArrayBool.from_list([[True, False], [False, True]])
    assert m.shape == [2, 2]
    assert m.to_list() == [[True, False], [False, True]]
    assert m.count() == 2


def test_bool_construct_zeros():
    m = linalg.ArrayBool([4])
    assert m.to_list() == [False, False, False, False]
    assert m.any() is False
    assert m.all() is False


def test_truth_value_ambiguous():
    m = linalg.ArrayBool.from_list([True])
    with pytest.raises(RuntimeError, match="ambiguous"):
        bool(m)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_compare_scalar(Array):
    a, _ = filled(Array, [4], start=1)
    m = a > cast(Array, 2)
    assert m.to_list() == [False, False, True, True]
    assert (a >= cast(Array, 2)).to_list() == [False, True, True, True]
    assert (a < cast(Array, 3)).to_list() == [True, True, False, False]
    assert (a == cast(Array, 2)).to_list() == [False, True, False, False]
    assert (a != cast(Array, 2)).to_list() == [True, False, True, True]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_compare_arrays(Array):
    a = Array.from_list([cast(Array, 1), cast(Array, 5), cast(Array, 3)])
    b = Array.from_list([cast(Array, 2), cast(Array, 5), cast(Array, 1)])
    assert (a > b).to_list() == [False, False, True]
    assert (a == b).to_list() == [False, True, False]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_compare_shape_mismatch(Array):
    a, _ = filled(Array, [3])
    b, _ = filled(Array, [2])
    with pytest.raises(RuntimeError, match="compare"):
        _ = a > b


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_compare_strided(Array):
    a, data = filled(Array, [2, 3], start=1)
    t = a.transpose()
    m = t > cast(Array, 3)
    expected = [[data[i][j] > cast(Array, 3) for i in range(2)] for j in range(3)]
    assert m.to_list() == expected


def test_mask_and_or_not_xor():
    a = linalg.ArrayBool.from_list([True, True, False, False])
    b = linalg.ArrayBool.from_list([True, False, True, False])
    assert (a & b).to_list() == [True, False, False, False]
    assert (a | b).to_list() == [True, True, True, False]
    assert (a ^ b).to_list() == [False, True, True, False]
    assert (~a).to_list() == [False, False, True, True]


def test_mask_combine_shape_mismatch():
    a = linalg.ArrayBool.from_list([True, False])
    b = linalg.ArrayBool.from_list([True, False, True])
    with pytest.raises(RuntimeError, match="mask"):
        _ = a & b


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_same_shape_gather_1d(Array):
    a, data = filled(Array, [5], start=1)
    m = linalg.ArrayBool.from_list([True, False, True, False, True])
    g = a[m]
    assert g.shape == [3]
    assert_close(g, [data[0], data[2], data[4]], Array)
    assert g.is_contiguous() is True


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_prefix_mask_keeps_trailing_points(Array):
    # 3D-object pattern: points (N, 3) + visible (N,) -> (K, 3)
    a, data = filled(Array, [4, 3], start=1)
    m = linalg.ArrayBool.from_list([True, False, True, False])
    g = a[m]
    assert g.shape == [2, 3]
    assert_close(g, [data[0], data[2]], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_prefix_mask_image_hwc(Array):
    a, data = filled(Array, [2, 2, 3], start=1)
    m = linalg.ArrayBool.from_list([[True, False], [False, True]])
    g = a[m]
    assert g.shape == [2, 3]
    assert_close(g, [data[0][0], data[1][1]], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_full_rank_mask_gathers_scalars(Array):
    a, data = filled(Array, [2, 3], start=1)
    m = a > cast(Array, 3)
    g = a[m]
    expected = [v for row in data for v in row if v > cast(Array, 3)]
    assert g.shape == [len(expected)]
    assert_close(g, expected, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_empty_mask_keeps_trailing_shape(Array):
    a, _ = filled(Array, [4, 3], start=1)
    m = linalg.ArrayBool.from_list([False, False, False, False])
    g = a[m]
    assert g.shape == [0, 3]
    assert g.to_list() == []


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_gather_is_a_copy(Array):
    a, data = filled(Array, [4, 3], start=1)
    m = linalg.ArrayBool.from_list([True, False, True, False])
    g = a[m]
    g[0, 0] = cast(Array, 0)
    assert a[0, 0] == data[0][0]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_no_broadcast_mask(Array):
    a, _ = filled(Array, [2, 3], start=1)
    m = linalg.ArrayBool.from_list([[True], [False]])
    with pytest.raises(RuntimeError, match="no broadcast"):
        _ = a[m]
    m1 = linalg.ArrayBool.from_list([True, False, True])
    with pytest.raises(RuntimeError, match="no broadcast"):
        _ = a[m1]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_mask_rank_too_high(Array):
    a, _ = filled(Array, [3], start=1)
    m = linalg.ArrayBool.from_list([[True, False], [False, True]])
    with pytest.raises(RuntimeError, match="rank greater"):
        _ = a[m]


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_masked_fill_scalar_prefix(Array):
    a, data = filled(Array, [4, 3], start=1)
    m = linalg.ArrayBool.from_list([True, False, True, False])
    a[m] = cast(Array, 0)
    assert_close(a[0], [0, 0, 0], Array)
    assert_close(a[1], data[1], Array)
    assert_close(a[2], [0, 0, 0], Array)
    assert_close(a[3], data[3], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_masked_fill_same_shape(Array):
    a, _ = filled(Array, [4], start=1)
    m = linalg.ArrayBool.from_list([True, False, True, False])
    a[m] = cast(Array, 9)
    assert_close(a, [9, 2, 9, 4], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_masked_scatter_prefix(Array):
    a, data = filled(Array, [4, 3], start=1)
    m = linalg.ArrayBool.from_list([True, False, True, False])
    vals = Array.from_list([
        [cast(Array, 10), cast(Array, 11), cast(Array, 12)],
        [cast(Array, 20), cast(Array, 21), cast(Array, 22)],
    ])
    a[m] = vals
    assert_close(a[0], [10, 11, 12], Array)
    assert_close(a[1], data[1], Array)
    assert_close(a[2], [20, 21, 22], Array)
    assert_close(a[3], data[3], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_scatter_shape_mismatch(Array):
    a, _ = filled(Array, [4, 3], start=1)
    m = linalg.ArrayBool.from_list([True, False, True, False])
    bad = Array.from_list([cast(Array, 1), cast(Array, 2)])
    with pytest.raises(RuntimeError, match="scatter"):
        a[m] = bad


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_strided_src_gather(Array):
    a, data = filled(Array, [3, 2], start=1)
    t = a.transpose()  # (2, 3)
    m = linalg.ArrayBool.from_list([True, False])
    g = t[m]
    assert g.shape == [1, 3]
    assert_close(g, [[data[i][0] for i in range(3)]], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_filter_by_last_feature_via_column(Array):
    # keep points whose z (col 2) > 3
    a = Array.from_list([
        [cast(Array, 1), cast(Array, 0), cast(Array, 1)],
        [cast(Array, 2), cast(Array, 0), cast(Array, 5)],
        [cast(Array, 3), cast(Array, 0), cast(Array, 4)],
        [cast(Array, 4), cast(Array, 0), cast(Array, 2)],
    ])
    z = a[:, 2]
    keep = z > cast(Array, 3)
    g = a[keep]
    assert g.shape == [2, 3]
    assert_close(g, [[2, 0, 5], [3, 0, 4]], Array)


def test_methods_match_getitem():
    a = linalg.ArrayF32.from_list([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    m = linalg.ArrayBool.from_list([False, True, True])
    assert_close(a.masked_select(m), a[m], linalg.ArrayF32)
    b = a.contiguous()
    b.masked_fill(m, 0.0)
    a[m] = 0.0
    assert_close(b, a, linalg.ArrayF32)
