"""matmul: 2D, batched, 2D-broadcast, strided, inplace, errors."""

from __future__ import annotations

import pytest

from unit.linalg.helpers import ARRAY_PARAMS, assert_close, cast, filled, matmul_2d


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_matmul_2x2(Array):
    a = Array.from_list([[cast(Array, 1), cast(Array, 2)], [cast(Array, 3), cast(Array, 4)]])
    b = Array.from_list([[cast(Array, 5), cast(Array, 6)], [cast(Array, 7), cast(Array, 8)]])
    c = a.matmul(b)
    assert c.shape == [2, 2]
    assert_close(c, [[19, 22], [43, 50]], Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_matmul_rectangular(Array):
    a, da = filled(Array, [2, 3], start=1)
    b, db = filled(Array, [3, 4], start=2)
    c = a.matmul(b)
    assert c.shape == [2, 4]
    assert_close(c, matmul_2d(da, db), Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
@pytest.mark.parametrize("m,k,n", [(1, 1, 1), (1, 3, 1), (3, 1, 2), (4, 4, 4), (5, 8, 3), (8, 8, 8), (9, 7, 5)])
def test_matmul_sizes(Array, m, k, n):
    a, da = filled(Array, [m, k], start=1)
    b, db = filled(Array, [k, n], start=2)
    c = a.matmul(b)
    assert c.shape == [m, n]
    assert_close(c, matmul_2d(da, db), Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_batched_matmul_same_leading(Array):
    a, da = filled(Array, [2, 2, 3], start=1)
    b, db = filled(Array, [2, 3, 2], start=2)
    c = a.matmul(b)
    assert c.shape == [2, 2, 2]
    expected = [matmul_2d(da[i], db[i]) for i in range(2)]
    assert_close(c, expected, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_batched_matmul_3d_leading(Array):
    a, da = filled(Array, [2, 2, 2, 2], start=1)
    b, db = filled(Array, [2, 2, 2, 2], start=1)
    c = a.matmul(b)
    assert c.shape == [2, 2, 2, 2]
    expected = [
        [matmul_2d(da[p][q], db[p][q]) for q in range(2)]
        for p in range(2)
    ]
    assert_close(c, expected, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_broadcast_a_2d_over_batched_b(Array):
    a, da = filled(Array, [2, 3], start=1)
    b, db = filled(Array, [4, 3, 2], start=2)
    c = a.matmul(b)
    assert c.shape == [4, 2, 2]
    expected = [matmul_2d(da, db[i]) for i in range(4)]
    assert_close(c, expected, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_broadcast_b_2d_over_batched_a(Array):
    a, da = filled(Array, [3, 2, 4], start=1)
    b, db = filled(Array, [4, 2], start=2)
    c = a.matmul(b)
    assert c.shape == [3, 2, 2]
    expected = [matmul_2d(da[i], db) for i in range(3)]
    assert_close(c, expected, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_strided_matmul_matches_contiguous(Array):
    a, da = filled(Array, [3, 2], start=1)
    b, db = filled(Array, [4, 2], start=3)
    at = a.transpose()  # 2x3
    bt = b.transpose()  # 2x4  wait b is 4x2, T is 2x4 — need (2x3) @ (3x4)
    # Build B as 3x4 then transpose to 4x3 for a strided RHS of 3x4? 
    rhs, dr = filled(Array, [4, 3], start=3)
    rt = rhs.transpose()  # 3x4 strided
    c = at.matmul(rt)
    expected = matmul_2d(
        [[da[i][j] for i in range(3)] for j in range(2)],
        [[dr[i][j] for i in range(4)] for j in range(3)],
    )
    assert c.shape == [2, 4]
    assert_close(c, expected, Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_matmul_rank_error(Array):
    a, _ = filled(Array, [3])
    b, _ = filled(Array, [3, 2])
    with pytest.raises(RuntimeError, match="rank"):
        a.matmul(b)
    with pytest.raises(RuntimeError, match="rank"):
        b.matmul(a)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_matmul_inner_mismatch(Array):
    a, _ = filled(Array, [2, 3])
    b, _ = filled(Array, [4, 2])
    with pytest.raises(RuntimeError, match="inner"):
        a.matmul(b)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_matmul_leading_rank_differ(Array):
    a, _ = filled(Array, [2, 2, 3, 4])
    b, _ = filled(Array, [2, 4, 5])
    with pytest.raises(RuntimeError, match="leading ranks"):
        a.matmul(b)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_matmul_leading_dims_mismatch(Array):
    a, _ = filled(Array, [2, 3, 4])
    b, _ = filled(Array, [3, 4, 5])
    with pytest.raises(RuntimeError, match="leading dimensions"):
        a.matmul(b)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_inplace_matmul(Array):
    a, da = filled(Array, [2, 2], start=1)
    b, db = filled(Array, [2, 2], start=5)
    a._matmul(b)
    assert a.shape == [2, 2]
    assert_close(a, matmul_2d(da, db), Array)


@pytest.mark.parametrize("Array", ARRAY_PARAMS)
def test_matmul_does_not_mutate_inputs(Array):
    a, da = filled(Array, [2, 3], start=1)
    b, db = filled(Array, [3, 2], start=2)
    _ = a.matmul(b)
    assert_close(a, da, Array)
    assert_close(b, db, Array)
