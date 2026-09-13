"""Unit tests for gpu_marshal helpers."""

from __future__ import annotations

import pytest

from cthreads.gpu.gpu_kernel_meta import build_gpu_kernel_meta
from cthreads.gpu.gpu_marshal import infer_group_count_x, ordered_values_for_meta


def _saxpy_meta():
    def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
        pass

    return build_gpu_kernel_meta(saxpy).to_dict()


def test_ordered_values_ok():
    meta = _saxpy_meta()
    args = (4, 2.0, [1.0], [0.0])
    out = ordered_values_for_meta(meta, args)
    assert out == [4, 2.0, [1.0], [0.0]]
    assert out[2] is args[2]


def test_ordered_values_arity():
    meta = _saxpy_meta()
    with pytest.raises(TypeError, match="expected 4"):
        ordered_values_for_meta(meta, (1, 2.0))


@pytest.mark.parametrize("extra", [0, 1, 5, 10])
def test_ordered_values_too_many(extra):
    meta = _saxpy_meta()
    args = (1, 2.0, [0.0], [0.0]) + (0,) * extra
    if extra == 0:
        assert ordered_values_for_meta(meta, args) == list(args)
    else:
        with pytest.raises(TypeError, match="expected 4"):
            ordered_values_for_meta(meta, args)


def test_ordered_values_missing_params():
    with pytest.raises(TypeError, match="params"):
        ordered_values_for_meta({"symbol": "k"}, (1,))


def test_ordered_values_params_not_list():
    with pytest.raises(TypeError, match="params"):
        ordered_values_for_meta({"params": "bad"}, (1,))


@pytest.mark.parametrize(
    "n, local, expect",
    [
        (1, 64, 1),
        (63, 64, 1),
        (64, 64, 1),
        (65, 64, 2),
        (128, 64, 2),
        (129, 64, 3),
        (130, 64, 3),
        (200, 64, 4),
        (1, 1, 1),
        (10, 1, 10),
        (10, 8, 2),
        (16, 8, 2),
        (17, 8, 3),
    ],
)
def test_infer_group_count_table(n, local, expect):
    meta = _saxpy_meta()
    meta["local_size_x"] = local
    args = [n, 1.0, [0.0] * n, [0.0] * n]
    assert infer_group_count_x(meta, args) == expect


def test_infer_group_count_from_n():
    meta = _saxpy_meta()
    meta["local_size_x"] = 64
    args = [130, 1.0, [0.0] * 130, [0.0] * 130]
    assert infer_group_count_x(meta, args) == 3


def test_infer_group_count_exact_multiple():
    meta = _saxpy_meta()
    meta["local_size_x"] = 64
    args = [128, 1.0, [0.0] * 128, [0.0] * 128]
    assert infer_group_count_x(meta, args) == 2


def test_infer_group_count_from_list_len_without_n():
    def k(x: list[float], y: list[float]) -> None:
        pass

    meta = build_gpu_kernel_meta(k).to_dict()
    meta["local_size_x"] = 64
    assert infer_group_count_x(meta, [[0.0] * 100, [0.0] * 50]) == 2


def test_infer_group_count_empty_defaults_one():
    def k(a: float) -> None:
        pass

    meta = build_gpu_kernel_meta(k).to_dict()
    assert infer_group_count_x(meta, [1.0]) == 1


def test_infer_group_count_n_zero_defaults_one():
    meta = _saxpy_meta()
    assert infer_group_count_x(meta, [0, 1.0, [], []]) == 1


def test_infer_group_count_n_negative_defaults_one():
    meta = _saxpy_meta()
    assert infer_group_count_x(meta, [-5, 1.0, [], []]) == 1


def test_infer_group_count_bad_local_size_falls_back_64():
    meta = _saxpy_meta()
    meta["local_size_x"] = 0
    # n=130 / 64 => 3
    assert infer_group_count_x(meta, [130, 1.0, [0.0] * 130, [0.0] * 130]) == 3


def test_infer_group_count_missing_local_size_defaults_64():
    meta = _saxpy_meta()
    meta.pop("local_size_x", None)
    assert infer_group_count_x(meta, [64, 1.0, [0.0] * 64, [0.0] * 64]) == 1


def test_infer_group_count_ignores_non_n_int_named_differently():
    def k(count: int, x: list[float]) -> None:
        pass

    meta = build_gpu_kernel_meta(k).to_dict()
    meta["local_size_x"] = 64
    # No param named `n` -> use longest list (100) => ceil(100/64)=2
    # even though count is 1
    assert infer_group_count_x(meta, [1, [0.0] * 100]) == 2


def test_infer_group_count_no_params_returns_one():
    assert infer_group_count_x({"local_size_x": 64}, []) == 1


def test_infer_group_count_skips_non_dict_params():
    meta = {
        "local_size_x": 64,
        "params": ["bad", {"name": "n", "kind": "int"}],
    }
    # Index of `n` is 1 — args must be long enough for that slot.
    assert infer_group_count_x(meta, [None, 130]) == 3
