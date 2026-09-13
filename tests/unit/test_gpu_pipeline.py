"""
Full GPU pipeline tests: translate -> SPIR-V -> register -> launch via gpu().

Modular pieces are covered elsewhere; this file stresses end-to-end behavior
and edge cases across the public API.
"""

from __future__ import annotations

import math

import pytest

from helpers_gpu import prepare_module

from cthreads.frontend.Registry import REGISTRY
from cthreads.gpu import GlobalIdx, Gpu, ThreadIdx, available, gpu, shutdown
from cthreads.gpu.compiler.translation.translate import translate_function_for_gpu


@pytest.fixture(autouse=True)
def _reset():
    prepare_mod = prepare_module()
    REGISTRY.clear()
    prepare_mod._gpu_prepared = False
    yield
    REGISTRY.clear()
    try:
        shutdown()
    except Exception:
        pass
    prepare_mod._gpu_prepared = False


def test_pipeline_translate_source_is_shaderc_ready():
    def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = a * x[i] + y[i]

    r = translate_function_for_gpu(saxpy, compile_spirv=True)
    assert r.spirv is not None
    assert "layout(set = 0, binding = 0" in r.source
    assert "void main()" in r.source


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
@pytest.mark.parametrize(
    "n,a",
    [
        (1, 1.0),
        (2, -0.5),
        (4, 2.0),
        (8, 0.25),
        (16, -2.0),
        (31, 1.5),
        (32, 0.0),
        (33, 7.0),
        (63, 0.5),
        (64, -1.0),
        (65, 3.25),
        (96, 1.0),
        (127, -3.0),
        (128, 0.0),
        (129, 2.5),
        (200, math.pi),
        (256, math.e),
        (300, 1.125),
    ],
)
def test_pipeline_saxpy_sizes(n, a):
    @Gpu
    def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = a * x[i] + y[i]

    x = [float(i + 1) for i in range(n)]
    y = [float(i) for i in range(n)]
    expect = [a * xi + yi for xi, yi in zip(x, y)]
    gpu(saxpy, n, a, x, y).join()
    assert y == pytest.approx(expect)


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
@pytest.mark.parametrize("n", [1, 2, 3, 7, 15, 16, 31, 32, 63, 64, 65, 100, 128, 200])
def test_pipeline_int_list(n):
    @Gpu
    def add(n: int, x: list[int], y: list[int]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = x[i] + y[i]

    x = list(range(n))
    y = [10] * n
    expect = [x[i] + 10 for i in range(n)]
    gpu(add, n, x, y).join()
    assert y == expect


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
@pytest.mark.parametrize("n", [1, 4, 17, 64, 100])
def test_pipeline_bool_list(n):
    @Gpu
    def invert(n: int, flags: list[bool], ys: list[int]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        if flags[i]:
            ys[i] = 1
        else:
            ys[i] = 0

    flags = [(i % 2) == 0 for i in range(n)]
    ys = [9] * n
    gpu(invert, n, flags, ys).join()
    assert ys == [1 if f else 0 for f in flags]


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_pipeline_bool_scalar():
    @Gpu
    def set_if(n: int, flag: bool, y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        if flag:
            y[i] = 1.0
        else:
            y[i] = 0.0

    y = [9.0, 9.0, 9.0]
    gpu(set_if, 3, True, y).join()
    assert y == [1.0, 1.0, 1.0]
    gpu(set_if, 3, False, y).join()
    assert y == [0.0, 0.0, 0.0]


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_pipeline_early_return_leaves_tail_untouched():
    @Gpu
    def partial(n: int, y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = 5.0

    y = [0.0] * 10
    gpu(partial, 3, y).join()
    assert y[:3] == [5.0, 5.0, 5.0]
    assert y[3:] == [0.0] * 7


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_pipeline_two_kernels_same_process():
    @Gpu
    def fill(n: int, y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = 1.0

    @Gpu
    def double(n: int, y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = y[i] * 2.0

    y = [0.0, 0.0, 0.0, 0.0]
    gpu(fill, 4, y).join()
    gpu(double, 4, y).join()
    assert y == [2.0, 2.0, 2.0, 2.0]


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_pipeline_inplace_identity_lists():
    @Gpu
    def copy_x_to_y(n: int, x: list[float], y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = x[i]

    x = [1.5, 2.5, 3.5]
    y = [0.0, 0.0, 0.0]
    x_id = id(x)
    y_id = id(y)
    gpu(copy_x_to_y, 3, x, y).join()
    assert id(x) == x_id and id(y) == y_id
    assert y == x


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_pipeline_n_less_than_list_len_only_updates_prefix():
    @Gpu
    def mark(n: int, y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = 1.0

    y = [0.0] * 8
    gpu(mark, 2, y).join()
    assert y == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_pipeline_three_lists():
    @Gpu
    def madd(n: int, a: list[float], b: list[float], c: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        c[i] = a[i] * b[i] + c[i]

    n = 16
    a = [float(i) for i in range(n)]
    b = [2.0] * n
    c = [1.0] * n
    expect = [a[i] * 2.0 + 1.0 for i in range(n)]
    gpu(madd, n, a, b, c).join()
    assert c == pytest.approx(expect)


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_pipeline_local_temps_and_augassign():
    @Gpu
    def scale_add(n: int, a: float, x: list[float], y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        t: float = a * x[i]
        t += y[i]
        y[i] = t

    x = [1.0, 2.0, 3.0]
    y = [10.0, 20.0, 30.0]
    gpu(scale_add, 3, 2.0, x, y).join()
    assert y == pytest.approx([12.0, 24.0, 36.0])


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_pipeline_for_range_reduction_into_slot():
    @Gpu
    def sum_prefix(n: int, x: list[int], ys: list[int]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        s: int = 0
        for j in range(i + 1):
            s += x[j]
        ys[i] = s

    x = [1, 2, 3, 4]
    ys = [0, 0, 0, 0]
    gpu(sum_prefix, 4, x, ys).join()
    assert ys == [1, 3, 6, 10]


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_pipeline_if_else_branch():
    @Gpu
    def abs_copy(n: int, x: list[float], y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        if x[i] < 0.0:
            y[i] = 0.0 - x[i]
        else:
            y[i] = x[i]

    x = [-2.0, 0.0, 3.5]
    y = [0.0, 0.0, 0.0]
    gpu(abs_copy, 3, x, y).join()
    assert y == pytest.approx([2.0, 0.0, 3.5])


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_pipeline_threadidx_local_only_first_lane_writes():
    """ThreadIdx.x == 0 within each workgroup; GlobalIdx still unique."""

    @Gpu
    def mark_lane0(n: int, y: list[int]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        lid: int = ThreadIdx.x
        if lid == 0:
            y[i] = 1
        else:
            y[i] = 0

    n = 130
    y = [9] * n
    gpu(mark_lane0, n, y).join()
    # Workgroup size 64: indices 0, 64, 128 are lane 0 in each group.
    for i in range(n):
        expect = 1 if (i % 64) == 0 else 0
        assert y[i] == expect, f"i={i}"


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_pipeline_empty_n_no_write():
    @Gpu
    def mark(n: int, y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = 1.0

    y = [0.0, 0.0, 0.0]
    gpu(mark, 0, y).join()
    assert y == [0.0, 0.0, 0.0]


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_pipeline_scalar_only_kernel_runs():
    @Gpu
    def noop(n: int) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return

    gpu(noop, 8).join()


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_pipeline_lists_only_no_scalars():
    @Gpu
    def copy_lists(x: list[float], y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= 4:
            return
        y[i] = x[i]

    x = [1.0, 2.0, 3.0, 4.0]
    y = [0.0, 0.0, 0.0, 0.0]
    gpu(copy_lists, x, y).join()
    assert y == x


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_pipeline_lists_with_dummy_n():
    """Workaround path: always pass `n` so binding 0 exists."""

    @Gpu
    def copy_n(n: int, x: list[float], y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = x[i]

    x = [1.0, 2.0, 3.0, 4.0]
    y = [0.0, 0.0, 0.0, 0.0]
    gpu(copy_n, 4, x, y).join()
    assert y == x


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
@pytest.mark.parametrize(
    "ops",
    [
        ("add", lambda a, b: a + b),
        ("sub", lambda a, b: a - b),
        ("mul", lambda a, b: a * b),
    ],
)
def test_pipeline_binop_variants(ops):
    name, py_op = ops

    if name == "add":

        @Gpu
        def kern(n: int, x: list[float], y: list[float], z: list[float]) -> None:
            i: int = GlobalIdx.x
            if i >= n:
                return
            z[i] = x[i] + y[i]

    elif name == "sub":

        @Gpu
        def kern(n: int, x: list[float], y: list[float], z: list[float]) -> None:
            i: int = GlobalIdx.x
            if i >= n:
                return
            z[i] = x[i] - y[i]

    else:

        @Gpu
        def kern(n: int, x: list[float], y: list[float], z: list[float]) -> None:
            i: int = GlobalIdx.x
            if i >= n:
                return
            z[i] = x[i] * y[i]

    x = [1.0, 2.0, 3.0, 4.0]
    y = [4.0, 3.0, 2.0, 1.0]
    z = [0.0] * 4
    gpu(kern, 4, x, y, z).join()
    assert z == pytest.approx([py_op(a, b) for a, b in zip(x, y)])


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_pipeline_compare_chain_guard():
    @Gpu
    def clamp_mark(n: int, lo: int, hi: int, y: list[int]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        if lo <= i < hi:
            y[i] = 1
        else:
            y[i] = 0

    y = [9] * 10
    gpu(clamp_mark, 10, 3, 7, y).join()
    assert y == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_pipeline_many_launches_same_kernel():
    @Gpu
    def add_k(n: int, k: float, y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = y[i] + k

    y = [0.0] * 8
    for _ in range(10):
        gpu(add_k, 8, 0.5, y).join()
    assert y == pytest.approx([5.0] * 8)


@pytest.mark.skipif(not available(), reason="Vulkan GPU not available")
def test_pipeline_int_and_float_mixed_scalars():
    @Gpu
    def mix(n: int, a: float, b: float, y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        # i is int; promote via add with float literal (no float() call plugin yet)
        t: float = a * b
        t += 0.0 + 0.0  # keep float
        if i == 0:
            y[i] = t
        elif i == 1:
            y[i] = t + 1.0
        else:
            y[i] = t + 2.0

    y = [0.0, 0.0, 0.0]
    gpu(mix, 3, 1.5, 2.0, y).join()
    assert y == pytest.approx([3.0, 4.0, 5.0])
