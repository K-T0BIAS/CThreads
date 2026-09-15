"""GpuArena residency: bind once, relaunch without re-upload, sync download."""

from __future__ import annotations

import pytest

from helpers_gpu import prepare_module

from cthreads.frontend.Registry import REGISTRY
from cthreads.gpu import (
    GlobalIdx,
    Gpu,
    GpuArena,
    GpuInvalidArgument,
    available,
    gpu,
    shutdown,
)


pytestmark = pytest.mark.skipif(
    not available(),
    reason="GPU / Vulkan not available",
)


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


def test_arena_bind_reuse_and_sync():
    @Gpu
    def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = a * x[i] + y[i]

    x = [1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 30.0, 40.0]
    with GpuArena() as arena:
        arena.bind(x=x, y=y)
        gpu(saxpy, len(x), 2.0, x, y).join(download=False)
        gpu(saxpy, len(x), 2.0, x, y).join(download=False)
        # Host lists unchanged until sync (device authoritative).
        assert y == [10.0, 20.0, 30.0, 40.0]
        arena.sync()
        assert y == pytest.approx([14.0, 28.0, 42.0, 56.0])


def test_arena_join_download_true_still_writeback():
    @Gpu
    def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = a * x[i] + y[i]

    x = [1.0, 2.0]
    y = [0.0, 0.0]
    with GpuArena() as arena:
        arena.bind(x=x, y=y)
        gpu(saxpy, len(x), 3.0, x, y).join(download=True)
        assert y == pytest.approx([3.0, 6.0])


def test_arena_length_mismatch_raises():
    @Gpu
    def saxpy(n: int, a: float, x: list[float], y: list[float]) -> None:
        i: int = GlobalIdx.x
        if i >= n:
            return
        y[i] = a * x[i] + y[i]

    x = [1.0, 2.0, 3.0]
    y = [0.0, 0.0, 0.0]
    with GpuArena() as arena:
        arena.bind(x=x, y=y)
        x.append(4.0)
        with pytest.raises(GpuInvalidArgument, match="length changed"):
            gpu(saxpy, len(x), 1.0, x, y)


def test_arena_duplicate_list_bind_raises():
    x = [1.0, 2.0]
    with GpuArena() as a1:
        a1.bind(x=x)
        with GpuArena() as a2:
            with pytest.raises(GpuInvalidArgument, match="already bound"):
                a2.bind(x=x)
