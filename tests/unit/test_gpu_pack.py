"""
Issue GPU-02: GpuPack upload/download round-trips (test-only _ext.gpu.testing).

Does not use public cthreads.gpu pack APIs (there are none). Live checks skip
when the extension was built without CTHREADS_GPU or when no device is available.
"""

from __future__ import annotations

import struct

import pytest

from cthreads import gpu
from cthreads.gpu.frontend.errors import (
    GpuInvalidArgument,
    GpuUseAfterDestroy,
)


def _ext_gpu():
    return gpu._ext_gpu_api._gpu


def _require_gpu_testing():
    ext = _ext_gpu()
    if ext is None:
        pytest.skip("cthreads built without CTHREADS_GPU (_ext.gpu missing)")
    if not hasattr(ext, "testing"):
        pytest.skip(
            "_ext.gpu.testing missing (local gpu/testing/ not in tree; "
            "optional gitignored helpers)"
        )
    if not gpu.available():
        pytest.skip("Vulkan loader/device not available in this environment")
    return ext.testing


def _map_probe(exc: BaseException):
    return gpu._map_error(exc)


def test_public_gpu_has_no_pack_roundtrip_exports():
    """Product package must not re-export test-only pack helpers."""
    assert not hasattr(gpu, "roundtrip_float_pack")
    assert not hasattr(gpu, "roundtrip_int_pack")
    assert not hasattr(gpu, "testing")
    assert "roundtrip_float_pack" not in gpu.__all__
    for name in ("GpuInvalidArgument", "GpuUseAfterDestroy", "VulkanOutOfMemory"):
        assert name in gpu.__all__


def test_testing_submodule_absent_when_not_built(monkeypatch):
    monkeypatch.setattr(gpu._ext_gpu_api, "_gpu", None)
    assert _ext_gpu() is None


def test_live_roundtrip_float_scalars_and_lists():
    testing = _require_gpu_testing()
    try:
        scalar = struct.pack("<if", 4, 2.0)
        lists = [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
        out_scalar, out_lists = testing.roundtrip_float_pack(scalar, lists)
        assert out_scalar == scalar
        assert out_lists == lists
    finally:
        gpu.shutdown()


def test_live_roundtrip_float_empty_list_slot():
    testing = _require_gpu_testing()
    try:
        scalar = struct.pack("<i", 0)
        lists = [
            [],
            [1.5, 2.5],
            [],
        ]
        out_scalar, out_lists = testing.roundtrip_float_pack(scalar, lists)
        assert out_scalar == scalar
        assert out_lists == lists
        assert out_lists[0] == []
        assert out_lists[2] == []
    finally:
        gpu.shutdown()


def test_live_roundtrip_int_lists():
    testing = _require_gpu_testing()
    try:
        scalar = struct.pack("<ii", 7, 9)
        lists = [
            [1, 2, 3],
            [],
            [100, -5],
        ]
        out_scalar, out_lists = testing.roundtrip_int_pack(scalar, lists)
        assert out_scalar == scalar
        assert out_lists == lists
    finally:
        gpu.shutdown()


def test_live_roundtrip_scalars_only():
    testing = _require_gpu_testing()
    try:
        scalar = struct.pack("<fff", 1.0, 2.0, 3.0)
        out_scalar, out_lists = testing.roundtrip_float_pack(scalar, [])
        assert out_scalar == scalar
        assert out_lists == []
    finally:
        gpu.shutdown()


def test_live_roundtrip_lists_only_no_scalars():
    testing = _require_gpu_testing()
    try:
        out_scalar, out_lists = testing.roundtrip_float_pack(b"", [[1.0], [], [2.0, 3.0]])
        assert out_scalar == b""
        assert out_lists == [[1.0], [], [2.0, 3.0]]
    finally:
        gpu.shutdown()


def test_live_roundtrip_empty_pack():
    testing = _require_gpu_testing()
    try:
        out_scalar, out_lists = testing.roundtrip_float_pack(b"", [])
        assert out_scalar == b""
        assert out_lists == []
    finally:
        gpu.shutdown()


def test_live_staging_grows_across_roundtrips():
    """Second larger transfer should grow TransferEngine staging and still round-trip."""
    testing = _require_gpu_testing()
    try:
        small = struct.pack("<f", 1.0)
        big_list = [float(i) for i in range(256)]
        s1, l1 = testing.roundtrip_float_pack(small, [[1.0, 2.0]])
        assert s1 == small and l1 == [[1.0, 2.0]]
        s2, l2 = testing.roundtrip_float_pack(small, [big_list])
        assert s2 == small and l2 == [big_list]
    finally:
        gpu.shutdown()


def test_live_probe_invalid_elem_bytes_maps():
    testing = _require_gpu_testing()
    try:
        with pytest.raises(Exception) as ei:
            testing.probe_invalid_elem_bytes()
        mapped = _map_probe(ei.value)
        assert isinstance(mapped, GpuInvalidArgument)
        assert "GpuInvalidArgument" in str(mapped)
    finally:
        gpu.shutdown()


def test_live_probe_use_after_destroy_maps():
    testing = _require_gpu_testing()
    try:
        with pytest.raises(Exception) as ei:
            testing.probe_use_after_destroy_scalars()
        mapped = _map_probe(ei.value)
        assert isinstance(mapped, GpuUseAfterDestroy)
        assert "GpuUseAfterDestroy" in str(mapped)
    finally:
        gpu.shutdown()
