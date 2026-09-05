"""
Issue GPU-01: cthreads.gpu Vulkan context probe.

Always-safe paths run everywhere. Live Vulkan checks skip when the
extension was built without CTHREADS_GPU or when no compute device
is available (CI / headless / no driver).
"""

from __future__ import annotations

import pytest

from cthreads import gpu
from cthreads.gpu.errors import (
    CThreadsGPUError,
    GPUNotAvailable,
    VulkanInitFailed,
    VulkanLoaderNotFound,
    VulkanNoDevice,
    VulkanNotBuiltError,
)


# ---------------------------------------------------------------------------
# Error types (no native / GPU required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        CThreadsGPUError,
        VulkanNotBuiltError,
        VulkanLoaderNotFound,
        VulkanNoDevice,
        VulkanInitFailed,
        GPUNotAvailable,
    ],
)
def test_error_is_cthreads_gpu_error(cls):
    err = cls()
    assert isinstance(err, CThreadsGPUError)
    assert isinstance(err, Exception)
    assert "cthreads gpu error" in str(err)
    assert err.detail


def test_error_custom_detail():
    err = VulkanInitFailed("cthreads.gpu.VulkanInitFailed: boom")
    assert err.detail == "cthreads.gpu.VulkanInitFailed: boom"
    assert "boom" in str(err)


def test_gpu_module_exports():
    for name in (
        "available",
        "device_name",
        "init",
        "shutdown",
        "CThreadsGPUError",
        "VulkanNotBuiltError",
        "VulkanLoaderNotFound",
        "VulkanNoDevice",
        "VulkanInitFailed",
        "GPUNotAvailable",
    ):
        assert hasattr(gpu, name)


# ---------------------------------------------------------------------------
# Soft path: extension built without CTHREADS_GPU (_gpu is None)
# ---------------------------------------------------------------------------


def test_not_built_available_false(monkeypatch):
    monkeypatch.setattr(gpu, "_gpu", None)
    assert gpu.available() is False


def test_not_built_device_name_raises(monkeypatch):
    monkeypatch.setattr(gpu, "_gpu", None)
    with pytest.raises(VulkanNotBuiltError, match="CTHREADS_GPU"):
        gpu.device_name()


def test_not_built_init_raises(monkeypatch):
    monkeypatch.setattr(gpu, "_gpu", None)
    with pytest.raises(VulkanNotBuiltError, match="CTHREADS_GPU"):
        gpu.init()


def test_not_built_shutdown_noop(monkeypatch):
    monkeypatch.setattr(gpu, "_gpu", None)
    gpu.shutdown()  # must not raise


# ---------------------------------------------------------------------------
# Error mapping from C++ message prefixes (fake _ext.gpu)
# ---------------------------------------------------------------------------


class _FakeGpu:
    def __init__(self, exc: BaseException | None = None, ready: bool = False):
        self._exc = exc
        self._ready = ready
        self.init_calls = 0
        self.shutdown_calls = 0

    def available(self) -> bool:
        if self._exc is not None:
            raise self._exc
        return self._ready

    def device_name(self) -> str:
        if self._exc is not None:
            raise self._exc
        return "FakeGPU"

    def init(self) -> None:
        self.init_calls += 1
        if self._exc is not None:
            raise self._exc

    def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest.mark.parametrize(
    "msg,exc_type",
    [
        ("cthreads.gpu.VulkanLoaderNotFound: vulkan-1.dll not found", VulkanLoaderNotFound),
        ("cthreads.gpu.VulkanNoDevice: no physical devices", VulkanNoDevice),
        ("cthreads.gpu.VulkanInitFailed: vkCreateInstance failed", VulkanInitFailed),
        ("cthreads.gpu.VulkanNotBuilt: should not happen from C++", VulkanNotBuiltError),
        ("something else entirely", VulkanInitFailed),
    ],
)
def test_map_error_via_device_name(monkeypatch, msg, exc_type):
    monkeypatch.setattr(gpu, "_gpu", _FakeGpu(RuntimeError(msg)))
    with pytest.raises(exc_type) as ei:
        gpu.device_name()
    assert msg in str(ei.value)


def test_map_error_via_init(monkeypatch):
    monkeypatch.setattr(
        gpu,
        "_gpu",
        _FakeGpu(RuntimeError("cthreads.gpu.VulkanLoaderNotFound: missing")),
    )
    with pytest.raises(VulkanLoaderNotFound):
        gpu.init()


def test_fake_available_false_without_raising(monkeypatch):
    """Mirrors C++ available(): False when init would fail, no exception."""
    monkeypatch.setattr(gpu, "_gpu", _FakeGpu(ready=False))
    assert gpu.available() is False


def test_fake_ready_device_name(monkeypatch):
    monkeypatch.setattr(gpu, "_gpu", _FakeGpu(ready=True))
    assert gpu.available() is True
    assert gpu.device_name() == "FakeGPU"
    gpu.init()
    gpu.shutdown()
    assert gpu._gpu.shutdown_calls == 1


# ---------------------------------------------------------------------------
# Live Vulkan (skip when not built / no device)
# ---------------------------------------------------------------------------


def _ext_gpu_built() -> bool:
    return gpu._gpu is not None


def _require_gpu():
    if not _ext_gpu_built():
        pytest.skip("cthreads built without CTHREADS_GPU (_ext.gpu missing)")
    if not gpu.available():
        pytest.skip("Vulkan loader/device not available in this environment")


def test_live_available_is_bool():
    """available() must never raise; False is fine without GPU."""
    assert isinstance(gpu.available(), bool)


def test_live_shutdown_safe_without_init():
    """shutdown is always safe (no-op if not built / not ready)."""
    gpu.shutdown()


def test_live_unavailable_device_name_raises_mapped():
    """When built but init fails, device_name raises a mapped CThreadsGPUError."""
    if not _ext_gpu_built():
        pytest.skip("cthreads built without CTHREADS_GPU")
    if gpu.available():
        pytest.skip("GPU is available — covered by live success tests")
    with pytest.raises(CThreadsGPUError):
        gpu.device_name()


def test_live_unavailable_init_raises_mapped():
    if not _ext_gpu_built():
        pytest.skip("cthreads built without CTHREADS_GPU")
    if gpu.available():
        pytest.skip("GPU is available — covered by live success tests")
    with pytest.raises(CThreadsGPUError):
        gpu.init()


def test_live_device_name_nonempty():
    _require_gpu()
    name = gpu.device_name()
    assert isinstance(name, str)
    assert len(name.strip()) > 0


def test_live_init_idempotent_then_shutdown_reinit():
    _require_gpu()
    try:
        gpu.init()
        gpu.init()  # second call no-ops
        name1 = gpu.device_name()
        gpu.shutdown()
        assert gpu.available() is True  # re-inits
        name2 = gpu.device_name()
        assert name1 == name2
    finally:
        gpu.shutdown()


def test_live_available_true_matches_device_name():
    _require_gpu()
    try:
        assert gpu.available() is True
        assert gpu.device_name()
    finally:
        gpu.shutdown()
