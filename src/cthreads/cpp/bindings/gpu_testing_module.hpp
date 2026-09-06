#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * Register ``cthreads._ext.gpu.testing`` (GpuPack round-trip smoke API).
 * Test-only — not part of the public ``cthreads.gpu`` package.
 */
void bind_gpu_testing(py::module_& gpu_parent);
