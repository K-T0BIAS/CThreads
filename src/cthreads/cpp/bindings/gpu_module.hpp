#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * Register ``cthreads._ext.gpu`` (probe API, launch_gpu_kernel / GpuJob,
 * GpuState singleton, and test-only ``testing`` submodule).
 */
void bind_gpu(py::module_& parent);
