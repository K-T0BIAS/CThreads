#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * Register ``cthreads._ext.gpu`` (probe API, launch_gpu_kernel / GpuJob,
 * GpuState singleton). Optional ``testing`` submodule only when
 * CTHREADS_GPU_TESTING is set (local gpu/testing/ sources present).
 */
void bind_gpu(py::module_& parent);
