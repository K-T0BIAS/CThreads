#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

/** Register ``cthreads._ext.gpu`` (probe API + test-only ``testing`` submodule). */
void bind_gpu(py::module_& parent);