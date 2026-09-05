#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

/** Register ``cthreads._ext.gpu`` (Vulkan context probe API). */
void bind_gpu(py::module_& parent);
