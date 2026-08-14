#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_linalg(py::module_& parent);

#include "linalg.tpp"
