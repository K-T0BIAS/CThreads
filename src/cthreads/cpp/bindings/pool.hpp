#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
Pool bindings (`cthreads._ext.pool`).

Unlike `linalg.hpp`, `pool.tpp` is included from `module.cpp` inside the
anonymous namespace (after `SpawnedKernel` / `spawn_from_meta`) because
`ThreadPool.submit` returns the same Job type as `thread()`.
*/
void bind_pool(py::module_& parent);
