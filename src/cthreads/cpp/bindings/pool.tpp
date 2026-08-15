#pragma once

#include <pybind11/stl.h>

#include "../headers/pool/threadPool.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

/**
Bind ``cthreads._ext.pool`` (FixedPool / ThreadPool).

Included from module.cpp inside the anonymous namespace so submit can call
``spawn_from_meta`` / ``bind_args`` with the same Job type as ``thread()``.
Called via the global ``bind_pool`` wrapper in module.cpp.
*/
inline void bind_pool_impl(py::module_& parent) {
    py::module_ pool = parent.def_submodule("pool", "thread pools");

    py::class_<cthreads::pool::ThreadPool>(pool, "ThreadPool")
        .def(
            py::init<size_t>(),
            py::arg("capacity"),
            "Create a fixed-size pool (call start() before submit)."
        )
        .def(
            "start",
            &cthreads::pool::ThreadPool::start,
            "Spawn worker threads and begin accepting tasks."
        )
        .def(
            "stop",
            &cthreads::pool::ThreadPool::stop,
            py::call_guard<py::gil_scoped_release>(),
            "Stop workers; in-flight tasks finish, queued tasks are dropped."
        )
        .def(
            "join",
            &cthreads::pool::ThreadPool::join,
            py::call_guard<py::gil_scoped_release>(),
            "Join worker threads (usually used via stop())."
        )
        .def_property_readonly(
            "capacity",
            &cthreads::pool::ThreadPool::get_capacity
        )
        .def(
            "is_running",
            [](const cthreads::pool::ThreadPool& self, int thread_id) {
                if (thread_id < 0 ||
                    static_cast<size_t>(thread_id) >= self.get_capacity()) {
                    throw py::index_error("thread_id out of range");
                }
                return self.is_running(thread_id);
            },
            py::arg("thread_id"),
            "True if worker ``thread_id`` is currently executing a task."
        )
        .def(
            "is_running",
            [](const cthreads::pool::ThreadPool& self) {
                return self.is_running();
            },
            "Per-worker busy flags (length == capacity)."
        )
        .def(
            "submit",
            [](cthreads::pool::ThreadPool& self,
               py::object fn,
               py::args args,
               const py::kwargs& kwargs) -> std::shared_ptr<SpawnedKernel> {
                if (!py::hasattr(fn, "__threaded") ||
                    !fn.attr("__threaded").cast<bool>()) {
                    throw py::type_error(
                        "cthreads.pool.ThreadPool.submit: expected a @Thread function"
                    );
                }
                if (!py::hasattr(fn, "__kernel_meta__")) {
                    throw std::runtime_error(
                        "cthreads.pool.ThreadPool.submit: missing __kernel_meta__ — "
                        "call cthreads.compile() first"
                    );
                }
                py::dict meta = fn.attr("__kernel_meta__").cast<py::dict>();
                py::list ordered = bind_args(meta, args, kwargs);
                return spawn_from_meta(meta, ordered, &self);
            },
            py::arg("fn"),
            "Pack a @Thread function and enqueue it. Returns a Job that is "
            "already queued (start() is a no-op; await/join/result work as usual)."
        );
}
