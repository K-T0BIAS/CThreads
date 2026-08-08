#include <pybind11/pybind11.h>

#include "../headers/pyLock.hpp"
#include "../headers/pyEvent.hpp"
#include "../headers/pyRWLock.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cthreads, m) {
    m.doc() = "cthreads native core";

    py::module_ sync = m.def_submodule("sync", "synchronization primitives");

    auto lock = py::class_<cthreads::sync::Lock>(sync, "Lock")
        .def(py::init<>())
        .def("acquire", &cthreads::sync::Lock::acquire,
             py::call_guard<py::gil_scoped_release>())
        .def("release", &cthreads::sync::Lock::release)
        .def("try_acquire", &cthreads::sync::Lock::try_acquire,
             py::call_guard<py::gil_scoped_release>())
        .def("__enter__", [](cthreads::sync::Lock& self) -> cthreads::sync::Lock& {
            self.acquire();
            return self;
        }, py::call_guard<py::gil_scoped_release>(),
           py::return_value_policy::reference_internal)
        .def("__exit__", [](cthreads::sync::Lock& self, py::object, py::object, py::object) {
            self.release();
        });

     lock.attr("__cthreads_internal__") = true;

    auto event = py::class_<cthreads::sync::Event>(sync, "Event")
        .def(py::init<>())
        .def("set", &cthreads::sync::Event::set)
        .def("clear", &cthreads::sync::Event::clear)
        .def("is_set", &cthreads::sync::Event::is_set)
        .def("wait", &cthreads::sync::Event::wait,
             py::call_guard<py::gil_scoped_release>())
        .def("wait_for", &cthreads::sync::Event::wait_for,
             py::arg("seconds"),
             py::call_guard<py::gil_scoped_release>());

     event.attr("__cthreads_internal__") = true;

    auto rwlock = py::class_<cthreads::sync::RWLock>(sync, "RWLock")
        .def(py::init<>())
        .def("acquire_read", &cthreads::sync::RWLock::acquire_read,
             py::call_guard<py::gil_scoped_release>())
        .def("release_read", &cthreads::sync::RWLock::release_read)
        .def("try_acquire_read", &cthreads::sync::RWLock::try_acquire_read,
             py::call_guard<py::gil_scoped_release>())
        .def("acquire_write", &cthreads::sync::RWLock::acquire_write,
             py::call_guard<py::gil_scoped_release>())
        .def("release_write", &cthreads::sync::RWLock::release_write)
        .def("try_acquire_write", &cthreads::sync::RWLock::try_acquire_write,
             py::call_guard<py::gil_scoped_release>());

     rwlock.attr("__cthreads_internal__") = true;
}
