#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../headers/dispatch.hpp"
#include "../headers/pyEvent.hpp"
#include "../headers/pyLock.hpp"
#include "../headers/pyRWLock.hpp"
#include "../headers/pyThread.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

struct SpawnedKernel {
    std::unique_ptr<cthreads::CThread> thr;
    std::shared_ptr<py::object> result;

    void start() { thr->start(); }
    void join() { thr->join(); }
    void wait() { thr->wait(); }
    bool done() { return thr->done(); }
    py::object get_result() const {
        if (!result || !(*result)) {
            return py::none();
        }
        return *result;
    }
};

void fill_pack_from_values(
    const std::string& symbol,
    const py::list& params,
    const py::list& values,
    void* pack,
    py::dict types,
    py::dict schemas
) {
    py::module_ marshal = py::module_::import("cthreads.marshal");
    marshal.attr("pack_params")(
        symbol,
        params,
        values,
        reinterpret_cast<std::uintptr_t>(pack),
        types,
        schemas
    );
}

void writeback_params(
    const std::string& symbol,
    const py::list& params,
    const py::list& values,
    void* pack,
    py::dict types,
    py::dict schemas
) {
    py::module_ marshal = py::module_::import("cthreads.marshal");
    marshal.attr("writeback_params")(
        symbol,
        params,
        values,
        reinterpret_cast<std::uintptr_t>(pack),
        types,
        schemas
    );
}

py::object read_return(py::dict meta, void* pack) {
    py::module_ marshal = py::module_::import("cthreads.marshal");
    return marshal.attr("unpack_return")(
        meta,
        reinterpret_cast<std::uintptr_t>(pack)
    );
}

std::unique_ptr<SpawnedKernel> spawn_from_meta(
    py::dict meta,
    py::list ordered_values
) {
    if (!cthreads::kernels().loaded()) {
        throw std::runtime_error(
            "cthreads.thread: kernel library not loaded — "
            "call cthreads.load_kernels(BINARY_PATH) after build()"
        );
    }

    const std::string symbol = meta["symbol"].cast<std::string>();
    const std::string call_sym = meta["call_symbol"].cast<std::string>();
    const std::string new_sym = meta["args_new_symbol"].cast<std::string>();
    const std::string free_sym = meta["args_free_symbol"].cast<std::string>();
    py::list params = meta["params"].cast<py::list>();

    if (ordered_values.size() != params.size()) {
        throw py::type_error(
            "cthreads.thread: expected " + std::to_string(params.size()) +
            " args for '" + symbol + "', got " +
            std::to_string(ordered_values.size())
        );
    }

    if (cthreads::kernels().sym(call_sym.c_str()) == nullptr) {
        throw std::runtime_error(
            "cthreads.thread: no compiled kernel trampoline '" + call_sym +
            "' in '" + cthreads::kernels().path() + "'"
        );
    }

    using NewFn = void* (*)();
    using FreeFn = void (*)(void*);
    using CallFn = void (*)(void*);

    void* pack = cthreads::kernels().get<NewFn>(new_sym.c_str())();
    try {
        py::dict types = py::dict();
        py::dict schemas = py::dict();
        if (meta.contains("types")) {
            types = meta["types"].cast<py::dict>();
        }
        if (meta.contains("schemas")) {
            schemas = meta["schemas"].cast<py::dict>();
        }
        fill_pack_from_values(symbol, params, ordered_values, pack, types, schemas);
    } catch (...) {
        cthreads::kernels().get<FreeFn>(free_sym.c_str())(pack);
        throw;
    }

    auto result_slot = std::make_shared<py::object>(py::none());
    // Keep Python arg objects alive for writeback.
    auto values_keep = std::make_shared<py::list>(ordered_values);
    auto meta_keep = std::make_shared<py::dict>(meta);

    CallFn call_fn = cthreads::kernels().get<CallFn>(call_sym.c_str());
    FreeFn free_fn = cthreads::kernels().get<FreeFn>(free_sym.c_str());

    auto job = [call_fn, free_fn, pack, result_slot, values_keep, meta_keep, symbol]() mutable {
        call_fn(pack);
        {
            py::gil_scoped_acquire gil;
            py::dict types = py::dict();
            py::dict schemas = py::dict();
            if (meta_keep->contains("types")) {
                types = (*meta_keep)["types"].cast<py::dict>();
            }
            if (meta_keep->contains("schemas")) {
                schemas = (*meta_keep)["schemas"].cast<py::dict>();
            }
            writeback_params(
                symbol,
                (*meta_keep)["params"].cast<py::list>(),
                *values_keep,
                pack,
                types,
                schemas
            );
            *result_slot = read_return(*meta_keep, pack);
        }
        free_fn(pack);
        pack = nullptr;
    };

    auto spawned = std::make_unique<SpawnedKernel>();
    spawned->result = result_slot;
    spawned->thr = cthreads::CThread::thread(std::move(job));
    return spawned;
}

py::list bind_args(py::dict meta, py::args args, const py::kwargs& kwargs) {
    py::list params = meta["params"].cast<py::list>();
    const size_t n = params.size();
    std::vector<bool> filled(n, false);
    py::list ordered;
    for (size_t i = 0; i < n; ++i) {
        ordered.append(py::none());
    }

    // kwargs by name
    for (auto item : kwargs) {
        std::string key = py::str(item.first);
        bool found = false;
        for (size_t i = 0; i < n; ++i) {
            py::dict p = params[i].cast<py::dict>();
            if (p["name"].cast<std::string>() == key) {
                if (filled[i]) {
                    throw py::type_error(
                        "cthreads.thread: multiple values for argument '" + key + "'"
                    );
                }
                ordered[i] = py::reinterpret_borrow<py::object>(item.second);
                filled[i] = true;
                found = true;
                break;
            }
        }
        if (!found) {
            throw py::type_error(
                "cthreads.thread: unexpected keyword argument '" + key + "'"
            );
        }
    }

    // positional
    size_t ai = 0;
    for (size_t i = 0; i < n && ai < args.size(); ++i) {
        if (filled[i]) {
            continue;
        }
        ordered[i] = args[ai++];
        filled[i] = true;
    }
    if (ai != args.size()) {
        throw py::type_error("cthreads.thread: too many positional arguments");
    }
    for (size_t i = 0; i < n; ++i) {
        if (!filled[i]) {
            py::dict p = params[i].cast<py::dict>();
            throw py::type_error(
                "cthreads.thread: missing argument '" +
                p["name"].cast<std::string>() + "'"
            );
        }
    }
    return ordered;
}

} // namespace

PYBIND11_MODULE(_ext, m) {
    m.doc() = "cthreads native core (_ext)";

    m.attr("host_os") = cthreads::kHostOS;

    m.def(
        "load_kernels",
        &cthreads::load_kernels,
        py::arg("path"),
        "Load the shared library produced by api.build() (BINARY_PATH)."
    );

    m.def(
        "kernel_path",
        []() -> py::object {
            if (!cthreads::kernels().loaded()) {
                return py::none();
            }
            return py::str(cthreads::kernels().path());
        },
        "Path of the currently loaded kernel shared library, or None."
    );

    m.def(
        "unload_kernels",
        []() { cthreads::kernels().close(); },
        "Unload the kernel shared library (needed before force-rebuild on Windows)."
    );

    py::class_<SpawnedKernel, std::unique_ptr<SpawnedKernel>>(m, "Job")
        .def("start", &SpawnedKernel::start)
        .def("join", &SpawnedKernel::join,
             py::call_guard<py::gil_scoped_release>())
        .def("done", &SpawnedKernel::done)
        .def("wait", &SpawnedKernel::wait,
             py::call_guard<py::gil_scoped_release>())
        .def("result", &SpawnedKernel::get_result);

    m.def(
        "thread",
        [](py::object fn, py::args args, const py::kwargs& kwargs)
            -> std::unique_ptr<SpawnedKernel> {
            if (!py::hasattr(fn, "__threaded") || !fn.attr("__threaded").cast<bool>()) {
                throw py::type_error(
                    "cthreads.thread: expected a @Thread function"
                );
            }
            if (!py::hasattr(fn, "__kernel_meta__")) {
                throw std::runtime_error(
                    "cthreads.thread: missing __kernel_meta__ — call cthreads.compile() first"
                );
            }
            py::dict meta = fn.attr("__kernel_meta__").cast<py::dict>();
            py::list ordered = bind_args(meta, args, kwargs);
            return spawn_from_meta(meta, ordered);
        },
        py::arg("fn"),
        "Bind args/kwargs using compile-time kernel meta, pack into the DLL "
        "trampoline, and return an unstarted Job."
    );

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
