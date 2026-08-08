#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../headers/dispatch.hpp"
#include "../headers/pyEvent.hpp"
#include "../headers/pyLock.hpp"
#include "../headers/pyRWLock.hpp"
#include "../headers/pyThread.hpp"

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
    void* pack
) {
    auto& lib = cthreads::kernels();
    for (size_t i = 0; i < params.size(); ++i) {
        py::dict p = params[i].cast<py::dict>();
        const std::string kind = p["kind"].cast<std::string>();
        const std::string setter = symbol + "__set_a" + std::to_string(i);
        py::object val = values[i];

        if (kind == "int") {
            using Fn = void (*)(void*, int);
            lib.get<Fn>(setter.c_str())(pack, val.cast<int>());
        } else if (kind == "float") {
            using Fn = void (*)(void*, double);
            lib.get<Fn>(setter.c_str())(pack, val.cast<double>());
        } else if (kind == "bool") {
            using Fn = void (*)(void*, bool);
            lib.get<Fn>(setter.c_str())(pack, val.cast<bool>());
        } else if (kind == "threadable") {
            py::list fields = p["fields"].cast<py::list>();
            // Build positional field values from object attrs or mapping.
            std::vector<py::object> field_vals;
            field_vals.reserve(fields.size());
            for (py::handle fh : fields) {
                py::dict f = fh.cast<py::dict>();
                const std::string fname = f["name"].cast<std::string>();
                if (py::isinstance<py::dict>(val)) {
                    field_vals.push_back(val[py::str(fname)]);
                } else {
                    field_vals.push_back(val.attr(fname.c_str()));
                }
            }
            // Only the field layouts we emit: all-float Particle-style, or mixed.
            // Call through a small switch on field count + kinds.
            if (fields.size() == 3
                && fields[0].cast<py::dict>()["kind"].cast<std::string>() == "float"
                && fields[1].cast<py::dict>()["kind"].cast<std::string>() == "float"
                && fields[2].cast<py::dict>()["kind"].cast<std::string>() == "float") {
                using Fn = void (*)(void*, double, double, double);
                lib.get<Fn>(setter.c_str())(
                    pack,
                    field_vals[0].cast<double>(),
                    field_vals[1].cast<double>(),
                    field_vals[2].cast<double>()
                );
            } else {
                throw std::runtime_error(
                    "cthreads.thread: unsupported Threadable field layout for '" +
                    symbol + "' a" + std::to_string(i)
                );
            }
        } else if (kind == "list") {
            const std::string inner = p["list_inner"].cast<std::string>();
            if (inner == "float") {
                auto vec = val.cast<std::vector<double>>();
                using Fn = void (*)(void*, const double*, size_t);
                lib.get<Fn>(setter.c_str())(pack, vec.data(), vec.size());
            } else if (inner == "int") {
                auto vec = val.cast<std::vector<int>>();
                using Fn = void (*)(void*, const int*, size_t);
                lib.get<Fn>(setter.c_str())(pack, vec.data(), vec.size());
            } else {
                throw std::runtime_error(
                    "cthreads.thread: unsupported list inner type '" + inner + "'"
                );
            }
        } else {
            throw std::runtime_error(
                "cthreads.thread: unsupported param kind '" + kind + "'"
            );
        }
    }
}

void writeback_threadables(
    const std::string& symbol,
    const py::list& params,
    const py::list& values,
    void* pack
) {
    auto& lib = cthreads::kernels();
    for (size_t i = 0; i < params.size(); ++i) {
        py::dict p = params[i].cast<py::dict>();
        if (p["kind"].cast<std::string>() != "threadable") {
            continue;
        }
        py::object val = values[i];
        if (py::isinstance<py::dict>(val)) {
            continue; // no object to write back into
        }
        py::list fields = p["fields"].cast<py::list>();
        const std::string getter = symbol + "__get_a" + std::to_string(i);
        if (fields.size() == 3
            && fields[0].cast<py::dict>()["kind"].cast<std::string>() == "float"
            && fields[1].cast<py::dict>()["kind"].cast<std::string>() == "float"
            && fields[2].cast<py::dict>()["kind"].cast<std::string>() == "float") {
            double x = 0, y = 0, v = 0;
            using Fn = void (*)(void*, double*, double*, double*);
            lib.get<Fn>(getter.c_str())(pack, &x, &y, &v);
            const std::string n0 = fields[0].cast<py::dict>()["name"].cast<std::string>();
            const std::string n1 = fields[1].cast<py::dict>()["name"].cast<std::string>();
            const std::string n2 = fields[2].cast<py::dict>()["name"].cast<std::string>();
            val.attr(n0.c_str()) = x;
            val.attr(n1.c_str()) = y;
            val.attr(n2.c_str()) = v;
        }
    }
}

py::object read_return(
    const std::string& symbol,
    const std::string& return_kind,
    void* pack
) {
    if (return_kind == "void") {
        return py::none();
    }
    auto& lib = cthreads::kernels();
    const std::string getter = symbol + "__get_ret";
    if (return_kind == "int") {
        return py::int_(lib.get<int (*)(void*)>(getter.c_str())(pack));
    }
    if (return_kind == "float") {
        return py::float_(lib.get<double (*)(void*)>(getter.c_str())(pack));
    }
    if (return_kind == "bool") {
        return py::bool_(lib.get<bool (*)(void*)>(getter.c_str())(pack));
    }
    return py::none();
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
    const std::string return_kind = meta["return_kind"].cast<std::string>();
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
        fill_pack_from_values(symbol, params, ordered_values, pack);
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

    auto job = [call_fn, free_fn, pack, result_slot, values_keep, meta_keep, symbol, return_kind]() mutable {
        call_fn(pack);
        {
            py::gil_scoped_acquire gil;
            writeback_threadables(
                symbol,
                (*meta_keep)["params"].cast<py::list>(),
                *values_keep,
                pack
            );
            *result_slot = read_return(symbol, return_kind, pack);
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

PYBIND11_MODULE(cthreads, m) {
    m.doc() = "cthreads native core";

    m.attr("host_os") = cthreads::kHostOS;

    m.def(
        "load_kernels",
        &cthreads::load_kernels,
        py::arg("path"),
        "Load the shared library produced by api.build() (BINARY_PATH)."
    );

    py::class_<SpawnedKernel, std::unique_ptr<SpawnedKernel>>(m, "Thread")
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
                    "cthreads.thread: missing __kernel_meta__ — call api.compile() first"
                );
            }
            py::dict meta = fn.attr("__kernel_meta__").cast<py::dict>();
            py::list ordered = bind_args(meta, args, kwargs);
            return spawn_from_meta(meta, ordered);
        },
        py::arg("fn"),
        "Bind args/kwargs using compile-time kernel meta, pack into the DLL "
        "trampoline, and return an unstarted Thread."
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
