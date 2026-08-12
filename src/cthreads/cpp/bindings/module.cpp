// Copyright (c) 2026 Tobias Karusseit
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../headers/dispatch.hpp"
#include "../headers/math/abs.hpp"
#include "../headers/math/clamps.hpp"
#include "../headers/math/random.hpp"
#include "../headers/pyThread.hpp"
#include "../headers/sync/pyEvent.hpp"
#include "../headers/sync/pyLock.hpp"
#include "../headers/sync/pyRWLock.hpp"
#include "../headers/sync/syncState.hpp"
#include "../headers/sync/t_buffer.hpp"

#include <cstdint>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

// JobContext TLS lives ONLY in this extension DLL. Kernels call into
// cthreads_ext_sync_state via sync_bridge (bound at load_kernels).
thread_local cthreads::detail::JobContext* g_job = nullptr;
std::atomic<int> g_ext_sync_invocations{0};

void set_job_context(cthreads::detail::JobContext* ctx) { g_job = ctx; }

void cthreads_ext_sync_state() {
    g_ext_sync_invocations.fetch_add(1, std::memory_order_relaxed);
    cthreads::detail::JobContext* ctx = g_job;
    if (!ctx || !ctx->do_writeback || !ctx->pack) {
        return;
    }
    ctx->do_writeback(ctx);
}

void bind_sync_state_to_kernels() {
    if (!cthreads::kernels().loaded()) {
        return;
    }
    void* p = cthreads::kernels().sym("cthreads_bind_sync_state");
    if (!p) {
        return; // older kernel DLL without bridge — __sync_state no-ops
    }
    using BindFn = void (*)(void (*)());
    reinterpret_cast<BindFn>(p)(&cthreads_ext_sync_state);
}

/**
Defines the c++ object behind a python Job object.
It holds the compiled cthreads threaded function (from @Thread) and 
the place to stash the result value(s).

Also owns pack + marshal metadata so mid-run `__sync_state()` / host
`sync_state(job)` can writeback Threadable/list/dict args into Python.

Attributes:
- thr: std::unique_ptr<cthreads::CThread> = pointer to the compiled cthreads threaded function
- result: std::shared_ptr<py::object> = pointer to the place to stash the result value(s)
- state_mu: mutex stolen by host sync_state / held during writeback
- pack / free_fn: args struct kept alive until final writeback
- symbol / values_keep / meta_keep / types_keep / schemas_keep: marshal inputs
- finished: true after final writeback (pack may already be freed)

Methods:
- start(): void = calls start() on the compiled cthreads threaded function
- join(): void = calls join() on the compiled cthreads threaded function
- wait(): void = calls wait() on the compiled cthreads threaded function
- done(): bool = calls done() on the compiled cthreads threaded function
- get_result(): py::object = returns the result value(s) (if no result, this returns py::none)
- sync_state(): void = host steal of state_mu + writeback into Python
*/
struct SpawnedKernel {

    std::unique_ptr<cthreads::CThread> thr;
    std::shared_ptr<py::object> result;

    // Mid-run sync / writeback ownership (was previously only lambda-local).
    std::mutex state_mu;
    void* pack = nullptr;
    void (*free_fn)(void*) = nullptr;
    std::string symbol;
    std::shared_ptr<py::list> values_keep;
    std::shared_ptr<py::dict> meta_keep;
    std::shared_ptr<py::dict> types_keep;
    std::shared_ptr<py::dict> schemas_keep;
    bool finished = false;

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

    // Host API: steal state_mu, writeback Threadables into Python, release.
    void sync_state();

    ~SpawnedKernel() {
        // Safety net if the worker never ran / free was skipped.
        if (pack && free_fn) {
            free_fn(pack);
            pack = nullptr;
        }
    }
};


/**
Converts the python side args for a @Thread function into the struct FnName_args.
It creates a perfect copy that can be passes in the generated trapoline fn (FnName__call)
so that the GIL stays released during the function call.

Args:
- symbol: the name of the kernel function (SpawnedKernel::thr)
- params: compile time param metadata from __kernel_meta__
- values: the actual python values (ordered to match params)
- pack: the pointer to the pack slot in the kernel library
- types: layout info for @Threadables or nested containers (if present)
- schemas: ^^^^

Returns:
- void = fills the pack slot with the struct FnName_args (inplace)
*/
void fill_pack_from_values(
    const std::string& symbol, // the name of the kernel function (SpawnedKernel::thr)
    const py::list& params,    // compile time param metadata from __kernel_meta__
    const py::list& values,    // the actual python values (ordered to match params)
    void* pack,                // the pointer to the pack slot in the kernel library
    py::dict types,            // layout info for @Threadables or nested containers (if present)
    py::dict schemas           // ^^^^
) {
    // get the marshall module
    py::module_ marshal = py::module_::import("cthreads.marshal");
    // call the pack_params function from the marshall module
    // this function walks all the trampoline setters for each param and calls them with the arg values
    // when done it holds a copy of the py object in the form of struct FnName_args 
    marshal.attr("pack_params")( 
        symbol,
        params,
        values,
        reinterpret_cast<std::uintptr_t>(pack),
        types,
        schemas
    );
}

/**
Writes back the result value(s) from the pack slot to the python side.

This function helps update mutable python objects such that the c++ state is mirrored 
in python. It runs for (@Threadable, list, dict) not for function local data

Args:
- symbol: the name of the kernel function (SpawnedKernel::thr)
- params: compile time param metadata from __kernel_meta__
- values: the actual python values (ordered to match params)
- pack: the pointer to the pack slot in the kernel library
- types: layout info for @Threadables or nested containers (if present)
- schemas: ^^^^
*/
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

/**
 * Extension-side writeback used by kernel `__sync_state()` (via TLS JobContext).
 * Lock order: state_mu first, then GIL (never reverse — avoids deadlock).
 */
void job_do_writeback(cthreads::detail::JobContext* ctx) {
    if (!ctx || !ctx->state_mu || !ctx->pack) {
        return;
    }
    std::lock_guard<std::mutex> g(*ctx->state_mu);
    py::gil_scoped_acquire gil;

    auto* symbol = static_cast<std::string*>(ctx->symbol);
    auto* params = static_cast<py::list*>(ctx->params);
    auto* values = static_cast<py::list*>(ctx->values);
    auto* types = static_cast<py::dict*>(ctx->types);
    auto* schemas = static_cast<py::dict*>(ctx->schemas);
    if (!symbol || !params || !values || !types || !schemas) {
        return;
    }
    writeback_params(*symbol, *params, *values, ctx->pack, *types, *schemas);
}

/**
 * Host steal: same writeback as `__sync_state()`, using SpawnedKernel-owned pack/meta.
 * No-op if the job already finished (final writeback already ran).
 */
void SpawnedKernel::sync_state() {
    if (finished || !pack) {
        return;
    }
    std::lock_guard<std::mutex> g(state_mu);
    py::gil_scoped_acquire gil;
    if (finished || !pack) {
        return;
    }
    // get the params list from the kept meta (same shape as final writeback)
    py::list params = (*meta_keep)["params"].cast<py::list>();
    writeback_params(
        symbol,
        params,
        *values_keep,
        pack,
        *types_keep,
        *schemas_keep
    );
}

/**
Reads the return data and converts it to python object(s)

Args:
- meta: the compile time metadata from __kernel_meta__
- pack: the pointer to the pack (struct FnName_args)
*/
py::object read_return(py::dict meta, void* pack) {
    py::module_ marshal = py::module_::import("cthreads.marshal");
    return marshal.attr("unpack_return")(
        meta,
        reinterpret_cast<std::uintptr_t>(pack)
    );
}


/**
Collects args from the python side and prepares the pacck struct.
Then builds the job object for the @Thread fn and merges it with the param pack.
Finally threads the job and builds a SpawnedKernel object to hold the job and relevant data

Args:
- meta: the compile time metadata from __kernel_meta__
- ordered_values: the actual python values (ordered to match params)

Returns:
- std::unique_ptr<SpawnedKernel> = the spawned kernel object
*/
std::unique_ptr<SpawnedKernel> spawn_from_meta(
    py::dict meta,
    py::list ordered_values
) {
    // ensure that the kernel library is loaded 
    if (!cthreads::kernels().loaded()) {
        throw std::runtime_error(
            "cthreads.thread: kernel library not loaded. "
            "call cthreads.load_kernels(BINARY_PATH) after build()"
        );
    }

    const std::string symbol = meta["symbol"].cast<std::string>(); // the name of the kernel function (SpawnedKernel::thr)
    const std::string call_sym = meta["call_symbol"].cast<std::string>(); // the name of the call function (FnName__call)
    const std::string new_sym = meta["args_new_symbol"].cast<std::string>(); // the name of the new function (FnName__args_new)
    const std::string free_sym = meta["args_free_symbol"].cast<std::string>(); // the name of the free function (FnName__args_free)
    py::list params = meta["params"].cast<py::list>(); // the compile time param metadata from __kernel_meta__

    // ensure the num of parameters actually matches the num of values given by the user
    if (ordered_values.size() != params.size()) {
        throw py::type_error(
            "cthreads.thread: expected " + std::to_string(params.size()) +
            " args for '" + symbol + "', got " +
            std::to_string(ordered_values.size())
        );
    }

    // verify that the call function symbol (FnName__call) is present in the kernel library
    if (cthreads::kernels().sym(call_sym.c_str()) == nullptr) {
        throw std::runtime_error(
            "cthreads.thread: no compiled kernel trampoline '" + call_sym +
            "' in '" + cthreads::kernels().path() + "'"
        );
    }

    using NewFn = void* (*)();
    using FreeFn = void (*)(void*);
    using CallFn = void (*)(void*);

    // get the pointer to the pack struct (struct FnName_args) from the kernel addressed by the new_sym (FnName__args_new)
    void* pack = cthreads::kernels().get<NewFn>(new_sym.c_str())();
    // keep types/schemas outside try so we can store them on SpawnedKernel for mid-run sync
    py::dict types = py::dict();
    py::dict schemas = py::dict();
    try {
        // try to collect the types and schemas from the meta data
        if (meta.contains("types")) {
            types = meta["types"].cast<py::dict>();
        }
        if (meta.contains("schemas")) {
            schemas = meta["schemas"].cast<py::dict>();
        }
        // write the actuall data to the pack struct (came from the kernel library, edited inplace here)
        fill_pack_from_values(symbol, params, ordered_values, pack, types, schemas);
    } catch (...) {
        cthreads::kernels().get<FreeFn>(free_sym.c_str())(pack);
        throw;
    }

    // prep the result slot to be filled by the call function
    auto result_slot = std::make_shared<py::object>(py::none());
    // Keep Python arg objects alive for writeback.
    auto values_keep = std::make_shared<py::list>(ordered_values);
    auto meta_keep = std::make_shared<py::dict>(meta);
    auto types_keep = std::make_shared<py::dict>(types);
    auto schemas_keep = std::make_shared<py::dict>(schemas);
    // params list owned for job lifetime (TLS points here; must not be a worker-stack temp)
    auto params_keep = std::make_shared<py::list>(params);

    CallFn call_fn = cthreads::kernels().get<CallFn>(call_sym.c_str()); // the function to call the kernels internal fn from
    FreeFn free_fn = cthreads::kernels().get<FreeFn>(free_sym.c_str()); // the function to free the pack struct

    // make a spawned kernel object to hold the job and relevant data
    auto spawned = std::make_unique<SpawnedKernel>();
    spawned->result = result_slot;
    // own pack + marshal inputs for the full job lifetime (mid-run __sync_state / sync_state)
    spawned->pack = pack;
    spawned->free_fn = free_fn;
    spawned->symbol = symbol;
    spawned->values_keep = values_keep;
    spawned->meta_keep = meta_keep;
    spawned->types_keep = types_keep;
    spawned->schemas_keep = schemas_keep;

    SpawnedKernel* self = spawned.get(); // raw ptr for the worker lambda (spawned outlives the thread start)
    // params_keep captured by value so the shared_ptr keeps the list alive with the lambda/job
    auto params_keep_for_job = params_keep;

    // create the cob object that runs the function, cleanes up and returns/updates the python side
    auto job = [call_fn, self, params_keep_for_job]() mutable {
        // Install TLS context so codegen'd `__sync_state()` can writeback mid-run.
        cthreads::detail::JobContext ctx{};
        ctx.state_mu = &self->state_mu;
        ctx.pack = self->pack;
        ctx.symbol = &self->symbol;
        ctx.params = params_keep_for_job.get();
        ctx.values = self->values_keep.get();
        ctx.types = self->types_keep.get();
        ctx.schemas = self->schemas_keep.get();
        ctx.do_writeback = &job_do_writeback;

        set_job_context(&ctx);
        try {
            call_fn(self->pack); // calls the kernels internal translated python fn with the packed args
        } catch (...) {
            set_job_context(nullptr);
            throw;
        }
        set_job_context(nullptr);

        {
            // acuire the gil to write back the result to the python side
            // also hold state_mu so host sync_state cannot race final writeback
            std::lock_guard<std::mutex> g(self->state_mu);
            py::gil_scoped_acquire gil;
            // write back the result to the python side
            writeback_params(
                self->symbol,
                *params_keep_for_job,
                *self->values_keep,
                self->pack,
                *self->types_keep,
                *self->schemas_keep
            );
            // read the return data and convert it to python object(s)
            *self->result = read_return(*self->meta_keep, self->pack);
            // free the pack struct (only after final writeback)
            if (self->pack && self->free_fn) {
                self->free_fn(self->pack);
                self->pack = nullptr;
            }
            self->finished = true;
        }
    };

    spawned->thr = cthreads::CThread::thread(std::move(job)); // therad the job function and return the pointer to the cthreads::CThread object
    return spawned; // return the spawned kernel object
}

/**
Takes python side args, kwargs and builds a list of arg values ordered to match the expected param signature

Args:
- meta: the compile time metadata from __kernel_meta__
- args: the passed in python args
- kwargs: the passed in python kwargs

Returns:
- py::list = the list of arg values ordered to match the expected param signature
*/
py::list bind_args(py::dict meta, py::args args, const py::kwargs& kwargs) {
    py::list params = meta["params"].cast<py::list>(); // the compile time param metadata from __kernel_meta__
    const size_t n = params.size();
    std::vector<bool> filled(n, false); // vector to keep track of values that have been assigned their spot
    py::list ordered; // the list that will hold the ordered arg values
    for (size_t i = 0; i < n; ++i) {
        ordered.append(py::none()); // initialize the list with None for each param
    }

    // kwargs by name
    for (auto item : kwargs) {
        std::string key = py::str(item.first);
        bool found = false;
        for (size_t i = 0; i < n; ++i) { // search all positions for the keyword arg
            py::dict p = params[i].cast<py::dict>();
            if (p["name"].cast<std::string>() == key) { // if the name matches, assign the value to the position
                if (filled[i]) {
                    throw py::type_error(
                        "cthreads.thread: multiple values for argument '" + key + "'"
                    );
                }
                ordered[i] = py::reinterpret_borrow<py::object>(item.second); // assign the value to the position
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

    // positional (fills free spots from left to right)
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

std::uintptr_t tbuffer_native_ptr(py::object obj, py::object type_name_obj = py::none()) {
    using cthreads::sync::tripple_buffer;

    if (py::isinstance<tripple_buffer<double>>(obj)) {
        return reinterpret_cast<std::uintptr_t>(&obj.cast<tripple_buffer<double>&>());
    }
    if (py::isinstance<tripple_buffer<int>>(obj)) {
        return reinterpret_cast<std::uintptr_t>(&obj.cast<tripple_buffer<int>&>());
    }
    if (py::isinstance<tripple_buffer<bool>>(obj)) {
        return reinterpret_cast<std::uintptr_t>(&obj.cast<tripple_buffer<bool>&>());
    }
    if (py::isinstance<tripple_buffer<std::string>>(obj)) {
        return reinterpret_cast<std::uintptr_t>(&obj.cast<tripple_buffer<std::string>&>());
    }
    if (py::isinstance<tripple_buffer<std::vector<double>>>(obj)) {
        return reinterpret_cast<std::uintptr_t>(
            &obj.cast<tripple_buffer<std::vector<double>>&>()
        );
    }
    if (py::isinstance<tripple_buffer<std::unordered_map<std::string, double>>>(obj)) {
        return reinterpret_cast<std::uintptr_t>(
            &obj.cast<tripple_buffer<std::unordered_map<std::string, double>>&>()
        );
    }
    if (py::isinstance<tripple_buffer<py::object>>(obj)) {
        return reinterpret_cast<std::uintptr_t>(&obj.cast<tripple_buffer<py::object>&>());
    }

    if (!type_name_obj.is_none()) {
        const std::string type_name = type_name_obj.cast<std::string>();
        if (py::isinstance<py::capsule>(obj)) {
            py::capsule cap = obj;
            const std::string expected = "cthreads.tbuffer." + type_name;
            if (cap.name() && std::string(cap.name()) == expected) {
                return reinterpret_cast<std::uintptr_t>(cap.get_pointer<void>());
            }
            throw std::runtime_error(
                "cthreads.tbuffer_native_ptr: capsule name does not match "
                "threadable type '" + type_name + "'");
        }
        throw std::runtime_error(
            "cthreads.tbuffer_native_ptr: no native triple buffer for "
            "threadable type '" + type_name + "'; pass a pybind buffer or "
            "capsule named cthreads.tbuffer." + type_name);
    }

    throw std::runtime_error(
        "cthreads.tbuffer_native_ptr: unsupported triple-buffer object type");
}

} // namespace

PYBIND11_MODULE(_ext, m) {
    m.doc() = "cthreads native core (_ext)";

    m.attr("host_os") = cthreads::kHostOS;

    m.def(
        "load_kernels",
        [](const std::string& path) {
            cthreads::load_kernels(path);
            bind_sync_state_to_kernels();
        },
        py::arg("path"),
        "Load the shared library produced by api.build() (BINARY_PATH)."
    );

    m.def(
        "tbuffer_native_ptr",
        &tbuffer_native_ptr,
        py::arg("obj"),
        py::arg("type_name") = py::none(),
        "Return the native address of a cthreads.sync.TBuffer* object "
        "(or threadable capsule) for kernel marshalling."
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

    m.def(
        "_debug_ext_sync_invocations",
        []() { return g_ext_sync_invocations.load(std::memory_order_relaxed); },
        "Test helper: times cthreads_ext_sync_state was entered."
    );
    m.def(
        "_debug_reset_ext_sync_invocations",
        []() { g_ext_sync_invocations.store(0, std::memory_order_relaxed); }
    );

    py::class_<SpawnedKernel, std::unique_ptr<SpawnedKernel>>(m, "Job")
        .def("start", &SpawnedKernel::start)
        .def("join", &SpawnedKernel::join,
             py::call_guard<py::gil_scoped_release>())
        .def("done", &SpawnedKernel::done)
        .def("wait", &SpawnedKernel::wait,
             py::call_guard<py::gil_scoped_release>())
        .def("result", &SpawnedKernel::get_result)
        .def(
            "sync_state",
            &SpawnedKernel::sync_state,
            py::call_guard<py::gil_scoped_release>(),
            "Steal the job state mutex and writeback Threadable/list/dict "
            "args into Python (same effect as kernel __sync_state())."
        );

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

    // Fixed-capacity triple buffers (baseline bindings for primitive/container types).
    // Thread-side writes are typically codegen'd as: buf[i].field = ... (for Threadables)
    // or buf[i] = value (for scalar/container slots).
    auto tbuf_f64 = py::class_<cthreads::sync::tripple_buffer<double>>(sync, "TBufferF64")
        .def(py::init<int>(), py::arg("capacity"))
        .def("publish", &cthreads::sync::tripple_buffer<double>::publish)
        .def("capacity", &cthreads::sync::tripple_buffer<double>::capacity)
        .def("generation", &cthreads::sync::tripple_buffer<double>::generation)
        .def("__len__", [](const cthreads::sync::tripple_buffer<double>& self) { return self.capacity(); })
        .def("__getitem__", [](cthreads::sync::tripple_buffer<double>& self, int index) -> double {
            if (index < 0 || index >= self.capacity()) throw py::index_error("TBufferF64 index out of range");
            return self[static_cast<size_t>(index)];
        })
        .def("__setitem__", [](cthreads::sync::tripple_buffer<double>& self, int index, double value) {
            if (index < 0 || index >= self.capacity()) throw py::index_error("TBufferF64 index out of range");
            self[static_cast<size_t>(index)] = value;
        })
        .def("read_copy", [](const cthreads::sync::tripple_buffer<double>& self) {
            auto* data = const_cast<cthreads::sync::tripple_buffer<double>&>(self).get_read_cpy();
            py::list out;
            const int n = self.capacity();
            for (int i = 0; i < n; ++i) out.append(data[i]);
            delete[] data;
            return out;
        });
    tbuf_f64.attr("__cthreads_internal__") = true;

    auto tbuf_i64 = py::class_<cthreads::sync::tripple_buffer<int>>(sync, "TBufferI64")
        .def(py::init<int>(), py::arg("capacity"))
        .def("publish", &cthreads::sync::tripple_buffer<int>::publish)
        .def("capacity", &cthreads::sync::tripple_buffer<int>::capacity)
        .def("generation", &cthreads::sync::tripple_buffer<int>::generation)
        .def("__len__", [](const cthreads::sync::tripple_buffer<int>& self) { return self.capacity(); })
        .def("__getitem__", [](cthreads::sync::tripple_buffer<int>& self, int index) -> int {
            if (index < 0 || index >= self.capacity()) throw py::index_error("TBufferI64 index out of range");
            return self[static_cast<size_t>(index)];
        })
        .def("__setitem__", [](cthreads::sync::tripple_buffer<int>& self, int index, int value) {
            if (index < 0 || index >= self.capacity()) throw py::index_error("TBufferI64 index out of range");
            self[static_cast<size_t>(index)] = value;
        })
        .def("read_copy", [](const cthreads::sync::tripple_buffer<int>& self) {
            auto* data = const_cast<cthreads::sync::tripple_buffer<int>&>(self).get_read_cpy();
            py::list out;
            const int n = self.capacity();
            for (int i = 0; i < n; ++i) out.append(data[i]);
            delete[] data;
            return out;
        });
    tbuf_i64.attr("__cthreads_internal__") = true;

    auto tbuf_bool = py::class_<cthreads::sync::tripple_buffer<bool>>(sync, "TBufferBool")
        .def(py::init<int>(), py::arg("capacity"))
        .def("publish", &cthreads::sync::tripple_buffer<bool>::publish)
        .def("capacity", &cthreads::sync::tripple_buffer<bool>::capacity)
        .def("generation", &cthreads::sync::tripple_buffer<bool>::generation)
        .def("__len__", [](const cthreads::sync::tripple_buffer<bool>& self) { return self.capacity(); })
        .def("__getitem__", [](cthreads::sync::tripple_buffer<bool>& self, int index) -> bool {
            if (index < 0 || index >= self.capacity()) throw py::index_error("TBufferBool index out of range");
            return self[static_cast<size_t>(index)];
        })
        .def("__setitem__", [](cthreads::sync::tripple_buffer<bool>& self, int index, bool value) {
            if (index < 0 || index >= self.capacity()) throw py::index_error("TBufferBool index out of range");
            self[static_cast<size_t>(index)] = value;
        })
        .def("read_copy", [](const cthreads::sync::tripple_buffer<bool>& self) {
            auto* data = const_cast<cthreads::sync::tripple_buffer<bool>&>(self).get_read_cpy();
            py::list out;
            const int n = self.capacity();
            for (int i = 0; i < n; ++i) out.append(data[i]);
            delete[] data;
            return out;
        });
    tbuf_bool.attr("__cthreads_internal__") = true;

    auto tbuf_str = py::class_<cthreads::sync::tripple_buffer<std::string>>(sync, "TBufferStr")
        .def(py::init<int>(), py::arg("capacity"))
        .def("publish", &cthreads::sync::tripple_buffer<std::string>::publish)
        .def("capacity", &cthreads::sync::tripple_buffer<std::string>::capacity)
        .def("generation", &cthreads::sync::tripple_buffer<std::string>::generation)
        .def("__len__", [](const cthreads::sync::tripple_buffer<std::string>& self) { return self.capacity(); })
        .def("__getitem__", [](cthreads::sync::tripple_buffer<std::string>& self, int index) -> std::string {
            if (index < 0 || index >= self.capacity()) throw py::index_error("TBufferStr index out of range");
            return self[static_cast<size_t>(index)];
        })
        .def("__setitem__", [](cthreads::sync::tripple_buffer<std::string>& self, int index, const std::string& value) {
            if (index < 0 || index >= self.capacity()) throw py::index_error("TBufferStr index out of range");
            self[static_cast<size_t>(index)] = value;
        })
        .def("read_copy", [](const cthreads::sync::tripple_buffer<std::string>& self) {
            auto* data = const_cast<cthreads::sync::tripple_buffer<std::string>&>(self).get_read_cpy();
            py::list out;
            const int n = self.capacity();
            for (int i = 0; i < n; ++i) out.append(data[i]);
            delete[] data;
            return out;
        });
    tbuf_str.attr("__cthreads_internal__") = true;

    auto tbuf_list_f64 = py::class_<cthreads::sync::tripple_buffer<std::vector<double>>>(
        sync, "TBufferListF64"
    )
        .def(py::init<int>(), py::arg("capacity"))
        .def("publish", &cthreads::sync::tripple_buffer<std::vector<double>>::publish)
        .def("capacity", &cthreads::sync::tripple_buffer<std::vector<double>>::capacity)
        .def("generation", &cthreads::sync::tripple_buffer<std::vector<double>>::generation)
        .def("__len__", [](const cthreads::sync::tripple_buffer<std::vector<double>>& self) { return self.capacity(); })
        .def("__getitem__", [](cthreads::sync::tripple_buffer<std::vector<double>>& self, int index) -> std::vector<double> {
            if (index < 0 || index >= self.capacity()) throw py::index_error("TBufferListF64 index out of range");
            return self[static_cast<size_t>(index)];
        })
        .def("__setitem__", [](cthreads::sync::tripple_buffer<std::vector<double>>& self, int index, const std::vector<double>& value) {
            if (index < 0 || index >= self.capacity()) throw py::index_error("TBufferListF64 index out of range");
            self[static_cast<size_t>(index)] = value;
        })
        .def("read_copy", [](const cthreads::sync::tripple_buffer<std::vector<double>>& self) {
            auto* data = const_cast<cthreads::sync::tripple_buffer<std::vector<double>>&>(self).get_read_cpy();
            py::list out;
            const int n = self.capacity();
            for (int i = 0; i < n; ++i) out.append(data[i]);
            delete[] data;
            return out;
        });
    tbuf_list_f64.attr("__cthreads_internal__") = true;

    auto tbuf_dict = py::class_<cthreads::sync::tripple_buffer<std::unordered_map<std::string, double>>>(
        sync, "TBufferDictStrF64"
    )
        .def(py::init<int>(), py::arg("capacity"))
        .def("publish", &cthreads::sync::tripple_buffer<std::unordered_map<std::string, double>>::publish)
        .def("capacity", &cthreads::sync::tripple_buffer<std::unordered_map<std::string, double>>::capacity)
        .def("generation", &cthreads::sync::tripple_buffer<std::unordered_map<std::string, double>>::generation)
        .def("__len__", [](const cthreads::sync::tripple_buffer<std::unordered_map<std::string, double>>& self) { return self.capacity(); })
        .def("__getitem__", [](cthreads::sync::tripple_buffer<std::unordered_map<std::string, double>>& self, int index) -> std::unordered_map<std::string, double> {
            if (index < 0 || index >= self.capacity()) throw py::index_error("TBufferDictStrF64 index out of range");
            return self[static_cast<size_t>(index)];
        })
        .def("__setitem__", [](cthreads::sync::tripple_buffer<std::unordered_map<std::string, double>>& self, int index, const std::unordered_map<std::string, double>& value) {
            if (index < 0 || index >= self.capacity()) throw py::index_error("TBufferDictStrF64 index out of range");
            self[static_cast<size_t>(index)] = value;
        })
        .def("read_copy", [](const cthreads::sync::tripple_buffer<std::unordered_map<std::string, double>>& self) {
            auto* data = const_cast<cthreads::sync::tripple_buffer<std::unordered_map<std::string, double>>&>(self).get_read_cpy();
            py::list out;
            const int n = self.capacity();
            for (int i = 0; i < n; ++i) out.append(data[i]);
            delete[] data;
            return out;
        });
    tbuf_dict.attr("__cthreads_internal__") = true;

    // Opaque Python-object slots (bridge type for Threadable objects on host side).
    // Note: kernel/codegen support for threadable field writes still requires
    // dedicated compile-time schema handling.
    auto tbuf_obj = py::class_<cthreads::sync::tripple_buffer<py::object>>(sync, "TBufferObj")
        .def(py::init<int>(), py::arg("capacity"))
        .def("publish", &cthreads::sync::tripple_buffer<py::object>::publish)
        .def("capacity", &cthreads::sync::tripple_buffer<py::object>::capacity)
        .def("generation", &cthreads::sync::tripple_buffer<py::object>::generation)
        .def("__len__", [](const cthreads::sync::tripple_buffer<py::object>& self) { return self.capacity(); })
        .def("__getitem__", [](cthreads::sync::tripple_buffer<py::object>& self, int index) -> py::object {
            if (index < 0 || index >= self.capacity()) throw py::index_error("TBufferObj index out of range");
            return self[static_cast<size_t>(index)];
        })
        .def("__setitem__", [](cthreads::sync::tripple_buffer<py::object>& self, int index, py::object value) {
            if (index < 0 || index >= self.capacity()) throw py::index_error("TBufferObj index out of range");
            self[static_cast<size_t>(index)] = std::move(value);
        })
        .def("read_copy", [](const cthreads::sync::tripple_buffer<py::object>& self) {
            auto* data = const_cast<cthreads::sync::tripple_buffer<py::object>&>(self).get_read_cpy();
            py::list out;
            const int n = self.capacity();
            for (int i = 0; i < n; ++i) out.append(data[i]);
            delete[] data;
            return out;
        });
    tbuf_obj.attr("__cthreads_internal__") = true;

    // cthreads.math — helpers not in stdlib math (abs/min/max/clamp/RNG).
    // Mark the *module* with __cthreads_internal__ (pybind bound functions are
    // builtins and cannot carry arbitrary attrs). Call lowering checks the
    // module / __module__ and emits cthreads::math::* + #include "math/*.hpp".
    py::module_ math = m.def_submodule("math", "cthreads math helpers for @Thread");
    math.attr("__cthreads_internal__") = true;

    math.def(
        "abs",
        [](double x) { return cthreads::math::abs(x); },
        py::arg("x")
    );

    math.def(
        "min",
        [](double a, double b) { return cthreads::math::min(a, b); },
        py::arg("a"),
        py::arg("b")
    );

    math.def(
        "max",
        [](double a, double b) { return cthreads::math::max(a, b); },
        py::arg("a"),
        py::arg("b")
    );

    math.def(
        "clamp",
        [](double value, double lo, double hi) {
            return cthreads::math::clamp(value, lo, hi);
        },
        py::arg("value"),
        py::arg("lo"),
        py::arg("hi")
    );

    math.def("random", &cthreads::math::random);

    math.def(
        "uniform",
        &cthreads::math::uniform,
        py::arg("lo"),
        py::arg("hi")
    );

    math.def(
        "randint",
        &cthreads::math::randint,
        py::arg("lo"),
        py::arg("hi")
    );

    math.def("seed", &cthreads::math::seed, py::arg("s"));
}
