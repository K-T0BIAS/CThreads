# Native types shipped in the pybind module (not codegen'd as Threadables).
# keyed by Python class __name__ as exposed on cthreads.sync / cthreads.linalg
SYNC_INTERNAL_NAMES: frozenset[str] = frozenset({"Lock", "Event", "RWLock", "Barrier"})

CTHREADS_INTERNAL_TYPES: dict[str, dict[str, str]] = {
    "Lock": {
        "cpp_name": "cthreads::sync::Lock",
        "cpp_include": "sync/pyLock.hpp",
    },
    "Event": {
        "cpp_name": "cthreads::sync::Event",
        "cpp_include": "sync/pyEvent.hpp",
    },
    "RWLock": {
        "cpp_name": "cthreads::sync::RWLock",
        "cpp_include": "sync/pyRWLock.hpp",
    },
    "Barrier": {
        "cpp_name": "cthreads::sync::Barrier",
        "cpp_include": "sync/pyBarrier.hpp",
    },
    # Fixed-capacity triple buffers (cthreads.sync.TBuffer*).
    "TBufferF64": {
        "cpp_name": "cthreads::sync::tripple_buffer<double>",
        "cpp_include": "sync/t_buffer.hpp",
    },
    "TBufferI64": {
        "cpp_name": "cthreads::sync::tripple_buffer<int>",
        "cpp_include": "sync/t_buffer.hpp",
    },
    "TBufferBool": {
        "cpp_name": "cthreads::sync::tripple_buffer<bool>",
        "cpp_include": "sync/t_buffer.hpp",
    },
    "TBufferStr": {
        "cpp_name": "cthreads::sync::tripple_buffer<std::string>",
        "cpp_include": "sync/t_buffer.hpp",
        "extra_includes": ("string",),
    },
    "TBufferListF64": {
        "cpp_name": "cthreads::sync::tripple_buffer<std::vector<double>>",
        "cpp_include": "sync/t_buffer.hpp",
        "extra_includes": ("vector",),
    },
    "TBufferDictStrF64": {
        "cpp_name": (
            "cthreads::sync::tripple_buffer<std::unordered_map<std::string, double>>"
        ),
        "cpp_include": "sync/t_buffer.hpp",
        "extra_includes": ("string", "unordered_map"),
    },
    "TBufferObj": {
        "cpp_name": "cthreads::sync::tripple_buffer<py::object>",
        "cpp_include": "sync/t_buffer.hpp",
        "extra_includes": ("pybind11/pybind11.h",),
    },
    # cthreads.linalg.* (Array / Shape / Slice).
    "ArrayF32": {
        "cpp_name": "cthreads::linalg::Array<float>",
        "cpp_include": "linalg/array.hpp",
    },
    "ArrayF64": {
        "cpp_name": "cthreads::linalg::Array<double>",
        "cpp_include": "linalg/array.hpp",
    },
    "ArrayI32": {
        "cpp_name": "cthreads::linalg::Array<int>",
        "cpp_include": "linalg/array.hpp",
    },
    "ArrayBool": {
        "cpp_name": "cthreads::linalg::Array<uint8_t>",
        "cpp_include": "linalg/array.hpp",
        "extra_includes": ("cstdint",),
    },
    "Shape": {
        "cpp_name": "cthreads::linalg::Shape",
        "cpp_include": "linalg/shape.hpp",
    },
    "Slice": {
        "cpp_name": "cthreads::linalg::Slice",
        "cpp_include": "linalg/slice.hpp",
    },
}