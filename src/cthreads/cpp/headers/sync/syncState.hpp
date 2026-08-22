// Copyright (c) 2026 Tobias Karusseit
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <mutex>

namespace cthreads::detail {

/**
 * Per-running-job context for mid-run Python writeback (`__sync_state()`).
 *
 * Owned and TLS-stored only inside the `_ext` module (see module.cpp).
 * Generated kernels call `cthreads::detail::__sync_state()`, which is
 * implemented in the kernel DLL bridge (`sync_bridge.cpp`) as a call into
 * `_ext`'s bound entry — so every worker reads the same per-thread notebook.
 *
 * Opaque void* fields avoid pulling pybind11 into kernel-compiled code.
 */
struct JobContext {
    std::mutex* state_mu = nullptr;
    void* pack = nullptr;
    void* shared_host = nullptr;
    void* symbol = nullptr;   // std::string*
    void* params = nullptr;   // py::list*
    void* values = nullptr;   // py::list*
    void* types = nullptr;    // py::dict*
    void* schemas = nullptr;  // py::dict*
    void* meta = nullptr;     // py::dict*
    void (*do_writeback)(JobContext*) = nullptr;
};

/**
 * Kernel barrier: mirror C++ pack Threadable/list/dict state into Python.
 * Codegen for bare `__sync_state()` emits this call.
 * Defined in the kernel shared library (sync_bridge.cpp), not inline here —
 * an inline/TLS copy in this header would be a separate slot per DLL.
 */
void __sync_state();

} // namespace cthreads::detail
