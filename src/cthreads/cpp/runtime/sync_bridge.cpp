// Copyright (c) 2026 Tobias Karusseit
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// Compiled into the user kernel DLL (cthreads_kernels). Holds a process-wide
// function pointer into `_ext`'s sync entry (TLS lives only there). Concurrent
// jobs are fine: the pointer is the same function; per-thread JobContext is TLS
// inside `_ext`.

#include "../headers/sync/syncState.hpp"

#if defined(_WIN32)
#  define CTHREADS_BRIDGE_API extern "C" __declspec(dllexport)
#else
#  define CTHREADS_BRIDGE_API extern "C"
#endif

namespace {

void (*g_ext_sync_state)() = nullptr;

} // namespace

CTHREADS_BRIDGE_API void cthreads_bind_sync_state(void (*fn)()) {
    g_ext_sync_state = fn;
}

CTHREADS_BRIDGE_API int cthreads_sync_state_bound(void) {
    return g_ext_sync_state != nullptr ? 1 : 0;
}

namespace cthreads::detail {

void __sync_state() {
    if (g_ext_sync_state) {
        g_ext_sync_state();
    }
}

} // namespace cthreads::detail
