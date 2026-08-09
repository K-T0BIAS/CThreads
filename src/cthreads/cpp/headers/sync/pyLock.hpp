// Copyright (c) 2026 Tobias Karusseit
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <mutex>

namespace cthreads::sync {

class Lock {
private:
    std::mutex _mu;

public:
    Lock() = default;
    Lock(const Lock&) = delete;
    Lock& operator=(const Lock&) = delete;

    void acquire() { _mu.lock(); }    // bound to __enter__
    void release() { _mu.unlock(); }  // bound to __exit__

    bool try_acquire() { return _mu.try_lock(); }  // bound to try_enter

    std::mutex& native_handle() { return _mu; }
};

}  // namespace cthreads::sync
