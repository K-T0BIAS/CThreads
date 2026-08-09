#pragma once

#include <shared_mutex>

namespace cthreads::sync {

class RWLock {
    std::shared_mutex _mu;

public:
    RWLock() = default;
    RWLock(const RWLock&) = delete;
    RWLock& operator=(const RWLock&) = delete;

    // Shared (read) lock
    void acquire_read() { _mu.lock_shared(); }
    void release_read() { _mu.unlock_shared(); }
    bool try_acquire_read() { return _mu.try_lock_shared(); }

    // Exclusive (write) lock
    void acquire_write() { _mu.lock(); }
    void release_write() { _mu.unlock(); }
    bool try_acquire_write() { return _mu.try_lock(); }

    std::shared_mutex& native_handle() { return _mu; }
};

}  // namespace cthreads::sync
