// Copyright (c) 2026 Tobias Karusseit
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <stdexcept>

namespace cthreads::sync {

/**
 * Fixed-party generation barrier for long-lived @Thread workers.
 *
 * All `parties` threads must call `arrive_and_wait()` before any proceeds.
 * Reusable across phases (grid → density → dynamics → …).
 */
class Barrier {
    std::mutex _mu;
    std::condition_variable _cv;
    const std::size_t _parties;
    std::size_t _count = 0;
    std::size_t _generation = 0;

public:
    explicit Barrier(std::size_t parties) : _parties(parties) {
        if (parties == 0) {
            throw std::invalid_argument("cthreads.sync.Barrier: parties must be >= 1");
        }
    }

    Barrier(const Barrier&) = delete;
    Barrier& operator=(const Barrier&) = delete;

    std::size_t parties() const { return _parties; }

    void arrive_and_wait() {
        std::unique_lock<std::mutex> g(_mu);
        const std::size_t gen = _generation;
        if (++_count == _parties) {
            _count = 0;
            ++_generation;
            _cv.notify_all();
            return;
        }
        _cv.wait(g, [this, gen] { return gen != _generation; });
    }
};

}  // namespace cthreads::sync
