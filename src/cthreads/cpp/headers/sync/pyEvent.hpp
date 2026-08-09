// Copyright (c) 2026 Tobias Karusseit
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>

namespace cthreads::sync {

class Event {
    std::mutex _mu;
    std::condition_variable _cv;
    bool _set = false;

public:
    Event() = default;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    void set() {
        {
            std::lock_guard<std::mutex> g(_mu);
            _set = true;
        }
        _cv.notify_all();
    }

    void clear() {
        std::lock_guard<std::mutex> g(_mu);
        _set = false;
    }

    bool is_set() {
        std::lock_guard<std::mutex> g(_mu);
        return _set;
    }

    // Block until set() (or already set)
    void wait() {
        std::unique_lock<std::mutex> g(_mu);
        _cv.wait(g, [this] { return _set; });
    }

    // Returns false on timeout
    bool wait_for(double seconds) {
        std::unique_lock<std::mutex> g(_mu);
        return _cv.wait_for(
            g,
            std::chrono::duration<double>(seconds),
            [this] { return _set; }
        );
    }
};

}  // namespace cthreads::sync
