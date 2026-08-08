#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>

namespace cthreads {

class CThread {
private:
    std::function<void()> _target;
    std::thread _thread;
    std::mutex _mu;
    std::condition_variable _cv;
    bool _started = false;
    bool _done = false;
    std::exception_ptr _eptr;

public:
    explicit CThread(std::function<void()> target) : _target(std::move(target)) {}

    CThread(const CThread&) = delete;
    CThread& operator=(const CThread&) = delete;

    ~CThread() {
        if (_thread.joinable()) {
            _thread.join();
        }
    }

    // Factory: bind typed fn+args into a void() job, or pass an already-closed job.
    template <class Fn, class... Args>
    static std::unique_ptr<CThread> thread(Fn fn, Args... args) {
        return std::make_unique<CThread>(
            [fn = std::move(fn),
             tup = std::make_tuple(std::move(args)...)]() mutable {
                std::apply(std::move(fn), std::move(tup));
            }
        );
    }

    void start() {
        {
            std::lock_guard<std::mutex> g(_mu);
            if (_started) {
                return;
            }
            _started = true;
        }
        // Create the OS thread outside _mu so a fast job can't deadlock
        // waiting to set _done while start() still holds the lock.
        _thread = std::thread([this] {
            try {
                _target();
            } catch (...) {
                _eptr = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> g(_mu);
                _done = true;
            }
            _cv.notify_all();
        });
    }

    void join() {
        if (_thread.joinable()) {
            _thread.join();
        }
        if (_eptr) {
            std::rethrow_exception(_eptr);
        }
    }

    bool done() {
        std::lock_guard<std::mutex> g(_mu);
        return _done;
    }

    void wait() {
        std::unique_lock<std::mutex> g(_mu);
        _cv.wait(g, [this] { return _done; });
    }
};

} // namespace cthreads
