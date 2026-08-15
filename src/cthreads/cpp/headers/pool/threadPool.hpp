#pragma once
#include "basePool.hpp"

namespace cthreads::pool {

    class ThreadPool : public BasePool {
        private:
            bool started{false};

        public:
            ThreadPool(size_t capacity):
                BasePool(capacity)
            {}

            ~ThreadPool() {
                this->stop();
            }

            ThreadPool(const ThreadPool& other) = delete;
            ThreadPool(ThreadPool&& other) noexcept = delete;
            ThreadPool& operator=(const ThreadPool& other) = delete;
            ThreadPool& operator=(ThreadPool&& other) noexcept = delete;

            virtual void start() override final {
                if (this->started) {
                    throw std::runtime_error("ThreadPool already started");
                }
                this->started = true;
                for (size_t i = 0; i < this->capacity; i++) {
                    this->threads.emplace_back(std::thread(
                        [this, i]() { // worker lambda for the threads
                            while (true) {
                                std::function<void()> task;
                                { // lock scope for the mutex
                                    // make a lock from the mutex to explicitly control this thread
                                    // (cant use the mutex.lock() since the cv.wait() cant read mutex directly)
                                    std::unique_lock<std::mutex> lock(this->tasks_queue_mutex);
                                    // using the lock test
                                    this->cv.wait(lock, [this]() { // lock (this thread) while waiting. when notified: check the predicate and if true unlock (this thread)
                                        return !this->tasks_queue.empty() || this->stop_signal.load();
                                    });
                                    // if the stop signal is set, break the loop
                                    if (this->stop_signal.load()) {
                                        this->thread_status[i].store(false);
                                        break;
                                    }
                                    // get the front of the queue (the wait predicate guarantees that this is popultated)
                                    task = std::move(this->tasks_queue.front());
                                    this->tasks_queue.pop(); // pop the front of the queue (removes it)
                                    lock.unlock(); // unlock the mutex (this thread)
                                }
                                this->thread_status[i].store(true); // set the thread status to true (indicating that the thread is running)
                                task(); // execute the task
                                this->thread_status[i].store(false); // set the thread status to false (indicating that the thread is not running)
                            }
                        }
                    ));
                }
            }

            virtual void stop() override final {
                this->stop_signal.store(true);
                this->cv.notify_all(); // notify all waiting worker threads that the pool is stopping
                this->join(); // wait for in-flight tasks; queued tasks are dropped below
                this->threads.clear();
                {
                    std::lock_guard<std::mutex> lock(this->tasks_queue_mutex);
                    while (!this->tasks_queue.empty()) {
                        this->tasks_queue.pop();
                    }
                }
                this->started = false;
                this->stop_signal.store(false);
            }

            virtual void join() override final {
                for (auto& thread : this->threads) {
                    thread.join();
                }
            }
            
            void submit(std::function<void()> func) override {
                {
                    std::lock_guard<std::mutex> lock(this->tasks_queue_mutex);
                    if (!this->started) {
                        throw std::runtime_error("ThreadPool is not started");
                    }
                    if (stop_signal.load()) {
                        throw std::runtime_error("ThreadPool is stopping");
                    }
                    this->tasks_queue.push(std::move(func));
                }
                this->cv.notify_one();
            }
    };
}