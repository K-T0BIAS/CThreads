#pragma once
#include <mutex>
#include <thread>
#include <vector>
#include <functional>
#include <stdexcept>
#include <queue>
#include <atomic>
#include <memory>

#include "../shared_host.hpp"


namespace cthreads::pool {

    class BasePool {
        protected:
            std::vector<std::thread> threads{};
            std::queue<std::function<void()>> tasks_queue{};
            const size_t capacity;
            std::unique_ptr<std::atomic<bool>[]> thread_status;

            // state handling
            std::mutex tasks_queue_mutex{};
            std::condition_variable cv{};
            std::atomic<bool> stop_signal{false};

            std::shared_ptr<SharedHost> shared_host;
            

        public:
            BasePool(size_t capacity):
                capacity(capacity)
            {
                if (capacity == 0)
                    throw std::invalid_argument("Capacity must be greater than 0");
                this->threads.reserve(capacity);
                this->thread_status = std::make_unique<std::atomic<bool>[]>(capacity);
                for (size_t i = 0; i < capacity; i++) {
                    this->thread_status[i].store(false);
                }
                this->shared_host = std::make_shared<SharedHost>();
            }

            BasePool(const BasePool& other) = delete;
            BasePool(BasePool&& other) noexcept = delete;
            BasePool& operator=(const BasePool& other) = delete;
            BasePool& operator=(BasePool&& other) noexcept = delete;

            virtual ~BasePool() = default;

            virtual void start() = 0;
            virtual void stop() = 0;
            virtual void join() = 0;
            virtual void submit(std::function<void()> func) = 0;

            virtual bool is_running(int thread_id) const final {
                return this->thread_status[thread_id].load();
            }

            virtual std::vector<bool> is_running() const final {
                std::vector<bool> status(this->capacity);
                for (size_t i = 0; i < this->capacity; i++) {
                    status[i] = this->thread_status[i].load();
                }
                return status;
            }

            virtual size_t get_capacity() const final {
                return capacity;
            }

            std::shared_ptr<SharedHost> shared_host_keep() const {
                return shared_host;
            }

            SharedHost* shared_host_ptr() {
                return shared_host.get();
            }
    };
}