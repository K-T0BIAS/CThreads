#pragma once
#include <unordered_map>
#include <string>
#include <mutex>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cthreads {


    class SharedHost {

        private:

            std::unordered_map<std::string, void*> smem;
            std::unordered_map<std::string, std::function<void(void*)>> destroyers;
            std::mutex smem_mutex;
            std::unordered_map<std::string, int> ref_counts; // counts the num of threads actively using a reference to a smem symbol
            // Nested wave holds: while > 0, slots are not destroyed at refcount 0.
            // New set()/replace() entries start with ref_counts == pin_depth_ so a
            // pin taken before any smem exists still covers jobs spawned inside the wave.
            int pin_depth_{0};

            void destroy_name_unlocked(const std::string& name) {
                destroyers[name](smem[name]);
                smem.erase(name);
                destroyers.erase(name);
                ref_counts.erase(name);
            }

        public:

            SharedHost() = default;

            SharedHost(const SharedHost&) = delete;
            SharedHost& operator=(const SharedHost&) = delete;
            SharedHost(SharedHost&&) = delete;
            SharedHost& operator=(SharedHost&&) = delete;

            ~SharedHost() {
                for (const auto& [name, destroyer] : destroyers) {
                    destroyer(smem.at(name));
                }
                smem.clear();
                destroyers.clear();
            }

            /**
            Callsite for getting shared memory.
            Returns a reference to the shared object.
            Safe to call concurrently only after all set() calls for this host are done.
            */
            template<typename T>
            T& get(const std::string& name) {
                auto it = smem.find(name);
                if (it == smem.end()) {
                    throw std::runtime_error("Shared memory not initialized");
                }
                return *static_cast<T*>(it->second);
            }

            /**
            Initialize a named shared object. Host takes ownership (copy or move into heap).
            If the name is already present, this function does nothing.
            Blocks on smem_mutex so concurrent set() on names is safe.
            While the host is pinned, the new slot starts with one ref per active pin
            so it cannot be freed before matching unpin() calls.
            */
            template<typename T>
            void set(const std::string& name, T value) {
                std::lock_guard<std::mutex> lock(smem_mutex);
                if (smem.find(name) != smem.end()) return;
                smem[name] = static_cast<void*>(new T(std::move(value)));
                ref_counts[name] = pin_depth_;
                destroyers[name] = [](void* data) {
                    delete static_cast<T*>(data);
                };
            }

            void register_job(const std::vector<std::string>& names) {
                std::lock_guard<std::mutex> lock(smem_mutex);
                for (const auto& name : names) {
                    if (ref_counts.find(name) == ref_counts.end()) {
                        throw std::runtime_error("Shared memory not initialized");
                    }
                    ref_counts[name]++;
                }
            }

            void unregister_job(const std::vector<std::string>& names) {
                std::lock_guard<std::mutex> lock(smem_mutex);
                for (const auto& name : names) {
                    auto rc = ref_counts.find(name);
                    if (rc == ref_counts.end()) {
                        throw std::runtime_error("Shared memory not initialized");
                    }
                    rc->second--;
                    // Keep idle slots alive across a pin wave so later submits can join.
                    if (rc->second <= 0 && pin_depth_ == 0) {
                        destroy_name_unlocked(name);
                    }
                }
            }

            bool contains(const std::string& name) {
                std::lock_guard<std::mutex> lock(this->smem_mutex);
                return smem.find(name) != smem.end();
            }

            std::uintptr_t native_ptr(const std::string& name) {
                std::lock_guard<std::mutex> lock(this->smem_mutex);
                auto it = smem.find(name);
                if (it == smem.end()) {
                    return 0;
                }
                return reinterpret_cast<std::uintptr_t>(it->second);
            }

            /** Overwrite or insert (used for shared return snapshots). */
            template<typename T>
            void replace(const std::string& name, T value) {
                std::lock_guard<std::mutex> lock(smem_mutex);
                auto it = smem.find(name);
                if (it != smem.end()) {
                    destroy_name_unlocked(name);
                }
                smem[name] = static_cast<void*>(new T(std::move(value)));
                ref_counts[name] = pin_depth_;
                destroyers[name] = [](void* data) {
                    delete static_cast<T*>(data);
                };
            }

            /**
            Begin a submit wave. Existing slots get +1; future set()/replace() slots
            start with the current pin depth so they are covered too.
            */
            void pin() {
                std::lock_guard<std::mutex> lock(this->smem_mutex);
                pin_depth_++;
                for (auto& [name, rc] : this->ref_counts) {
                    (void)name;
                    rc++;
                }
            }

            /**
            End a submit wave. Drops one pin ref from every slot. When the last pin
            is released, destroys any slot whose job refcount is already <= 0.
            */
            void unpin() {
                std::lock_guard<std::mutex> lock(this->smem_mutex);
                if (pin_depth_ <= 0) {
                    throw std::runtime_error("SharedHost::unpin without matching pin");
                }
                for (auto& [name, rc] : this->ref_counts) {
                    (void)name;
                    rc--;
                }
                pin_depth_--;
                if (pin_depth_ != 0) {
                    return;
                }
                std::vector<std::string> dead;
                for (const auto& [name, rc] : this->ref_counts) {
                    if (rc <= 0) {
                        dead.push_back(name);
                    }
                }
                for (const auto& name : dead) {
                    destroy_name_unlocked(name);
                }
            }
    };
} // namespace cthreads
