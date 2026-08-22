#pragma once
#include <unordered_map>
#include <string>
#include <mutex>
#include <functional>
#include <stdexcept>
#include <utility>

namespace cthreads {


    class SharedHost {

        private:

            std::unordered_map<std::string, void*> smem;
            std::unordered_map<std::string, std::function<void(void*)>> destroyers;
            std::mutex smem_mutex;
            std::unordered_map<std::string, int> ref_counts; // counts the num of threads actively using a reference to a smem symbol

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
            */
            template<typename T>
            void set(const std::string& name, T value) {
                std::lock_guard<std::mutex> lock(smem_mutex);
                if (smem.find(name) != smem.end()) return;
                smem[name] = static_cast<void*>(new T(std::move(value)));
                ref_counts[name] = 0;
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
                    if (rc->second <= 0) {
                        destroyers[name](smem[name]);
                        smem.erase(name);
                        destroyers.erase(name);
                        ref_counts.erase(name);
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
                    destroyers[name](it->second);
                    smem.erase(it);
                    destroyers.erase(name);
                    ref_counts.erase(name);
                }
                smem[name] = static_cast<void*>(new T(std::move(value)));
                ref_counts[name] = 0;
                destroyers[name] = [](void* data) {
                    delete static_cast<T*>(data);
                };
            }
    };
} // namespace cthreads
