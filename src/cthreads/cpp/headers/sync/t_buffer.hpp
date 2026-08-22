#pragma once

#include <algorithm>
#include <atomic>
#include <cstring>

namespace cthreads::sync {

#pragma region TripleBuffer
    // Triple-buffer for a fixed-capacity array of T.
    //
    // Invariants (single producer, single consumer typical use):
    // - writer fills `write_index` slot exclusively
    // - `publish()` flips `published_index` to the just-filled slot
    // - writer then picks a spare slot != published for the next fill
    //
    // This avoids memcpy at publish time: we only swap indices.
    template <typename T>
    class tripple_buffer { // kept original name for compatibility
    private:
        static constexpr int kSlotCount = 3;

        T** buffer = new T*[kSlotCount];

        // Total number of publish() calls (monotonic). Useful for host polling/debug.
        std::atomic<int> count;

        // Slot indices in [0, kSlotCount).
        // - writer writes to write_index and never touches published_index
        // - reader reads from published_index
        std::atomic<int> write_index;
        std::atomic<int> published_index;

        const int size; // number of elements per slot

        void increment_count() {
            count.fetch_add(1, std::memory_order_relaxed);
        }

        int next_spare_slot(int current_published) const {
            // Choose the next slot after current write/published in a ring,
            // but ensure we do not pick the published slot.
            const int w = write_index.load(std::memory_order_relaxed);
            int candidate = (w + 1) % kSlotCount;
            if (candidate == current_published) {
                candidate = (candidate + 1) % kSlotCount;
            }
            return candidate;
        }

    public:
        explicit tripple_buffer(int size) : size(size) {
            for (int s = 0; s < kSlotCount; ++s) {
                buffer[s] = new T[size];
            }

            count.store(0, std::memory_order_relaxed);

            // Start with:
            // - write slot = 0 (writer will fill)
            // - published slot = 1 (reader can read something valid-ish)
            // - spare slot = 2
            write_index.store(0, std::memory_order_relaxed);
            published_index.store(1, std::memory_order_relaxed);
        }

        ~tripple_buffer() {
            for (int s = 0; s < kSlotCount; ++s) {
                delete[] buffer[s];
            }
            delete[] buffer;
        }

            /**
            * Any write to the buffer must be generated in the c++ code of the cthread
            * This is to ensure that we can write the actual attributes correctly instead of always memcpy the full object
            *
            * Example of user side code:
            * auto* slot = std::static_cast<USER_SIDE_CLASS*>(buffer->get_write_slot())
            * slot[index].attribute = value;
            * 
            * Since we emit the ptr to the memory the write is in place.
            * This means that the user must also call publish to ensure the data is readable on the py side.
            */
        T* get_write_slot() {
            const int w = write_index.load(std::memory_order_relaxed);
            return buffer[w];
            }

            /**
            * Publish the most recently written slot.
            *
            * No memcpy is performed: we only flip indices so the freshly written slot
            * becomes visible to readers atomically.
            */
        void publish() {
            // Writer finished writing `write_index` already.
            const int w = write_index.load(std::memory_order_relaxed);

            // Flip published slot to the slot that was just written.
            published_index.store(w, std::memory_order_release);

            // Pick next write slot != published slot.
            const int next_write = next_spare_slot(w);
            write_index.store(next_write, std::memory_order_relaxed);

            increment_count();
            }

            /**
            * creates a new object and copies the data from the last published slot into it
            * this is to ensure that the user can read the data without having to worry about the buffer being modified
            * while they are reading it
            */
        T* get_read_cpy() {
            // Copy the entire slot (all `size` elements) into heap memory.
            // Reader can freely mutate/delete its copy without racing the writer.
            const int p = published_index.load(std::memory_order_acquire);
            T* read_slot = new T[size];
            // memcpy is only safe for trivially copyable T.
            // copy_n keeps std::string / vector / map element copies valid.
            std::copy_n(buffer[p], static_cast<size_t>(size), read_slot);
            return read_slot;
        }

        // Zero-copy read: pointer is valid until the next publish.
        // Prefer get_read_cpy() if the consumer might overlap with publish().
        T* get_read_slot() {
            const int p = published_index.load(std::memory_order_acquire);
            return buffer[p];
        }

        int generation() const {
            return count.load(std::memory_order_relaxed);
        }

        T& operator[](size_t index) {
            auto* w_slot = this->get_write_slot();
            return w_slot[index];
        }

        int capacity() const {
            return this->size;
        }
    };

#pragma endregion
}