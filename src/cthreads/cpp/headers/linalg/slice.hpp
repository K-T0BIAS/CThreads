#pragma once

#include <cstddef>
#include <stdexcept>

namespace cthreads::linalg {

// Python-style slice: start:stop:step (stop exclusive).
// Defaults: start=0, stop=end of axis, step=1.
// v1: step >= 1, no negative indices.
class Slice {
    public:
        static constexpr size_t npos = static_cast<size_t>(-1);

        size_t start = 0;
        size_t stop = npos;
        size_t step = 1;

        Slice() = default;
        explicit Slice(size_t stop_exclusive) : stop(stop_exclusive) {}
        Slice(size_t start, size_t stop, size_t step = 1) :
            start(start),
            stop(stop),
            step(step) {}

        struct Resolved {
            size_t start;
            size_t length;
            size_t step;
        };

        Resolved resolve(size_t dim) const {
            if (this->step == 0) {
                throw std::runtime_error("slice: step must be >= 1");
            }
            const size_t s = this->start > dim ? dim : this->start;
            const size_t e_raw = (this->stop == npos) ? dim : this->stop;
            const size_t e = e_raw > dim ? dim : e_raw;
            if (e <= s) {
                return Resolved{s, 0, this->step};
            }
            const size_t length = (e - s + this->step - 1) / this->step;
            return Resolved{s, length, this->step};
        }
};

} // namespace cthreads::linalg
