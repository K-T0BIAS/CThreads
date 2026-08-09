// Copyright (c) 2026 Tobias Karusseit
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <random>

namespace cthreads::math {

// Per-Job / per-OS-thread engine — safe under parallel cthreads::spawn.
inline std::mt19937& engine() {
    thread_local std::mt19937 gen{std::random_device{}()};
    return gen;
}

inline void seed(unsigned int s) {
    engine().seed(s);
}

// Python-like random(): float in [0.0, 1.0).
inline double random() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(engine());
}

inline double uniform(double lo, double hi) {
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(engine());
}

inline int randint(int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(engine());
}

}  // namespace cthreads::math
