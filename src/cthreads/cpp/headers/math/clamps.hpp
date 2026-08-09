#pragma once

#include <algorithm>

namespace cthreads::math {

template <typename T>
T clamp(T value, T lo, T hi) {
    return std::clamp(value, lo, hi);
}

template <typename T>
T min(T a, T b) {
    return std::min(a, b);
}

template <typename T>
T max(T a, T b) {
    return std::max(a, b);
}

}  // namespace cthreads::math
