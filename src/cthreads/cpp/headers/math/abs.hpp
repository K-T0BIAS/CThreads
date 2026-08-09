#pragma once

#include <cmath>
#include <cstdlib>

namespace cthreads::math {

inline int abs(int value) { return std::abs(value); }
inline long abs(long value) { return std::abs(value); }
inline long long abs(long long value) { return std::abs(value); }

inline float abs(float value) { return std::fabs(value); }
inline double abs(double value) { return std::fabs(value); }
inline long double abs(long double value) { return std::fabs(value); }

}  // namespace cthreads::math
