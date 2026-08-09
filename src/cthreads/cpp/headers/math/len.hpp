#pragma once

#include <cstddef>

namespace cthreads::math {

// Prefer lowering Python len(x) -> x.size() in codegen when possible.
template <typename C>
auto len(const C& c) -> decltype(static_cast<std::size_t>(c.size())) {
    return static_cast<std::size_t>(c.size());
}

}  // namespace cthreads::math
