#pragma once

#include <vector>
#include <cstddef>

namespace cthreads::linalg {

class Shape;

// Contiguous panel into a flat row-major buffer.
// tiles(data, shape) emits one tile per last-axis vector (row for 2D).
template<typename T>
struct Tile {
    T* data;
    size_t numel;

    // Flat chunks (elementwise / packing helpers).
    static std::vector<Tile<T>> tiles(const std::vector<T>& data);
    static std::vector<Tile<T>> tiles(const T* data, size_t numel);

    // Row-major vector tiles: one tile per index of all dims except the last.
    // shape [M, K]  -> M tiles of length K
    // shape [A,B,C] -> A*B tiles of length C
    static std::vector<Tile<T>> tiles(const std::vector<T>& data, const Shape& shape);
    static std::vector<Tile<T>> tiles(T* data, const Shape& shape);
};

} // namespace cthreads::linalg
