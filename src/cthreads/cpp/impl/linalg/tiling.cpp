#include "../../headers/linalg/tiling.hpp"
#include "../../headers/linalg/shape.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace cthreads::linalg {

namespace {

constexpr size_t kFlatTileSize = 4096;

template<typename T>
std::vector<Tile<T>> tiles_flat(T* data, size_t numel) {
    if (data == nullptr || numel == 0) {
        return {};
    }
    const size_t n = (numel + kFlatTileSize - 1) / kFlatTileSize;
    std::vector<Tile<T>> out(n);
    for (size_t i = 0; i < n; ++i) {
        const size_t offset = i * kFlatTileSize;
        out[i].data = data + offset;
        out[i].numel = std::min(kFlatTileSize, numel - offset);
    }
    return out;
}

template<typename T>
std::vector<Tile<T>> tiles_vectors(T* data, size_t storage_numel, const Shape& shape) {
    if (data == nullptr || shape.ndim() == 0) {
        return {};
    }
    const size_t expected = shape.numel();
    if (storage_numel < expected) {
        throw std::runtime_error("tiles: buffer shorter than shape.numel()");
    }

    // Last axis = contiguous vector (row-major).
    const size_t vec_len = shape[shape.ndim() - 1];
    if (vec_len == 0) {
        return {};
    }
    const size_t n_vecs = expected / vec_len;

    std::vector<Tile<T>> out(n_vecs);
    for (size_t i = 0; i < n_vecs; ++i) {
        out[i].data = data + i * vec_len;
        out[i].numel = vec_len;
    }
    return out;
}

} // namespace

template<typename T>
std::vector<Tile<T>> Tile<T>::tiles(const std::vector<T>& data) {
    return tiles_flat(const_cast<T*>(data.data()), data.size());
}

template<typename T>
std::vector<Tile<T>> Tile<T>::tiles(const T* data, size_t numel) {
    return tiles_flat(const_cast<T*>(data), numel);
}

template<typename T>
std::vector<Tile<T>> Tile<T>::tiles(const std::vector<T>& data, const Shape& shape) {
    return tiles_vectors(const_cast<T*>(data.data()), data.size(), shape);
}

template<typename T>
std::vector<Tile<T>> Tile<T>::tiles(T* data, const Shape& shape) {
    return tiles_vectors(data, shape.numel(), shape);
}

// Explicit instantiations - keep Tile<> usable from other TUs without the impl.
template struct Tile<float>;
template struct Tile<double>;
template struct Tile<int>;
template struct Tile<uint8_t>;

} // namespace cthreads::linalg
