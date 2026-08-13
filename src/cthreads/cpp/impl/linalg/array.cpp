#include "../../headers/linalg/array.hpp"
#include "../../headers/linalg/shape.hpp"
#include "../../headers/linalg/tiling.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace cthreads::linalg {

#pragma region math operations

#pragma region fast math operations

template<typename T>
T Array<T>::_fast_inner_product_scalar(const Array& lhs, const Array& rhs) {
    if (lhs.shape().numel() != rhs.shape().numel()) {
        throw std::runtime_error("inner_product: numel mismatch");
    }
    const T* a = lhs.data();
    const T* b = rhs.data();
    const size_t n = lhs.shape().numel();
    T result = 0;
    for (size_t i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

template<typename T>
T Array<T>::_fast_inner_product(const Array& lhs, const Array& rhs) {
    // NOTE: specialized arrays like vec3/vec4 should override with a constexpr version
    if (lhs.shape().numel() != rhs.shape().numel()) {
        throw std::runtime_error("inner_product: numel mismatch");
    }
    const T* a = lhs.data();
    const T* b = rhs.data();
    const size_t n = lhs.shape().numel();

#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        // AVX2 float: both vectors contiguous — loadu + fmadd along the length
        __m256 acc = _mm256_setzero_ps(); // initialize accumulator to zero (float32)
        size_t i = 0;
        for (; i + 8 <= n; i += 8) { // itter over 8 elements at a time
            __m256 va = _mm256_loadu_ps(a + i); // load 8 elements from a
            __m256 vb = _mm256_loadu_ps(b + i); // load 8 elements from b
            acc = _mm256_fmadd_ps(va, vb, acc); // acc += a * b
        }
        // horizontal sum of 8 lanes
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_add_ps(s, _mm_movehdup_ps(s));
        s = _mm_add_ss(s, _mm_movehl_ps(s, s));
        float sum = _mm_cvtss_f32(s);
        for (; i < n; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    } else if constexpr (std::is_same_v<T, double>) {
        // AVX2 double: both vectors contiguous — loadu + fmadd along the length
        __m256d acc = _mm256_setzero_pd(); // initialize accumulator to zero (float64)
        size_t i = 0;
        for (; i + 4 <= n; i += 4) { // itter over 4 elements at a time
            __m256d va = _mm256_loadu_pd(a + i); // load 4 elements from a
            __m256d vb = _mm256_loadu_pd(b + i); // load 4 elements from b
            acc = _mm256_fmadd_pd(va, vb, acc); // acc += a * b
        }
        // horizontal sum of 4 lanes
        __m128d lo = _mm256_castpd256_pd128(acc);
        __m128d hi = _mm256_extractf128_pd(acc, 1);
        __m128d s = _mm_add_pd(lo, hi);
        s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
        double sum = _mm_cvtsd_f64(s);
        for (; i < n; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    } else {
        // fallback: unsupported T under AVX2 build
        return _fast_inner_product_scalar(lhs, rhs);
    }
#else
    // fallback: no AVX2
    return _fast_inner_product_scalar(lhs, rhs);
#endif
}

template<typename T>
void Array<T>::_fast_cross_product(Array& lhs, const Array& rhs) {
    (void)lhs;
    (void)rhs;
    throw std::runtime_error("cross_product: not implemented");
}

template<typename T>
void Array<T>::_fast_matmul_scalar(Array& out, const Array& lhs, const Array& rhs) {
    // Dense row-major: lhs (M,K) @ rhs (K,N) -> out (M,N)
    if (lhs.ndim() != 2 || rhs.ndim() != 2 || out.ndim() != 2) {
        throw std::runtime_error("matmul: expected 2D arrays");
    }
    const size_t M = lhs.shape()[0];
    const size_t K = lhs.shape()[1];
    const size_t K2 = rhs.shape()[0];
    const size_t N = rhs.shape()[1];
    if (K != K2 || out.shape()[0] != M || out.shape()[1] != N) {
        throw std::runtime_error("matmul: incompatible shapes");
    }

    auto a_rows = Tile<T>::tiles(const_cast<T*>(lhs.data()), lhs.shape());
    T* c = out.data();
    const T* b = rhs.data();

    for (size_t i = 0; i < M; ++i) {
        const T* a_row = a_rows[i].data;
        for (size_t j = 0; j < N; ++j) {
            T sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += a_row[k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

template<typename T>
void Array<T>::_fast_matmul(Array& out, const Array& lhs, const Array& rhs) {
    // Dense row-major: lhs (M,K) @ rhs (K,N) -> out (M,N)
    if (lhs.ndim() != 2 || rhs.ndim() != 2 || out.ndim() != 2) {
        throw std::runtime_error("matmul: expected 2D arrays");
    }
    const size_t M = lhs.shape()[0];
    const size_t K = lhs.shape()[1];
    const size_t K2 = rhs.shape()[0];
    const size_t N = rhs.shape()[1];
    if (K != K2 || out.shape()[0] != M || out.shape()[1] != N) {
        throw std::runtime_error("matmul: incompatible shapes");
    }

    auto a_rows = Tile<T>::tiles(const_cast<T*>(lhs.data()), lhs.shape());
    T* c = out.data();
    const T* b = rhs.data();

#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        // AVX2 float: vectorize along K; A row contiguous, B column via gather.
        for (size_t i = 0; i < M; ++i) {  // row
            const float* a_row = a_rows[i].data;
            for (size_t j = 0; j < N; ++j) { // column
                __m256 acc = _mm256_setzero_ps(); // initialize accumulator to zero (float32)
                size_t k = 0;
                for (; k + 8 <= K; k += 8) { // itter over 8 elements at a time
                    __m256 a = _mm256_loadu_ps(a_row + k); // load 8 elements from a_row
                    // B[k..k+7, j] at b[(k+t)*N + j] — gather with stride N
                    __m256i vindex = _mm256_setr_epi32( // set up index vector for gather (ugly bc it needs to be every Nth element where N is the col)
                        0, (int)N, (int)(2 * N), (int)(3 * N),
                        (int)(4 * N), (int)(5 * N), (int)(6 * N), (int)(7 * N));
                    __m256 bvec = _mm256_i32gather_ps(b + k * N + j, vindex, 4); // gather 8 elements from b into the 256 bit register using the indexing vec from above
                    acc = _mm256_fmadd_ps(a, bvec, acc); // acc += a * b
                }
                // horizontal sum of 8 lanes
                __m128 lo = _mm256_castps256_ps128(acc);
                __m128 hi = _mm256_extractf128_ps(acc, 1);
                __m128 s = _mm_add_ps(lo, hi);
                s = _mm_add_ps(s, _mm_movehdup_ps(s));
                s = _mm_add_ss(s, _mm_movehl_ps(s, s));
                float sum = _mm_cvtss_f32(s);
                for (; k < K; ++k) {
                    sum += a_row[k] * b[k * N + j];
                }
                c[i * N + j] = sum;
            }
        }
    } else if constexpr (std::is_same_v<T, double>) {
        // AVX2 double: vectorize along K; A row contiguous, B column via gather.
        for (size_t i = 0; i < M; ++i) {  // row
            const double* a_row = a_rows[i].data;
            for (size_t j = 0; j < N; ++j) { // column
                __m256d acc = _mm256_setzero_pd(); // initialize accumulator to zero (float64)
                size_t k = 0;
                for (; k + 4 <= K; k += 4) { // itter over 4 elements at a time
                    __m256d a = _mm256_loadu_pd(a_row + k); // load 4 elements from a_row
                    // B[k..k+3, j] at b[(k+t)*N + j] — gather with stride N
                    __m128i vindex = _mm_setr_epi32( // set up index vector for gather (ugly bc it needs to be every Nth element where N is the col)
                        0, (int)N, (int)(2 * N), (int)(3 * N));
                    __m256d bvec = _mm256_i32gather_pd(b + k * N + j, vindex, 8); // gather 4 elements from b into the 256 bit register using the indexing vec from above
                    acc = _mm256_fmadd_pd(a, bvec, acc); // acc += a * b
                }
                // horizontal sum of 4 lanes
                __m128d lo = _mm256_castpd256_pd128(acc);
                __m128d hi = _mm256_extractf128_pd(acc, 1);
                __m128d s = _mm_add_pd(lo, hi);
                s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
                double sum = _mm_cvtsd_f64(s);
                for (; k < K; ++k) {
                    sum += a_row[k] * b[k * N + j];
                }
                c[i * N + j] = sum;
            }
        }
    } else {
        // fallback: unsupported T under AVX2 build
        _fast_matmul_scalar(out, lhs, rhs);
    }
#else
    // fallback: no AVX2
    _fast_matmul_scalar(out, lhs, rhs);
#endif
}

template<typename T>
void Array<T>::_fast_add(Array& lhs, const Array& rhs) {
    if (lhs.shape() != rhs.shape()) {
        throw std::runtime_error("add: shape mismatch");
    }
    T* a = lhs.data();
    const T* b = rhs.data();
    const size_t n = lhs.shape().numel();
    for (size_t i = 0; i < n; ++i) {
        a[i] += b[i];
    }
}

template<typename T>
void Array<T>::_fast_sub(Array& lhs, const Array& rhs) {
    if (lhs.shape() != rhs.shape()) {
        throw std::runtime_error("sub: shape mismatch");
    }
    T* a = lhs.data();
    const T* b = rhs.data();
    const size_t n = lhs.shape().numel();
    for (size_t i = 0; i < n; ++i) {
        a[i] -= b[i];
    }
}

template<typename T>
void Array<T>::_fast_mul(Array& lhs, const Array& rhs) {
    if (lhs.shape() != rhs.shape()) {
        throw std::runtime_error("mul: shape mismatch");
    }
    T* a = lhs.data();
    const T* b = rhs.data();
    const size_t n = lhs.shape().numel();
    for (size_t i = 0; i < n; ++i) {
        a[i] *= b[i];
    }
}

template<typename T>
void Array<T>::_fast_div(Array& lhs, const Array& rhs) {
    if (lhs.shape() != rhs.shape()) {
        throw std::runtime_error("div: shape mismatch");
    }
    T* a = lhs.data();
    const T* b = rhs.data();
    const size_t n = lhs.shape().numel();
    for (size_t i = 0; i < n; ++i) {
        a[i] /= b[i];
    }
}

template<typename T>
void Array<T>::_fast_neg(Array& lhs) {
    T* a = lhs.data();
    const size_t n = lhs.shape().numel();
    for (size_t i = 0; i < n; ++i) {
        a[i] = -a[i];
    }
}

#pragma endregion fast math operations

#pragma region public math API

template<typename T>
Array<T> Array<T>::matmul(const Array& other) const {
    if (this->ndim() != 2 || other.ndim() != 2) {
        throw std::runtime_error("matmul: expected 2D arrays");
    }
    const size_t M = this->_shape[0];
    const size_t K = this->_shape[1];
    const size_t K2 = other._shape[0];
    const size_t N = other._shape[1];
    if (K != K2) {
        throw std::runtime_error("matmul: inner dimensions must match (K)");
    }
    Array<T> out(Shape(std::vector<size_t>{M, N}));
    _fast_matmul(out, *this, other);
    return out;
}

template<typename T>
Array<T> Array<T>::matmul_scalar(const Array& other) const {
    if (this->ndim() != 2 || other.ndim() != 2) {
        throw std::runtime_error("matmul: expected 2D arrays");
    }
    const size_t M = this->_shape[0];
    const size_t K = this->_shape[1];
    const size_t K2 = other._shape[0];
    const size_t N = other._shape[1];
    if (K != K2) {
        throw std::runtime_error("matmul: inner dimensions must match (K)");
    }
    Array<T> out(Shape(std::vector<size_t>{M, N}));
    _fast_matmul_scalar(out, *this, other);
    return out;
}

template<typename T>
void Array<T>::_matmul(const Array& other) {
    *this = this->matmul(other);
}

template<typename T>
Array<T> Array<T>::dot(const Array& other) const {
    T s = _fast_inner_product(*this, other);
    Array<T> out(Shape(size_t{1}));
    out[0] = s;
    return out;
}

template<typename T>
Array<T> Array<T>::dot_scalar(const Array& other) const {
    T s = _fast_inner_product_scalar(*this, other);
    Array<T> out(Shape(size_t{1}));
    out[0] = s;
    return out;
}

template<typename T>
void Array<T>::_dot(const Array& other) {
    T s = _fast_inner_product(*this, other);
    *this = Array<T>(Shape(size_t{1}));
    (*this)[0] = s;
}

template<typename T>
Array<T> Array<T>::cross(const Array& other) const {
    Array<T> out = *this;
    _fast_cross_product(out, other);
    return out;
}

template<typename T>
void Array<T>::_cross(const Array& other) {
    _fast_cross_product(*this, other);
}

template<typename T>
Array<T> Array<T>::operator+(const Array& other) const {
    Array<T> out = *this;
    _fast_add(out, other);
    return out;
}

template<typename T>
Array<T> Array<T>::operator-(const Array& other) const {
    Array<T> out = *this;
    _fast_sub(out, other);
    return out;
}

template<typename T>
Array<T> Array<T>::operator*(const Array& other) const {
    Array<T> out = *this;
    _fast_mul(out, other);
    return out;
}

template<typename T>
Array<T> Array<T>::operator/(const Array& other) const {
    Array<T> out = *this;
    _fast_div(out, other);
    return out;
}

template<typename T>
Array<T> Array<T>::operator+(T value) const {
    Array<T> out = *this;
    T* p = out.data();
    const size_t n = out.shape().numel();
    for (size_t i = 0; i < n; ++i) {
        p[i] += value;
    }
    return out;
}

template<typename T>
Array<T> Array<T>::operator-(T value) const {
    Array<T> out = *this;
    T* p = out.data();
    const size_t n = out.shape().numel();
    for (size_t i = 0; i < n; ++i) {
        p[i] -= value;
    }
    return out;
}

template<typename T>
Array<T> Array<T>::operator*(T value) const {
    Array<T> out = *this;
    T* p = out.data();
    const size_t n = out.shape().numel();
    for (size_t i = 0; i < n; ++i) {
        p[i] *= value;
    }
    return out;
}

template<typename T>
Array<T> Array<T>::operator/(T value) const {
    Array<T> out = *this;
    T* p = out.data();
    const size_t n = out.shape().numel();
    for (size_t i = 0; i < n; ++i) {
        p[i] /= value;
    }
    return out;
}

template<typename T>
Array<T> Array<T>::operator-() const {
    Array<T> out = *this;
    _fast_neg(out);
    return out;
}

template<typename T>
void Array<T>::_add(const Array& other) { _fast_add(*this, other); }
template<typename T>
void Array<T>::_sub(const Array& other) { _fast_sub(*this, other); }
template<typename T>
void Array<T>::_mul(const Array& other) { _fast_mul(*this, other); }
template<typename T>
void Array<T>::_div(const Array& other) { _fast_div(*this, other); }

template<typename T>
void Array<T>::_add(T value) {
    T* p = this->data();
    const size_t n = this->shape().numel();
    for (size_t i = 0; i < n; ++i) {
        p[i] += value;
    }
}
template<typename T>
void Array<T>::_sub(T value) {
    T* p = this->data();
    const size_t n = this->shape().numel();
    for (size_t i = 0; i < n; ++i) {
        p[i] -= value;
    }
}
template<typename T>
void Array<T>::_mul(T value) {
    T* p = this->data();
    const size_t n = this->shape().numel();
    for (size_t i = 0; i < n; ++i) {
        p[i] *= value;
    }
}
template<typename T>
void Array<T>::_div(T value) {
    T* p = this->data();
    const size_t n = this->shape().numel();
    for (size_t i = 0; i < n; ++i) {
        p[i] /= value;
    }
}
template<typename T>
void Array<T>::_neg() { _fast_neg(*this); }

#pragma endregion public math API

#pragma region stubs (shape / view API)

template<typename T>
Array<T> Array<T>::view(const Shape&) const {
    throw std::runtime_error("Array::view not implemented");
}
template<typename T>
Array<T> Array<T>::reshape(const Shape&) const {
    throw std::runtime_error("Array::reshape not implemented");
}
template<typename T>
Array<T> Array<T>::flatten() const {
    throw std::runtime_error("Array::flatten not implemented");
}
template<typename T>
Array<T> Array<T>::transpose() const {
    throw std::runtime_error("Array::transpose not implemented");
}
template<typename T>
Array<T> Array<T>::permute(const Shape&) const {
    throw std::runtime_error("Array::permute not implemented");
}
template<typename T>
Array<T> Array<T>::squeeze(const Shape&) const {
    throw std::runtime_error("Array::squeeze not implemented");
}
template<typename T>
Array<T> Array<T>::unsqueeze(const Shape&) const {
    throw std::runtime_error("Array::unsqueeze not implemented");
}
template<typename T>
void Array<T>::_view(const Shape&) {
    throw std::runtime_error("Array::_view not implemented");
}
template<typename T>
void Array<T>::_reshape(const Shape&) {
    throw std::runtime_error("Array::_reshape not implemented");
}
template<typename T>
void Array<T>::_flatten() {
    throw std::runtime_error("Array::_flatten not implemented");
}
template<typename T>
void Array<T>::_transpose() {
    throw std::runtime_error("Array::_transpose not implemented");
}
template<typename T>
void Array<T>::_permute(const Shape&) {
    throw std::runtime_error("Array::_permute not implemented");
}
template<typename T>
void Array<T>::_squeeze(const Shape&) {
    throw std::runtime_error("Array::_squeeze not implemented");
}
template<typename T>
void Array<T>::_unsqueeze(const Shape&) {
    throw std::runtime_error("Array::_unsqueeze not implemented");
}

#pragma endregion stubs

#pragma endregion math operations

template class Array<float>;
template class Array<double>;
template class Array<int>;

} // namespace cthreads::linalg
