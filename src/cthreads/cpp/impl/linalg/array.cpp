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

namespace {

// Batched GEMM layout: A (..., M, K) @ B (..., K, N) -> C (..., M, N).
// Leading dims must match, or one side is plain 2D and is broadcast.
struct MatmulSpec {
    size_t M = 0;
    size_t K = 0;
    size_t N = 0;
    size_t batch = 0;
    size_t a_stride = 0; // 0 => broadcast same A every batch
    size_t b_stride = 0; // 0 => broadcast same B every batch
    size_t c_stride = 0;
    Shape out_shape{std::vector<size_t>{}};
};

MatmulSpec resolve_matmul(const Shape& a, const Shape& b) {
    if (a.ndim() < 2 || b.ndim() < 2) {
        throw std::runtime_error("matmul: expected rank >= 2");
    }
    const size_t M = a[a.ndim() - 2];
    const size_t K = a[a.ndim() - 1];
    const size_t K2 = b[b.ndim() - 2];
    const size_t N = b[b.ndim() - 1];
    if (K != K2) {
        throw std::runtime_error("matmul: inner dimensions must match (K)");
    }

    MatmulSpec spec;
    spec.M = M;
    spec.K = K;
    spec.N = N;
    spec.c_stride = M * N;

    const size_t a_matrix = M * K;
    const size_t b_matrix = K * N;

    if (a.ndim() == 2 && b.ndim() == 2) {
        spec.batch = 1;
        spec.a_stride = a_matrix;
        spec.b_stride = b_matrix;
        spec.out_shape = Shape(std::vector<size_t>{M, N});
        return spec;
    }

    if (a.ndim() == 2) {
        // Broadcast A over B's leading dims.
        std::vector<size_t> out_dims;
        out_dims.reserve(b.ndim());
        size_t batch = 1;
        for (size_t i = 0; i + 2 < b.ndim(); ++i) {
            out_dims.push_back(b[i]);
            batch *= b[i];
        }
        out_dims.push_back(M);
        out_dims.push_back(N);
        spec.batch = batch;
        spec.a_stride = 0;
        spec.b_stride = b_matrix;
        spec.out_shape = Shape(out_dims);
        return spec;
    }

    if (b.ndim() == 2) {
        // Broadcast B over A's leading dims.
        std::vector<size_t> out_dims;
        out_dims.reserve(a.ndim());
        size_t batch = 1;
        for (size_t i = 0; i + 2 < a.ndim(); ++i) {
            out_dims.push_back(a[i]);
            batch *= a[i];
        }
        out_dims.push_back(M);
        out_dims.push_back(N);
        spec.batch = batch;
        spec.a_stride = a_matrix;
        spec.b_stride = 0;
        spec.out_shape = Shape(out_dims);
        return spec;
    }

    if (a.ndim() != b.ndim()) {
        throw std::runtime_error("matmul: leading ranks differ (broadcast only plain 2D)");
    }
    std::vector<size_t> out_dims;
    out_dims.reserve(a.ndim());
    size_t batch = 1;
    for (size_t i = 0; i + 2 < a.ndim(); ++i) {
        if (a[i] != b[i]) {
            throw std::runtime_error("matmul: leading dimensions must match");
        }
        out_dims.push_back(a[i]);
        batch *= a[i];
    }
    out_dims.push_back(M);
    out_dims.push_back(N);
    spec.batch = batch;
    spec.a_stride = a_matrix;
    spec.b_stride = b_matrix;
    spec.out_shape = Shape(out_dims);
    return spec;
}

} // namespace

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
    // Last axis must be 3; leading dims must match. In-place into lhs.
    if (lhs.ndim() < 1 || rhs.ndim() < 1) {
        throw std::runtime_error("cross: expected rank >= 1");
    }
    if (lhs.shape() != rhs.shape()) {
        throw std::runtime_error("cross: shape mismatch");
    }
    if (lhs.shape()[lhs.ndim() - 1] != 3) {
        throw std::runtime_error("cross: last dimension must be 3");
    }
    const size_t n_vecs = lhs.shape().numel() / 3;
    T* a = lhs.data();
    const T* b = rhs.data();
    for (size_t i = 0; i < n_vecs; ++i) {
        const T* ai = a + i * 3;
        const T* bi = b + i * 3;
        const T x = ai[1] * bi[2] - ai[2] * bi[1];
        const T y = ai[2] * bi[0] - ai[0] * bi[2];
        const T z = ai[0] * bi[1] - ai[1] * bi[0];
        a[i * 3 + 0] = x;
        a[i * 3 + 1] = y;
        a[i * 3 + 2] = z;
    }
}

template<typename T>
void Array<T>::_fast_matmul_scalar(Array& out, const Array& lhs, const Array& rhs) {
    // Dense row-major batched: lhs (..., M, K) @ rhs (..., K, N) -> out (..., M, N)
    const MatmulSpec spec = resolve_matmul(lhs.shape(), rhs.shape());
    if (out.shape() != spec.out_shape) {
        throw std::runtime_error("matmul: incompatible output shape");
    }

    const size_t M = spec.M;
    const size_t K = spec.K;
    const size_t N = spec.N;
    const Shape row_shape(std::vector<size_t>{M, K});

    for (size_t bi = 0; bi < spec.batch; ++bi) {
        const T* a = lhs.data() + bi * spec.a_stride;
        const T* b = rhs.data() + bi * spec.b_stride;
        T* c = out.data() + bi * spec.c_stride;
        auto a_rows = Tile<T>::tiles(const_cast<T*>(a), row_shape);
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
}

template<typename T>
void Array<T>::_fast_matmul(Array& out, const Array& lhs, const Array& rhs) {
    // Dense row-major batched: lhs (..., M, K) @ rhs (..., K, N) -> out (..., M, N)
    const MatmulSpec spec = resolve_matmul(lhs.shape(), rhs.shape());
    if (out.shape() != spec.out_shape) {
        throw std::runtime_error("matmul: incompatible output shape");
    }

    const size_t M = spec.M;
    const size_t K = spec.K;
    const size_t N = spec.N;
    const Shape row_shape(std::vector<size_t>{M, K});

#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        std::vector<float> b_cols(K * N);
        const bool pack_once = (spec.b_stride == 0);
        if (pack_once) {
            const float* b0 = rhs.data();
            for (size_t j = 0; j < N; ++j) {
                float* col = b_cols.data() + j * K;
                for (size_t k = 0; k < K; ++k) {
                    col[k] = b0[k * N + j];
                }
            }
        }
        for (size_t bi = 0; bi < spec.batch; ++bi) {
            const float* a = lhs.data() + bi * spec.a_stride;
            const float* b = rhs.data() + bi * spec.b_stride;
            float* c = out.data() + bi * spec.c_stride;
            if (!pack_once) {
                for (size_t j = 0; j < N; ++j) {
                    float* col = b_cols.data() + j * K;
                    for (size_t k = 0; k < K; ++k) {
                        col[k] = b[k * N + j];
                    }
                }
            }
            auto a_rows = Tile<float>::tiles(const_cast<float*>(a), row_shape);
            for (size_t i = 0; i < M; ++i) {  // row
                const float* a_row = a_rows[i].data;
                for (size_t j = 0; j < N; ++j) { // column
                    const float* b_col = b_cols.data() + j * K;
                    __m256 acc = _mm256_setzero_ps(); // initialize accumulator to zero (float32)
                    size_t k = 0;
                    for (; k + 8 <= K; k += 8) { // itter over 8 elements at a time
                        __m256 av = _mm256_loadu_ps(a_row + k); // load 8 elements from a_row
                        __m256 bvec = _mm256_loadu_ps(b_col + k); // load 8 elements from packed B col
                        acc = _mm256_fmadd_ps(av, bvec, acc); // acc += a * b
                    }
                    // horizontal sum of 8 lanes
                    __m128 lo = _mm256_castps256_ps128(acc);
                    __m128 hi = _mm256_extractf128_ps(acc, 1);
                    __m128 s = _mm_add_ps(lo, hi);
                    s = _mm_add_ps(s, _mm_movehdup_ps(s));
                    s = _mm_add_ss(s, _mm_movehl_ps(s, s));
                    float sum = _mm_cvtss_f32(s);
                    for (; k < K; ++k) {
                        sum += a_row[k] * b_col[k];
                    }
                    c[i * N + j] = sum;
                }
            }
        }
    } else if constexpr (std::is_same_v<T, double>) {
        std::vector<double> b_cols(K * N);
        const bool pack_once = (spec.b_stride == 0);
        if (pack_once) {
            const double* b0 = rhs.data();
            for (size_t j = 0; j < N; ++j) {
                double* col = b_cols.data() + j * K;
                for (size_t k = 0; k < K; ++k) {
                    col[k] = b0[k * N + j];
                }
            }
        }
        for (size_t bi = 0; bi < spec.batch; ++bi) {
            const double* a = lhs.data() + bi * spec.a_stride;
            const double* b = rhs.data() + bi * spec.b_stride;
            double* c = out.data() + bi * spec.c_stride;
            if (!pack_once) {
                for (size_t j = 0; j < N; ++j) {
                    double* col = b_cols.data() + j * K;
                    for (size_t k = 0; k < K; ++k) {
                        col[k] = b[k * N + j];
                    }
                }
            }
            auto a_rows = Tile<double>::tiles(const_cast<double*>(a), row_shape);
            for (size_t i = 0; i < M; ++i) {  // row
                const double* a_row = a_rows[i].data;
                for (size_t j = 0; j < N; ++j) { // column
                    const double* b_col = b_cols.data() + j * K;
                    __m256d acc = _mm256_setzero_pd(); // initialize accumulator to zero (float64)
                    size_t k = 0;
                    for (; k + 4 <= K; k += 4) { // itter over 4 elements at a time
                        __m256d av = _mm256_loadu_pd(a_row + k); // load 4 elements from a_row
                        __m256d bvec = _mm256_loadu_pd(b_col + k); // load 4 elements from packed B col
                        acc = _mm256_fmadd_pd(av, bvec, acc); // acc += a * b
                    }
                    // horizontal sum of 4 lanes
                    __m128d lo = _mm256_castpd256_pd128(acc);
                    __m128d hi = _mm256_extractf128_pd(acc, 1);
                    __m128d s = _mm_add_pd(lo, hi);
                    s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
                    double sum = _mm_cvtsd_f64(s);
                    for (; k < K; ++k) {
                        sum += a_row[k] * b_col[k];
                    }
                    c[i * N + j] = sum;
                }
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
    if (rhs.shape().numel() == 1 && lhs.shape().numel() != 1) {
        _fast_add(lhs, rhs[0]);
        return;
    }
    if (lhs.shape() != rhs.shape()) {
        throw std::runtime_error("add: shape mismatch");
    }
    T* a = lhs.data();
    const T* b = rhs.data();
    const size_t n = lhs.shape().numel();

#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(a + i, _mm256_add_ps(va, vb));
        }
        for (; i < n; ++i) {
            a[i] += b[i];
        }
    } else if constexpr (std::is_same_v<T, double>) {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            __m256d vb = _mm256_loadu_pd(b + i);
            _mm256_storeu_pd(a + i, _mm256_add_pd(va, vb));
        }
        for (; i < n; ++i) {
            a[i] += b[i];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            a[i] += b[i];
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] += b[i];
    }
#endif
}

template<typename T>
void Array<T>::_fast_sub(Array& lhs, const Array& rhs) {
    if (rhs.shape().numel() == 1 && lhs.shape().numel() != 1) {
        _fast_sub(lhs, rhs[0]);
        return;
    }
    if (lhs.shape() != rhs.shape()) {
        throw std::runtime_error("sub: shape mismatch");
    }
    T* a = lhs.data();
    const T* b = rhs.data();
    const size_t n = lhs.shape().numel();

#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(a + i, _mm256_sub_ps(va, vb));
        }
        for (; i < n; ++i) {
            a[i] -= b[i];
        }
    } else if constexpr (std::is_same_v<T, double>) {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            __m256d vb = _mm256_loadu_pd(b + i);
            _mm256_storeu_pd(a + i, _mm256_sub_pd(va, vb));
        }
        for (; i < n; ++i) {
            a[i] -= b[i];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            a[i] -= b[i];
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] -= b[i];
    }
#endif
}

template<typename T>
void Array<T>::_fast_mul(Array& lhs, const Array& rhs) {
    if (rhs.shape().numel() == 1 && lhs.shape().numel() != 1) {
        _fast_mul(lhs, rhs[0]);
        return;
    }
    if (lhs.shape() != rhs.shape()) {
        throw std::runtime_error("mul: shape mismatch");
    }
    T* a = lhs.data();
    const T* b = rhs.data();
    const size_t n = lhs.shape().numel();

#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(a + i, _mm256_mul_ps(va, vb));
        }
        for (; i < n; ++i) {
            a[i] *= b[i];
        }
    } else if constexpr (std::is_same_v<T, double>) {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            __m256d vb = _mm256_loadu_pd(b + i);
            _mm256_storeu_pd(a + i, _mm256_mul_pd(va, vb));
        }
        for (; i < n; ++i) {
            a[i] *= b[i];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            a[i] *= b[i];
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] *= b[i];
    }
#endif
}

template<typename T>
void Array<T>::_fast_div(Array& lhs, const Array& rhs) {
    if (rhs.shape().numel() == 1 && lhs.shape().numel() != 1) {
        _fast_div(lhs, rhs[0]);
        return;
    }
    if (lhs.shape() != rhs.shape()) {
        throw std::runtime_error("div: shape mismatch");
    }
    T* a = lhs.data();
    const T* b = rhs.data();
    const size_t n = lhs.shape().numel();

#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(a + i, _mm256_div_ps(va, vb));
        }
        for (; i < n; ++i) {
            a[i] /= b[i];
        }
    } else if constexpr (std::is_same_v<T, double>) {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            __m256d vb = _mm256_loadu_pd(b + i);
            _mm256_storeu_pd(a + i, _mm256_div_pd(va, vb));
        }
        for (; i < n; ++i) {
            a[i] /= b[i];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            a[i] /= b[i];
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] /= b[i];
    }
#endif
}

template<typename T>
void Array<T>::_fast_neg(Array& lhs) {
    T* a = lhs.data();
    const size_t n = lhs.shape().numel();

#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        const __m256 z = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            _mm256_storeu_ps(a + i, _mm256_sub_ps(z, va));
        }
        for (; i < n; ++i) {
            a[i] = -a[i];
        }
    } else if constexpr (std::is_same_v<T, double>) {
        const __m256d z = _mm256_setzero_pd();
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            _mm256_storeu_pd(a + i, _mm256_sub_pd(z, va));
        }
        for (; i < n; ++i) {
            a[i] = -a[i];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            a[i] = -a[i];
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] = -a[i];
    }
#endif
}

template<typename T>
void Array<T>::_fast_add(Array& lhs, T value) {
    T* a = lhs.data();
    const size_t n = lhs.shape().numel();
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        const __m256 vb = _mm256_set1_ps(value);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            _mm256_storeu_ps(a + i, _mm256_add_ps(va, vb));
        }
        for (; i < n; ++i) {
            a[i] += value;
        }
    } else if constexpr (std::is_same_v<T, double>) {
        const __m256d vb = _mm256_set1_pd(value);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            _mm256_storeu_pd(a + i, _mm256_add_pd(va, vb));
        }
        for (; i < n; ++i) {
            a[i] += value;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            a[i] += value;
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] += value;
    }
#endif
}

template<typename T>
void Array<T>::_fast_sub(Array& lhs, T value) {
    T* a = lhs.data();
    const size_t n = lhs.shape().numel();
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        const __m256 vb = _mm256_set1_ps(value);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            _mm256_storeu_ps(a + i, _mm256_sub_ps(va, vb));
        }
        for (; i < n; ++i) {
            a[i] -= value;
        }
    } else if constexpr (std::is_same_v<T, double>) {
        const __m256d vb = _mm256_set1_pd(value);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            _mm256_storeu_pd(a + i, _mm256_sub_pd(va, vb));
        }
        for (; i < n; ++i) {
            a[i] -= value;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            a[i] -= value;
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] -= value;
    }
#endif
}

template<typename T>
void Array<T>::_fast_mul(Array& lhs, T value) {
    T* a = lhs.data();
    const size_t n = lhs.shape().numel();
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        const __m256 vb = _mm256_set1_ps(value);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            _mm256_storeu_ps(a + i, _mm256_mul_ps(va, vb));
        }
        for (; i < n; ++i) {
            a[i] *= value;
        }
    } else if constexpr (std::is_same_v<T, double>) {
        const __m256d vb = _mm256_set1_pd(value);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            _mm256_storeu_pd(a + i, _mm256_mul_pd(va, vb));
        }
        for (; i < n; ++i) {
            a[i] *= value;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            a[i] *= value;
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] *= value;
    }
#endif
}

template<typename T>
void Array<T>::_fast_div(Array& lhs, T value) {
    T* a = lhs.data();
    const size_t n = lhs.shape().numel();
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        const __m256 vb = _mm256_set1_ps(value);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            _mm256_storeu_ps(a + i, _mm256_div_ps(va, vb));
        }
        for (; i < n; ++i) {
            a[i] /= value;
        }
    } else if constexpr (std::is_same_v<T, double>) {
        const __m256d vb = _mm256_set1_pd(value);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            _mm256_storeu_pd(a + i, _mm256_div_pd(va, vb));
        }
        for (; i < n; ++i) {
            a[i] /= value;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            a[i] /= value;
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] /= value;
    }
#endif
}

#pragma endregion fast math operations

#pragma region public math API

template<typename T>
Array<T> Array<T>::matmul(const Array& other) const {
    const MatmulSpec spec = resolve_matmul(this->_shape, other._shape);
    Array<T> out(spec.out_shape);
    _fast_matmul(out, *this, other);
    return out;
}

template<typename T>
Array<T> Array<T>::matmul_scalar(const Array& other) const {
    const MatmulSpec spec = resolve_matmul(this->_shape, other._shape);
    Array<T> out(spec.out_shape);
    _fast_matmul_scalar(out, *this, other);
    return out;
}

template<typename T>
void Array<T>::_matmul(const Array& other) {
    *this = this->matmul(other);
}

template<typename T>
Array<T> Array<T>::dot(const Array& other) const {
    // 1D: full inner product -> shape {1}
    // ND: contract last axis; leading dims must match -> shape leading
    if (this->ndim() == 0 || other.ndim() == 0) {
        throw std::runtime_error("dot: expected rank >= 1");
    }
    if (this->ndim() == 1 && other.ndim() == 1) {
        T s = _fast_inner_product(*this, other);
        Array<T> out{Shape(size_t{1})};
        out[0] = s;
        return out;
    }
    if (this->_shape != other._shape) {
        throw std::runtime_error("dot: shape mismatch");
    }
    const size_t K = this->_shape[this->ndim() - 1];
    std::vector<size_t> leading;
    for (size_t i = 0; i + 1 < this->ndim(); ++i) {
        leading.push_back(this->_shape[i]);
    }
    if (leading.empty()) {
        // rank-1 already handled; keep defensive
        T s = _fast_inner_product(*this, other);
        Array<T> out{Shape(size_t{1})};
        out[0] = s;
        return out;
    }
    Array<T> out{Shape(leading)};
    auto a_tiles = Tile<T>::tiles(const_cast<T*>(this->data()), this->_shape);
    auto b_tiles = Tile<T>::tiles(const_cast<T*>(other.data()), other._shape);
    for (size_t i = 0; i < a_tiles.size(); ++i) {
        // Build tiny 1D views as Arrays would allocate; inline product on tile pointers.
        const T* a = a_tiles[i].data;
        const T* b = b_tiles[i].data;
        T sum = 0;
#ifdef __AVX2__
        if constexpr (std::is_same_v<T, float>) {
            __m256 acc = _mm256_setzero_ps();
            size_t k = 0;
            for (; k + 8 <= K; k += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc);
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_add_ps(s, _mm_movehdup_ps(s));
            s = _mm_add_ss(s, _mm_movehl_ps(s, s));
            sum = _mm_cvtss_f32(s);
            for (; k < K; ++k) {
                sum += a[k] * b[k];
            }
        } else if constexpr (std::is_same_v<T, double>) {
            __m256d acc = _mm256_setzero_pd();
            size_t k = 0;
            for (; k + 4 <= K; k += 4) {
                acc = _mm256_fmadd_pd(_mm256_loadu_pd(a + k), _mm256_loadu_pd(b + k), acc);
            }
            __m128d lo = _mm256_castpd256_pd128(acc);
            __m128d hi = _mm256_extractf128_pd(acc, 1);
            __m128d s = _mm_add_pd(lo, hi);
            s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
            sum = static_cast<T>(_mm_cvtsd_f64(s));
            for (; k < K; ++k) {
                sum += a[k] * b[k];
            }
        } else {
            for (size_t k = 0; k < K; ++k) {
                sum += a[k] * b[k];
            }
        }
#else
        for (size_t k = 0; k < K; ++k) {
            sum += a[k] * b[k];
        }
#endif
        out[i] = sum;
    }
    return out;
}

template<typename T>
Array<T> Array<T>::dot_scalar(const Array& other) const {
    if (this->ndim() == 0 || other.ndim() == 0) {
        throw std::runtime_error("dot: expected rank >= 1");
    }
    if (this->ndim() == 1 && other.ndim() == 1) {
        T s = _fast_inner_product_scalar(*this, other);
        Array<T> out{Shape(size_t{1})};
        out[0] = s;
        return out;
    }
    if (this->_shape != other._shape) {
        throw std::runtime_error("dot: shape mismatch");
    }
    const size_t K = this->_shape[this->ndim() - 1];
    std::vector<size_t> leading;
    for (size_t i = 0; i + 1 < this->ndim(); ++i) {
        leading.push_back(this->_shape[i]);
    }
    Array<T> out{Shape(leading)};
    auto a_tiles = Tile<T>::tiles(const_cast<T*>(this->data()), this->_shape);
    auto b_tiles = Tile<T>::tiles(const_cast<T*>(other.data()), other._shape);
    for (size_t i = 0; i < a_tiles.size(); ++i) {
        const T* a = a_tiles[i].data;
        const T* b = b_tiles[i].data;
        T sum = 0;
        for (size_t k = 0; k < K; ++k) {
            sum += a[k] * b[k];
        }
        out[i] = sum;
    }
    return out;
}

template<typename T>
void Array<T>::_dot(const Array& other) {
    *this = this->dot(other);
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
    _fast_add(out, value);
    return out;
}

template<typename T>
Array<T> Array<T>::operator-(T value) const {
    Array<T> out = *this;
    _fast_sub(out, value);
    return out;
}

template<typename T>
Array<T> Array<T>::operator*(T value) const {
    Array<T> out = *this;
    _fast_mul(out, value);
    return out;
}

template<typename T>
Array<T> Array<T>::operator/(T value) const {
    Array<T> out = *this;
    _fast_div(out, value);
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
void Array<T>::_add(T value) { _fast_add(*this, value); }
template<typename T>
void Array<T>::_sub(T value) { _fast_sub(*this, value); }
template<typename T>
void Array<T>::_mul(T value) { _fast_mul(*this, value); }
template<typename T>
void Array<T>::_div(T value) { _fast_div(*this, value); }
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
