#include "../../headers/linalg/array.hpp"
#include "../../headers/linalg/shape.hpp"
#include "../../headers/linalg/tiling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef __AVX2__ // if available use the avx2 intrinsics for vectorization
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

namespace cthreads::linalg {

namespace {

// Batched GEMM layout: A (..., M, K) @ B (..., K, N) -> C (..., M, N).
// Leading dims must match, or one side is plain 2D and is broadcast.
struct MatmulSpec {
    size_t M = 0; // number of rows in the left hand side matrix
    size_t K = 0; // number of columns in the left hand side matrix and number of rows in the right hand side matrix
    size_t N = 0; // number of columns in the right hand side matrix
    size_t batch = 0; // mul(dim for dim in ndim[:-2]) = number of batches
    size_t a_stride = 0; // 0 => broadcast same A every batch
    size_t b_stride = 0; // 0 => broadcast same B every batch
    size_t c_stride = 0;
    Shape out_shape{std::vector<size_t>{}}; // shape to pupulate for the result
};

/**
* Build a MatmulSpec object from the shapes of the two matrices
*
* #### args
* - a: Shape of the left hand side matrix
* - b: Shape of the right hand side matrix
*
* #### returns
* - MatmulSpec object
*
* #### throws
* - std::runtime_error: if the shapes are invalid
*/
MatmulSpec resolve_matmul(const Shape& a, const Shape& b) {
    if (a.ndim() < 2 || b.ndim() < 2) { // maticies need atleast 2 dimensions
        throw std::runtime_error("matmul: expected rank >= 2");
    }
    const size_t M = a[a.ndim() - 2]; // num rows in the lhs
    const size_t K = a[a.ndim() - 1]; // num cols in the lhs
    const size_t K2 = b[b.ndim() - 2]; // num rows in the rhs
    const size_t N = b[b.ndim() - 1]; // num cols in the rhs
    if (K != K2) { // left cols must match right rows
        throw std::runtime_error("matmul: inner dimensions must match (K)");
    }

    // initialize the spec
    MatmulSpec spec;
    spec.M = M;
    spec.K = K;
    spec.N = N;
    spec.c_stride = M * N;

    // matrix numel (excluding batch dims)
    const size_t a_matrix = M * K;
    const size_t b_matrix = K * N;


    // case: no batch dims in either array
    if (a.ndim() == 2 && b.ndim() == 2) {
        spec.batch = 1;
        spec.a_stride = a_matrix;
        spec.b_stride = b_matrix;
        spec.out_shape = Shape(std::vector<size_t>{M, N});
        return spec;
    }

    // case: lhs is 2D, rhs has batch dims
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

    // case: rhs is 2D, lhs has batch dims
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

    // case: both have batch dims, leading ranks must match
    if (a.ndim() != b.ndim()) {
        throw std::runtime_error("matmul: leading ranks differ (broadcast only plain 2D)");
    }
    // build the outshape
    std::vector<size_t> out_dims;
    out_dims.reserve(a.ndim());
    size_t batch = 1;
    for (size_t i = 0; i + 2 < a.ndim(); ++i) { // itter over the leading ranks
        if (a[i] != b[i]) { // leading ranks must match
            throw std::runtime_error("matmul: leading dimensions must match");
        }
        out_dims.push_back(a[i]);
        batch *= a[i];
    }
    // add the output dimensions (matrix dimensions)
    out_dims.push_back(M);
    out_dims.push_back(N);
    spec.batch = batch;
    spec.a_stride = a_matrix; // stride for the lhs
    spec.b_stride = b_matrix; // stride for the rhs
    spec.out_shape = Shape(out_dims); // shape for the output
    return spec;
}

// Copy logical elements (row-major order of src.shape) into a new dense C-contiguous array.
template<typename T>
Array<T> copy_to_contiguous(const Array<T>& src, const Shape& new_shape) {
    
    if (src.shape().numel() != new_shape.numel()) {
        throw std::runtime_error("copy_to_contiguous: shape numel mismatch");
    }
    // if the source is already contiguous, just return a new array with a data copy
    if (src.is_contiguous()) {
        auto arr = Array<T>(new_shape);
        std::copy(src.data(), src.data() + src.shape().numel(), arr.data());
        return arr;
    }

    Array<T> out(new_shape);
    const size_t n = new_shape.numel();
    if (n == 0) {
        return out;
    }

    // Largest trailing run whose element strides match C-order (skip size-1 axes).
    const Shape& src_shape = src.shape();
    const Shape& src_strides = src.strides();
    const size_t nd = src_shape.ndim();
    size_t segment = 1;
    size_t expected = 1;
    int last_outer = static_cast<int>(nd) - 1;
    for (; last_outer >= 0; --last_outer) {
        const size_t dim = src_shape[static_cast<size_t>(last_outer)];
        if (dim <= 1) {
            continue;
        }
        if (src_strides[static_cast<size_t>(last_outer)] != expected) {
            break;
        }
        segment *= dim;
        expected *= dim;
    }

    T* dst = out.data();
    const T* base = src.data(); // view start (offset already applied)
    std::vector<size_t> idx(nd, 0);
    for (size_t done = 0; done < n; done += segment) {
        size_t src_off = 0;
        for (size_t i = 0; i < nd; ++i) {
            src_off += idx[i] * src_strides[i];
        }
        if (segment == 1) {
            dst[done] = base[src_off];
        } else {
            std::memcpy(dst + done, base + src_off, segment * sizeof(T));
        }
        for (int d = last_outer; d >= 0; --d) {
            const size_t ud = static_cast<size_t>(d);
            ++idx[ud];
            if (idx[ud] < src_shape[ud]) {
                break;
            }
            idx[ud] = 0;
        }
    }
    return out;
}

void bump_index(std::vector<size_t>& idx, const Shape& shape) {
    if (idx.empty()) {
        return;
    }
    for (size_t d = idx.size(); d-- > 0; ) {
        ++idx[d];
        if (idx[d] < shape[d]) {
            return;
        }
        idx[d] = 0;
    }
}

void check_mask_prefix(const Shape& a, const Shape& mask) {
    if (mask.ndim() > a.ndim()) {
        throw std::runtime_error("mask: rank greater than array");
    }
    for (size_t i = 0; i < mask.ndim(); ++i) {
        if (mask[i] != a[i]) {
            throw std::runtime_error("mask: leading dimensions must match (no broadcast)");
        }
    }
}

size_t trailing_product(const Shape& sh, size_t skip) {
    size_t n = 1;
    for (size_t i = skip; i < sh.ndim(); ++i) {
        n *= sh[i];
    }
    return n;
}

Shape gathered_shape(const Shape& a, size_t mask_ndim, size_t count) {
    std::vector<size_t> dims;
    dims.reserve(a.ndim() - mask_ndim + 1);
    dims.push_back(count);
    for (size_t i = mask_ndim; i < a.ndim(); ++i) {
        dims.push_back(a[i]);
    }
    return Shape(dims);
}

inline int popcnt32(unsigned v) {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt(v));
#else
    return __builtin_popcount(v);
#endif
}

size_t count_bytes_nonzero(const uint8_t* p, size_t n) {
    size_t c = 0;
#ifdef __AVX2__
    size_t i = 0;
    __m256i acc = _mm256_setzero_si256();
    const __m256i z = _mm256_setzero_si256();
    const __m256i one = _mm256_set1_epi8(1);
    for (; i + 32 <= n; i += 32) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i));
        const __m256i ones = _mm256_andnot_si256(_mm256_cmpeq_epi8(v, z), one);
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(ones, z));
    }
    alignas(32) uint64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);
    c = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < n; ++i) {
        c += p[i] ? 1 : 0;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        c += p[i] ? 1 : 0;
    }
#endif
    return c;
}

size_t count_true(const Array<uint8_t>& mask) {
    const size_t n = mask.shape().numel();
    if (mask.is_contiguous()) {
        return count_bytes_nonzero(mask.data(), n);
    }
    size_t c = 0;
    std::vector<size_t> idx(mask.ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        if (mask[Shape(idx)]) {
            ++c;
        }
        bump_index(idx, mask.shape());
    }
    return c;
}

Shape trailing_shape_of(const Shape& sh, size_t skip) {
    std::vector<size_t> dims;
    for (size_t i = skip; i < sh.ndim(); ++i) {
        dims.push_back(sh[i]);
    }
    return Shape(dims);
}

template<typename T>
void copy_slab(Array<T>& dest, size_t dest_row, const Array<T>& src, const std::vector<size_t>& prefix) {
    const size_t mnd = prefix.size();
    const size_t trail_n = trailing_product(src.shape(), mnd);
    std::vector<size_t> full(src.ndim(), 0);
    for (size_t i = 0; i < mnd; ++i) {
        full[i] = prefix[i];
    }
    const size_t base = dest_row * trail_n;
    if (mnd == src.ndim()) {
        dest[base] = src[Shape(full)];
        return;
    }
    std::vector<size_t> trail(src.ndim() - mnd, 0);
    const Shape trail_shape = trailing_shape_of(src.shape(), mnd);
    for (size_t t = 0; t < trail_n; ++t) {
        for (size_t i = 0; i < trail.size(); ++i) {
            full[mnd + i] = trail[i];
        }
        dest[base + t] = src[Shape(full)];
        bump_index(trail, trail_shape);
    }
}

template<typename T>
void fill_slab(Array<T>& dest, const std::vector<size_t>& prefix, T value) {
    const size_t mnd = prefix.size();
    const size_t trail_n = trailing_product(dest.shape(), mnd);
    std::vector<size_t> full(dest.ndim(), 0);
    for (size_t i = 0; i < mnd; ++i) {
        full[i] = prefix[i];
    }
    if (mnd == dest.ndim()) {
        dest[Shape(full)] = value;
        return;
    }
    std::vector<size_t> trail(dest.ndim() - mnd, 0);
    const Shape trail_shape = trailing_shape_of(dest.shape(), mnd);
    for (size_t t = 0; t < trail_n; ++t) {
        for (size_t i = 0; i < trail.size(); ++i) {
            full[mnd + i] = trail[i];
        }
        dest[Shape(full)] = value;
        bump_index(trail, trail_shape);
    }
}

template<typename T>
void write_slab(Array<T>& dest, const std::vector<size_t>& prefix, const Array<T>& src, size_t src_row) {
    const size_t mnd = prefix.size();
    const size_t trail_n = trailing_product(dest.shape(), mnd);
    std::vector<size_t> full(dest.ndim(), 0);
    for (size_t i = 0; i < mnd; ++i) {
        full[i] = prefix[i];
    }
    std::vector<size_t> src_idx(src.ndim(), 0);
    src_idx[0] = src_row;
    if (mnd == dest.ndim()) {
        dest[Shape(full)] = src[Shape(src_idx)];
        return;
    }
    std::vector<size_t> trail(dest.ndim() - mnd, 0);
    const Shape trail_shape = trailing_shape_of(dest.shape(), mnd);
    for (size_t t = 0; t < trail_n; ++t) {
        for (size_t i = 0; i < trail.size(); ++i) {
            full[mnd + i] = trail[i];
            src_idx[1 + i] = trail[i];
        }
        dest[Shape(full)] = src[Shape(src_idx)];
        bump_index(trail, trail_shape);
    }
}

enum class CmpOp { Eq, Ne, Lt, Le, Gt, Ge };

template<typename T>
bool cmp_scalar(T x, T y, CmpOp op) {
    switch (op) {
        case CmpOp::Eq: return x == y;
        case CmpOp::Ne: return x != y;
        case CmpOp::Lt: return x < y;
        case CmpOp::Le: return x <= y;
        case CmpOp::Gt: return x > y;
        case CmpOp::Ge: return x >= y;
    }
    return false;
}

#ifdef __AVX2__
inline void store_f32_cmp_mask_u8(uint8_t* o, __m256 cmp) {
    const __m256i v = _mm256_and_si256(_mm256_castps_si256(cmp), _mm256_set1_epi32(1));
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    const __m128i p16 = _mm_packs_epi32(lo, hi);
    const __m128i p8 = _mm_packs_epi16(p16, p16);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(o), p8);
}

inline void store_f64_cmp_mask_u8(uint8_t* o, __m256d cmp) {
    const int bits = _mm256_movemask_pd(cmp);
    o[0] = static_cast<uint8_t>(bits & 1);
    o[1] = static_cast<uint8_t>((bits >> 1) & 1);
    o[2] = static_cast<uint8_t>((bits >> 2) & 1);
    o[3] = static_cast<uint8_t>((bits >> 3) & 1);
}

inline __m256 cmp_ps_op(__m256 a, __m256 b, CmpOp op) {
    switch (op) {
        case CmpOp::Eq: return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
        case CmpOp::Ne: return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ);
        case CmpOp::Lt: return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
        case CmpOp::Le: return _mm256_cmp_ps(a, b, _CMP_LE_OQ);
        case CmpOp::Gt: return _mm256_cmp_ps(a, b, _CMP_GT_OQ);
        case CmpOp::Ge: return _mm256_cmp_ps(a, b, _CMP_GE_OQ);
    }
    return _mm256_setzero_ps();
}

inline __m256d cmp_pd_op(__m256d a, __m256d b, CmpOp op) {
    switch (op) {
        case CmpOp::Eq: return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
        case CmpOp::Ne: return _mm256_cmp_pd(a, b, _CMP_NEQ_OQ);
        case CmpOp::Lt: return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
        case CmpOp::Le: return _mm256_cmp_pd(a, b, _CMP_LE_OQ);
        case CmpOp::Gt: return _mm256_cmp_pd(a, b, _CMP_GT_OQ);
        case CmpOp::Ge: return _mm256_cmp_pd(a, b, _CMP_GE_OQ);
    }
    return _mm256_setzero_pd();
}
#endif

template<typename T>
Array<uint8_t> compare_arrays(const Array<T>& a, const Array<T>& b, CmpOp op) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("compare: shape mismatch");
    }
    Array<uint8_t> out(a.shape());
    const size_t n = a.shape().numel();
    if (a.is_contiguous() && b.is_contiguous()) {
        const T* pa = a.data();
        const T* pb = b.data();
        uint8_t* o = out.data();
#ifdef __AVX2__
        if constexpr (std::is_same_v<T, float>) {
            size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                store_f32_cmp_mask_u8(
                    o + i,
                    cmp_ps_op(_mm256_loadu_ps(pa + i), _mm256_loadu_ps(pb + i), op)
                );
            }
            for (; i < n; ++i) {
                o[i] = cmp_scalar(pa[i], pb[i], op) ? 1 : 0;
            }
            return out;
        } else if constexpr (std::is_same_v<T, double>) {
            size_t i = 0;
            for (; i + 4 <= n; i += 4) {
                store_f64_cmp_mask_u8(
                    o + i,
                    cmp_pd_op(_mm256_loadu_pd(pa + i), _mm256_loadu_pd(pb + i), op)
                );
            }
            for (; i < n; ++i) {
                o[i] = cmp_scalar(pa[i], pb[i], op) ? 1 : 0;
            }
            return out;
        }
#endif
        for (size_t i = 0; i < n; ++i) {
            o[i] = cmp_scalar(pa[i], pb[i], op) ? 1 : 0;
        }
        return out;
    }
    std::vector<size_t> idx(a.ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        const Shape s(idx);
        out[s] = cmp_scalar(a[s], b[s], op) ? 1 : 0;
        bump_index(idx, a.shape());
    }
    return out;
}

template<typename T>
Array<uint8_t> compare_value(const Array<T>& a, T value, CmpOp op) {
    Array<uint8_t> out(a.shape());
    const size_t n = a.shape().numel();
    if (a.is_contiguous()) {
        const T* p = a.data();
        uint8_t* o = out.data();
#ifdef __AVX2__
        if constexpr (std::is_same_v<T, float>) {
            const __m256 vb = _mm256_set1_ps(value);
            size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                store_f32_cmp_mask_u8(o + i, cmp_ps_op(_mm256_loadu_ps(p + i), vb, op));
            }
            for (; i < n; ++i) {
                o[i] = cmp_scalar(p[i], value, op) ? 1 : 0;
            }
            return out;
        } else if constexpr (std::is_same_v<T, double>) {
            const __m256d vb = _mm256_set1_pd(value);
            size_t i = 0;
            for (; i + 4 <= n; i += 4) {
                store_f64_cmp_mask_u8(o + i, cmp_pd_op(_mm256_loadu_pd(p + i), vb, op));
            }
            for (; i < n; ++i) {
                o[i] = cmp_scalar(p[i], value, op) ? 1 : 0;
            }
            return out;
        }
#endif
        for (size_t i = 0; i < n; ++i) {
            o[i] = cmp_scalar(p[i], value, op) ? 1 : 0;
        }
        return out;
    }
    std::vector<size_t> idx(a.ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        const Shape s(idx);
        out[s] = cmp_scalar(a[s], value, op) ? 1 : 0;
        bump_index(idx, a.shape());
    }
    return out;
}

template<typename Fn>
Array<uint8_t> mask_zip(const Array<uint8_t>& a, const Array<uint8_t>& b, Fn fn) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("mask: shape mismatch");
    }
    Array<uint8_t> out(a.shape());
    const size_t n = a.shape().numel();
    if (a.is_contiguous() && b.is_contiguous()) {
        const uint8_t* pa = a.data();
        const uint8_t* pb = b.data();
        uint8_t* o = out.data();
        for (size_t i = 0; i < n; ++i) {
            o[i] = fn(pa[i], pb[i]) ? 1 : 0;
        }
        return out;
    }
    std::vector<size_t> idx(a.ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        const Shape s(idx);
        out[s] = fn(a[s], b[s]) ? 1 : 0;
        bump_index(idx, a.shape());
    }
    return out;
}

Shape matmul_index(const Shape& tensor_shape, size_t bi, size_t row, size_t col, bool broadcast_2d) {
    const size_t nd = tensor_shape.ndim();
    if (nd == 2 || broadcast_2d) {
        return Shape(std::vector<size_t>{row, col});
    }
    std::vector<size_t> idx(nd);
    idx[nd - 2] = row;
    idx[nd - 1] = col;
    size_t rest = bi;
    for (size_t d = nd - 2; d-- > 0; ) {
        idx[d] = rest % tensor_shape[d];
        rest /= tensor_shape[d];
    }
    return Shape(idx);
}

} // namespace

#pragma region math operations

#pragma region fast math operations

/**
* computes the inner product of two arrays (dot product / vector product)
* NOTE: THIS IS A DEBUGING FUNCTION AND SHOULD NOT BE USED FOR THE PY SIDE BINDINGS, USE _fast_inner_product INSTEAD
*/
template<typename T>
T Array<T>::_fast_inner_product_scalar(const Array& lhs, const Array& rhs) {
    if (lhs.shape().numel() != rhs.shape().numel()) {
        throw std::runtime_error("inner_product: numel mismatch");
    }
    const size_t n = lhs.shape().numel();
    T result = 0;
    if (n == 0) {
        return result;
    }
    std::vector<size_t> ia(lhs.ndim(), 0);
    std::vector<size_t> ib(rhs.ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        result += lhs[Shape(ia)] * rhs[Shape(ib)];
        bump_index(ia, lhs.shape());
        bump_index(ib, rhs.shape());
    }
    return result;
}

/**
* computes the inner product of two vectors (dor product / vector product)
* Uses AVX2 SIMD if available otherwise naive element wise loop
*
* #### args
* - lhs: left hand side array
* - rhs: right hand side array
*
* #### returns
* - inner product of the two arrays
*
* #### throws
* - std::runtime_error: if the numel mismatch
*/
template<typename T>
T Array<T>::_fast_inner_product(const Array& lhs, const Array& rhs) {
    // NOTE: specialized arrays like vec3/vec4 should override with a constexpr version
    if (!(lhs.is_contiguous() && rhs.is_contiguous())) {
        return _fast_inner_product_scalar(lhs, rhs);
    }
    if (lhs.shape().numel() != rhs.shape().numel()) {
        throw std::runtime_error("inner_product: numel mismatch");
    }
    const T* a = lhs.data();
    const T* b = rhs.data();
    const size_t n = lhs.shape().numel();

#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        // AVX2 float: both vectors contiguous - loadu + fmadd along the length
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
        // AVX2 double: both vectors contiguous - loadu + fmadd along the length
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

/**
* Compute the cross product of two vectors (must have inner dimension of 3)
* The result is in plaxe written into the lhs array
*
* #### args
* - lhs: left hand side array
* - rhs: right hand side array
*
* #### returns
* - void
*
* #### throws
* - std::runtime_error: if the shapes are invalid
*/
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
    if (lhs.is_contiguous() && rhs.is_contiguous()) {
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
        return;
    }
    const size_t nd = lhs.ndim();
    std::vector<size_t> idx(nd, 0);
    for (size_t v = 0; v < n_vecs; ++v) {
        idx[nd - 1] = 0;
        const T ax = lhs[Shape(idx)];
        const T bx = rhs[Shape(idx)];
        idx[nd - 1] = 1;
        const T ay = lhs[Shape(idx)];
        const T by = rhs[Shape(idx)];
        idx[nd - 1] = 2;
        const T az = lhs[Shape(idx)];
        const T bz = rhs[Shape(idx)];
        idx[nd - 1] = 0;
        lhs[Shape(idx)] = ay * bz - az * by;
        idx[nd - 1] = 1;
        lhs[Shape(idx)] = az * bx - ax * bz;
        idx[nd - 1] = 2;
        lhs[Shape(idx)] = ax * by - ay * bx;
        if (nd == 1) {
            break;
        }
        idx[nd - 1] = 0;
        for (size_t d = nd - 1; d-- > 0; ) {
            ++idx[d];
            if (idx[d] < lhs.shape()[d]) {
                break;
            }
            idx[d] = 0;
        }
    }
}

/**
* Compute the matrix multiplication of two matrices (dense row-major batched)
* #### NOTE: THIS IS A DEBUGING FUNCTION AND SHOULD NOT BE USED FOR THE PY SIDE BINDINGS, USE _fast_matmul INSTEAD
*
* #### args
* - out: output array
* - lhs: left hand side matrix
* - rhs: right hand side matrix
*
* #### returns
* - void
*/
template<typename T>
void Array<T>::_fast_matmul_scalar(Array& out, const Array& lhs, const Array& rhs) {
    // Batched GEMM using logical indices (correct for strided views).
    const MatmulSpec spec = resolve_matmul(lhs.shape(), rhs.shape());
    if (out.shape() != spec.out_shape) {
        throw std::runtime_error("matmul: incompatible output shape");
    }

    const size_t M = spec.M;
    const size_t K = spec.K;
    const size_t N = spec.N;
    const bool broadcast_a = (spec.a_stride == 0);
    const bool broadcast_b = (spec.b_stride == 0);

    for (size_t bi = 0; bi < spec.batch; ++bi) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                T sum = 0;
                for (size_t k = 0; k < K; ++k) {
                    sum += lhs[matmul_index(lhs.shape(), bi, i, k, broadcast_a)]
                         * rhs[matmul_index(rhs.shape(), bi, k, j, broadcast_b)];
                }
                out[matmul_index(out.shape(), bi, i, j, false)] = sum;
            }
        }
    }
}

namespace {

// Cache panels + register microkernel (f32 4x16 / f64 4x8). No L1 probe.
// Larger Kc/Nc amortize pack cost on mid/large GEMM (aim: panels fit L2).
constexpr size_t kGemmMc = 64;
constexpr size_t kGemmNc = 256;
constexpr size_t kGemmKc = 256;
// Parallel panel GEMM only when M*N*K is large enough that spawn cost pays off.
constexpr uint64_t kGemmParallelMnk = 64ull << 20; // ~67e6 mul-adds
constexpr double kGemmParallelCoreFrac = 0.8;      // leave ~20% for OS / main / pool
constexpr size_t kGemmMr = 4;

#ifdef __AVX2__
constexpr size_t kGemmNrF32 = 16;
constexpr size_t kGemmNrF64 = 8;

// C[4 x 16] += A_pack[4 x kc] @ B_pack[kc x nc] at column offset jj (nr=16).
// first_k: beta=0 (zero acc) - out is freshly allocated zeros on the first Kc panel.
void gemm_kernel_f32_4x16(
    float* c,
    size_t ldc,
    const float* a_pack,
    const float* b_pack,
    size_t kc,
    size_t nc,
    size_t jj,
    bool first_k
) {
    __m256 c00, c01, c10, c11, c20, c21, c30, c31;
    if (first_k) {
        c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = _mm256_setzero_ps();
    } else {
        c00 = _mm256_loadu_ps(c + 0 * ldc + jj);
        c01 = _mm256_loadu_ps(c + 0 * ldc + jj + 8);
        c10 = _mm256_loadu_ps(c + 1 * ldc + jj);
        c11 = _mm256_loadu_ps(c + 1 * ldc + jj + 8);
        c20 = _mm256_loadu_ps(c + 2 * ldc + jj);
        c21 = _mm256_loadu_ps(c + 2 * ldc + jj + 8);
        c30 = _mm256_loadu_ps(c + 3 * ldc + jj);
        c31 = _mm256_loadu_ps(c + 3 * ldc + jj + 8);
    }
    for (size_t k = 0; k < kc; ++k) {
        const __m256 b0 = _mm256_loadu_ps(b_pack + k * nc + jj);
        const __m256 b1 = _mm256_loadu_ps(b_pack + k * nc + jj + 8);
        const __m256 a0 = _mm256_set1_ps(a_pack[0 * kc + k]);
        const __m256 a1 = _mm256_set1_ps(a_pack[1 * kc + k]);
        const __m256 a2 = _mm256_set1_ps(a_pack[2 * kc + k]);
        const __m256 a3 = _mm256_set1_ps(a_pack[3 * kc + k]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
    }
    _mm256_storeu_ps(c + 0 * ldc + jj, c00);
    _mm256_storeu_ps(c + 0 * ldc + jj + 8, c01);
    _mm256_storeu_ps(c + 1 * ldc + jj, c10);
    _mm256_storeu_ps(c + 1 * ldc + jj + 8, c11);
    _mm256_storeu_ps(c + 2 * ldc + jj, c20);
    _mm256_storeu_ps(c + 2 * ldc + jj + 8, c21);
    _mm256_storeu_ps(c + 3 * ldc + jj, c30);
    _mm256_storeu_ps(c + 3 * ldc + jj + 8, c31);
}

void gemm_kernel_f32_1x16(
    float* c_row,
    const float* a_pack,
    const float* b_pack,
    size_t kc,
    size_t nc,
    size_t jj,
    bool first_k
) {
    __m256 c0 = first_k ? _mm256_setzero_ps() : _mm256_loadu_ps(c_row + jj);
    __m256 c1 = first_k ? _mm256_setzero_ps() : _mm256_loadu_ps(c_row + jj + 8);
    for (size_t k = 0; k < kc; ++k) {
        const __m256 b0 = _mm256_loadu_ps(b_pack + k * nc + jj);
        const __m256 b1 = _mm256_loadu_ps(b_pack + k * nc + jj + 8);
        const __m256 av = _mm256_set1_ps(a_pack[k]);
        c0 = _mm256_fmadd_ps(av, b0, c0);
        c1 = _mm256_fmadd_ps(av, b1, c1);
    }
    _mm256_storeu_ps(c_row + jj, c0);
    _mm256_storeu_ps(c_row + jj + 8, c1);
}

void gemm_kernel_f64_4x8(
    double* c,
    size_t ldc,
    const double* a_pack,
    const double* b_pack,
    size_t kc,
    size_t nc,
    size_t jj,
    bool first_k
) {
    __m256d c00, c01, c10, c11, c20, c21, c30, c31;
    if (first_k) {
        c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = _mm256_setzero_pd();
    } else {
        c00 = _mm256_loadu_pd(c + 0 * ldc + jj);
        c01 = _mm256_loadu_pd(c + 0 * ldc + jj + 4);
        c10 = _mm256_loadu_pd(c + 1 * ldc + jj);
        c11 = _mm256_loadu_pd(c + 1 * ldc + jj + 4);
        c20 = _mm256_loadu_pd(c + 2 * ldc + jj);
        c21 = _mm256_loadu_pd(c + 2 * ldc + jj + 4);
        c30 = _mm256_loadu_pd(c + 3 * ldc + jj);
        c31 = _mm256_loadu_pd(c + 3 * ldc + jj + 4);
    }
    for (size_t k = 0; k < kc; ++k) {
        const __m256d b0 = _mm256_loadu_pd(b_pack + k * nc + jj);
        const __m256d b1 = _mm256_loadu_pd(b_pack + k * nc + jj + 4);
        const __m256d a0 = _mm256_set1_pd(a_pack[0 * kc + k]);
        const __m256d a1 = _mm256_set1_pd(a_pack[1 * kc + k]);
        const __m256d a2 = _mm256_set1_pd(a_pack[2 * kc + k]);
        const __m256d a3 = _mm256_set1_pd(a_pack[3 * kc + k]);
        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c01 = _mm256_fmadd_pd(a0, b1, c01);
        c10 = _mm256_fmadd_pd(a1, b0, c10);
        c11 = _mm256_fmadd_pd(a1, b1, c11);
        c20 = _mm256_fmadd_pd(a2, b0, c20);
        c21 = _mm256_fmadd_pd(a2, b1, c21);
        c30 = _mm256_fmadd_pd(a3, b0, c30);
        c31 = _mm256_fmadd_pd(a3, b1, c31);
    }
    _mm256_storeu_pd(c + 0 * ldc + jj, c00);
    _mm256_storeu_pd(c + 0 * ldc + jj + 4, c01);
    _mm256_storeu_pd(c + 1 * ldc + jj, c10);
    _mm256_storeu_pd(c + 1 * ldc + jj + 4, c11);
    _mm256_storeu_pd(c + 2 * ldc + jj, c20);
    _mm256_storeu_pd(c + 2 * ldc + jj + 4, c21);
    _mm256_storeu_pd(c + 3 * ldc + jj, c30);
    _mm256_storeu_pd(c + 3 * ldc + jj + 4, c31);
}

void gemm_kernel_f64_1x8(
    double* c_row,
    const double* a_pack,
    const double* b_pack,
    size_t kc,
    size_t nc,
    size_t jj,
    bool first_k
) {
    __m256d c0 = first_k ? _mm256_setzero_pd() : _mm256_loadu_pd(c_row + jj);
    __m256d c1 = first_k ? _mm256_setzero_pd() : _mm256_loadu_pd(c_row + jj + 4);
    for (size_t k = 0; k < kc; ++k) {
        const __m256d b0 = _mm256_loadu_pd(b_pack + k * nc + jj);
        const __m256d b1 = _mm256_loadu_pd(b_pack + k * nc + jj + 4);
        const __m256d av = _mm256_set1_pd(a_pack[k]);
        c0 = _mm256_fmadd_pd(av, b0, c0);
        c1 = _mm256_fmadd_pd(av, b1, c1);
    }
    _mm256_storeu_pd(c_row + jj, c0);
    _mm256_storeu_pd(c_row + jj + 4, c1);
}
#endif

// Contiguous row-major GEMM over C columns [j0_begin, j0_end).
// Panel pack + 4x16/4x8 microkernel; thread_local Mc×Kc / Kc×Nc scratch.
template<typename T>
void gemm_tiled_j_range(
    T* c,
    const T* a,
    const T* b,
    size_t M,
    size_t K,
    size_t N,
    size_t j0_begin,
    size_t j0_end
) {
    thread_local std::vector<T> a_pack_tl;
    thread_local std::vector<T> b_pack_tl;
    a_pack_tl.resize(kGemmMc * kGemmKc);
    b_pack_tl.resize(kGemmKc * kGemmNc);
    T* a_pack = a_pack_tl.data();
    T* b_pack = b_pack_tl.data();

    for (size_t j0 = j0_begin; j0 < j0_end; j0 += kGemmNc) {
        const size_t nc = std::min(kGemmNc, j0_end - j0);
        for (size_t k0 = 0; k0 < K; k0 += kGemmKc) {
            const size_t kc = std::min(kGemmKc, K - k0);
            const bool first_k = (k0 == 0);

            for (size_t k = 0; k < kc; ++k) {
                std::memcpy(
                    b_pack + k * nc,
                    b + (k0 + k) * N + j0,
                    nc * sizeof(T)
                );
            }

            for (size_t i0 = 0; i0 < M; i0 += kGemmMc) {
                const size_t mc = std::min(kGemmMc, M - i0);
                for (size_t i = 0; i < mc; ++i) {
                    std::memcpy(
                        a_pack + i * kc,
                        a + (i0 + i) * K + k0,
                        kc * sizeof(T)
                    );
                }

#ifdef __AVX2__
                if constexpr (std::is_same_v<T, float>) {
                    size_t ii = 0;
                    for (; ii + kGemmMr <= mc; ii += kGemmMr) {
                        float* c_block = c + (i0 + ii) * N + j0;
                        const float* ap = a_pack + ii * kc;
                        size_t jj = 0;
                        for (; jj + kGemmNrF32 <= nc; jj += kGemmNrF32) {
                            gemm_kernel_f32_4x16(c_block, N, ap, b_pack, kc, nc, jj, first_k);
                        }
                        for (; jj < nc; ++jj) {
                            for (size_t r = 0; r < kGemmMr; ++r) {
                                float sum = first_k ? 0.f : c_block[r * N + jj];
                                for (size_t k = 0; k < kc; ++k) {
                                    sum += ap[r * kc + k] * b_pack[k * nc + jj];
                                }
                                c_block[r * N + jj] = sum;
                            }
                        }
                    }
                    for (; ii < mc; ++ii) {
                        float* c_row = c + (i0 + ii) * N + j0;
                        const float* ap = a_pack + ii * kc;
                        size_t jj = 0;
                        for (; jj + kGemmNrF32 <= nc; jj += kGemmNrF32) {
                            gemm_kernel_f32_1x16(c_row, ap, b_pack, kc, nc, jj, first_k);
                        }
                        for (; jj < nc; ++jj) {
                            float sum = first_k ? 0.f : c_row[jj];
                            for (size_t k = 0; k < kc; ++k) {
                                sum += ap[k] * b_pack[k * nc + jj];
                            }
                            c_row[jj] = sum;
                        }
                    }
                } else if constexpr (std::is_same_v<T, double>) {
                    size_t ii = 0;
                    for (; ii + kGemmMr <= mc; ii += kGemmMr) {
                        double* c_block = c + (i0 + ii) * N + j0;
                        const double* ap = a_pack + ii * kc;
                        size_t jj = 0;
                        for (; jj + kGemmNrF64 <= nc; jj += kGemmNrF64) {
                            gemm_kernel_f64_4x8(c_block, N, ap, b_pack, kc, nc, jj, first_k);
                        }
                        for (; jj < nc; ++jj) {
                            for (size_t r = 0; r < kGemmMr; ++r) {
                                double sum = first_k ? 0.0 : c_block[r * N + jj];
                                for (size_t k = 0; k < kc; ++k) {
                                    sum += ap[r * kc + k] * b_pack[k * nc + jj];
                                }
                                c_block[r * N + jj] = sum;
                            }
                        }
                    }
                    for (; ii < mc; ++ii) {
                        double* c_row = c + (i0 + ii) * N + j0;
                        const double* ap = a_pack + ii * kc;
                        size_t jj = 0;
                        for (; jj + kGemmNrF64 <= nc; jj += kGemmNrF64) {
                            gemm_kernel_f64_1x8(c_row, ap, b_pack, kc, nc, jj, first_k);
                        }
                        for (; jj < nc; ++jj) {
                            double sum = first_k ? 0.0 : c_row[jj];
                            for (size_t k = 0; k < kc; ++k) {
                                sum += ap[k] * b_pack[k * nc + jj];
                            }
                            c_row[jj] = sum;
                        }
                    }
                } else {
                    for (size_t i = 0; i < mc; ++i) {
                        T* c_row = c + (i0 + i) * N + j0;
                        const T* ap = a_pack + i * kc;
                        for (size_t j = 0; j < nc; ++j) {
                            T sum = first_k ? T{} : c_row[j];
                            for (size_t k = 0; k < kc; ++k) {
                                sum += ap[k] * b_pack[k * nc + j];
                            }
                            c_row[j] = sum;
                        }
                    }
                }
#else
                for (size_t i = 0; i < mc; ++i) {
                    T* c_row = c + (i0 + i) * N + j0;
                    const T* ap = a_pack + i * kc;
                    for (size_t j = 0; j < nc; ++j) {
                        T sum = first_k ? T{} : c_row[j];
                        for (size_t k = 0; k < kc; ++k) {
                            sum += ap[k] * b_pack[k * nc + j];
                        }
                        c_row[j] = sum;
                    }
                }
#endif
            }
        }
    }
}

template<typename T>
void gemm_tiled_contiguous(T* c, const T* a, const T* b, size_t M, size_t K, size_t N, bool parallel) {
    if (!parallel) {
        gemm_tiled_j_range(c, a, b, M, K, N, 0, N);
        return;
    }
    const uint64_t mnk = static_cast<uint64_t>(M) * static_cast<uint64_t>(N) * static_cast<uint64_t>(K);
    if (mnk < kGemmParallelMnk) {
        gemm_tiled_j_range(c, a, b, M, K, N, 0, N);
        return;
    }
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        hw = 4;
    }
    size_t workers = static_cast<size_t>(hw * kGemmParallelCoreFrac);
    if (workers < 1) {
        workers = 1;
    }
    const size_t n_panels = (N + kGemmNc - 1) / kGemmNc;
    if (workers > n_panels) {
        workers = n_panels;
    }
    if (workers <= 1) {
        gemm_tiled_j_range(c, a, b, M, K, N, 0, N);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (size_t tid = 0; tid < workers; ++tid) {
        const size_t p0 = (n_panels * tid) / workers;
        const size_t p1 = (n_panels * (tid + 1)) / workers;
        if (p0 >= p1) {
            continue;
        }
        const size_t j0_begin = p0 * kGemmNc;
        const size_t j0_end = std::min(N, p1 * kGemmNc);
        threads.emplace_back([=]() {
            gemm_tiled_j_range(c, a, b, M, K, N, j0_begin, j0_end);
        });
    }
    for (std::thread& th : threads) {
        th.join();
    }
}

} // namespace

/**
* Compute the matrix multiplication of two matrices (dense row-major batched)
* Uses AVX2 SIMD if available otherwise naive element wise loop
* The result is in place written to the out array
*
* #### args
* - out: output array
* - lhs: left hand side matrix
* - rhs: right hand side matrix
* - parallel: if true, panel-parallel GEMM when M*N*K is above threshold
*
* #### returns
* - void
*
* #### throws
* - std::runtime_error: if the shapes are invalid
*/
template<typename T>
void Array<T>::_fast_matmul(Array& out, const Array& lhs, const Array& rhs, bool parallel) {
    // Dense row-major batched: lhs (..., M, K) @ rhs (..., K, N) -> out (..., M, N)
    if (!(lhs.is_contiguous() && rhs.is_contiguous() && out.is_contiguous())) {
        _fast_matmul_scalar(out, lhs, rhs);
        return;
    }
    const MatmulSpec spec = resolve_matmul(lhs.shape(), rhs.shape());
    if (out.shape() != spec.out_shape) {
        throw std::runtime_error("matmul: incompatible output shape");
    }
    const size_t M = spec.M;
    const size_t K = spec.K;
    const size_t N = spec.N;

    // out written by first_k (beta=0) then accumulated; no zero-fill required.
    for (size_t bi = 0; bi < spec.batch; ++bi) {
        const T* a = lhs.data() + bi * spec.a_stride;
        const T* b = rhs.data() + bi * spec.b_stride;
        T* c = out.data() + bi * spec.c_stride;
        gemm_tiled_contiguous(c, a, b, M, K, N, parallel);
    }
}

/**
* Add two arrays elementwise
* Uses AVX2 SIMD if available otherwise naive element wise loop
*
* #### args
* - lhs: left hand side array
* - rhs: right hand side array
*
* #### returns
* - void
*
* #### throws
* - std::runtime_error: if the shapes are invalid
*/
namespace {

// Contiguous ewise: out[i] = a[i] ⊕ b[i] (out may alias a for inplace).
template<typename T>
void ewise_add_contig(T* out, const T* a, const T* b, size_t n) {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        }
        for (; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
    } else if constexpr (std::is_same_v<T, double>) {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            _mm256_storeu_pd(out + i, _mm256_add_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
        }
        for (; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
#endif
}

template<typename T>
void ewise_sub_contig(T* out, const T* a, const T* b, size_t n) {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(out + i, _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        }
        for (; i < n; ++i) {
            out[i] = a[i] - b[i];
        }
    } else if constexpr (std::is_same_v<T, double>) {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            _mm256_storeu_pd(out + i, _mm256_sub_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
        }
        for (; i < n; ++i) {
            out[i] = a[i] - b[i];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            out[i] = a[i] - b[i];
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
#endif
}

template<typename T>
void ewise_mul_contig(T* out, const T* a, const T* b, size_t n) {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        }
        for (; i < n; ++i) {
            out[i] = a[i] * b[i];
        }
    } else if constexpr (std::is_same_v<T, double>) {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            _mm256_storeu_pd(out + i, _mm256_mul_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
        }
        for (; i < n; ++i) {
            out[i] = a[i] * b[i];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            out[i] = a[i] * b[i];
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
#endif
}

template<typename T>
void ewise_div_contig(T* out, const T* a, const T* b, size_t n) {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(out + i, _mm256_div_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        }
        for (; i < n; ++i) {
            out[i] = a[i] / b[i];
        }
    } else if constexpr (std::is_same_v<T, double>) {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            _mm256_storeu_pd(out + i, _mm256_div_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
        }
        for (; i < n; ++i) {
            out[i] = a[i] / b[i];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            out[i] = a[i] / b[i];
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] / b[i];
    }
#endif
}

template<typename T>
void ewise_neg_contig(T* out, const T* a, size_t n) {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        const __m256 z = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(out + i, _mm256_sub_ps(z, _mm256_loadu_ps(a + i)));
        }
        for (; i < n; ++i) {
            out[i] = -a[i];
        }
    } else if constexpr (std::is_same_v<T, double>) {
        const __m256d z = _mm256_setzero_pd();
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            _mm256_storeu_pd(out + i, _mm256_sub_pd(z, _mm256_loadu_pd(a + i)));
        }
        for (; i < n; ++i) {
            out[i] = -a[i];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            out[i] = -a[i];
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = -a[i];
    }
#endif
}

template<typename T>
void ewise_add_scalar_contig(T* out, const T* a, T value, size_t n) {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        const __m256 vb = _mm256_set1_ps(value);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i), vb));
        }
        for (; i < n; ++i) {
            out[i] = a[i] + value;
        }
    } else if constexpr (std::is_same_v<T, double>) {
        const __m256d vb = _mm256_set1_pd(value);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            _mm256_storeu_pd(out + i, _mm256_add_pd(_mm256_loadu_pd(a + i), vb));
        }
        for (; i < n; ++i) {
            out[i] = a[i] + value;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            out[i] = a[i] + value;
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + value;
    }
#endif
}

template<typename T>
void ewise_sub_scalar_contig(T* out, const T* a, T value, size_t n) {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        const __m256 vb = _mm256_set1_ps(value);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(out + i, _mm256_sub_ps(_mm256_loadu_ps(a + i), vb));
        }
        for (; i < n; ++i) {
            out[i] = a[i] - value;
        }
    } else if constexpr (std::is_same_v<T, double>) {
        const __m256d vb = _mm256_set1_pd(value);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            _mm256_storeu_pd(out + i, _mm256_sub_pd(_mm256_loadu_pd(a + i), vb));
        }
        for (; i < n; ++i) {
            out[i] = a[i] - value;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            out[i] = a[i] - value;
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] - value;
    }
#endif
}

template<typename T>
void ewise_mul_scalar_contig(T* out, const T* a, T value, size_t n) {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        const __m256 vb = _mm256_set1_ps(value);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), vb));
        }
        for (; i < n; ++i) {
            out[i] = a[i] * value;
        }
    } else if constexpr (std::is_same_v<T, double>) {
        const __m256d vb = _mm256_set1_pd(value);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            _mm256_storeu_pd(out + i, _mm256_mul_pd(_mm256_loadu_pd(a + i), vb));
        }
        for (; i < n; ++i) {
            out[i] = a[i] * value;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            out[i] = a[i] * value;
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * value;
    }
#endif
}

template<typename T>
void ewise_div_scalar_contig(T* out, const T* a, T value, size_t n) {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        const __m256 vb = _mm256_set1_ps(value);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(out + i, _mm256_div_ps(_mm256_loadu_ps(a + i), vb));
        }
        for (; i < n; ++i) {
            out[i] = a[i] / value;
        }
    } else if constexpr (std::is_same_v<T, double>) {
        const __m256d vb = _mm256_set1_pd(value);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            _mm256_storeu_pd(out + i, _mm256_div_pd(_mm256_loadu_pd(a + i), vb));
        }
        for (; i < n; ++i) {
            out[i] = a[i] / value;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            out[i] = a[i] / value;
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] / value;
    }
#endif
}

} // namespace

template<typename T>
void Array<T>::_fast_add(Array& lhs, const Array& rhs) {
    if (rhs.shape().numel() == 1 && lhs.shape().numel() != 1) { // if the rhs is a scalar, add it to the lhs
        _fast_add(lhs, rhs[0]); // call the scalar overload
        return;
    }
    if (lhs.shape() != rhs.shape()) { // shapes must match
        throw std::runtime_error("add: shape mismatch");
    }
    if (!(lhs.is_contiguous() && rhs.is_contiguous())) {
        std::vector<size_t> idx(lhs.ndim(), 0);
        const size_t n = lhs.shape().numel();
        for (size_t t = 0; t < n; ++t) {
            const Shape s(idx);
            lhs[s] += rhs[s];
            bump_index(idx, lhs.shape());
        }
        return;
    }
    T* a = lhs.data();
    ewise_add_contig(a, a, rhs.data(), lhs.shape().numel());
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
    if (!(lhs.is_contiguous() && rhs.is_contiguous())) {
        std::vector<size_t> idx(lhs.ndim(), 0);
        const size_t n = lhs.shape().numel();
        for (size_t t = 0; t < n; ++t) {
            const Shape s(idx);
            lhs[s] -= rhs[s];
            bump_index(idx, lhs.shape());
        }
        return;
    }
    T* a = lhs.data();
    ewise_sub_contig(a, a, rhs.data(), lhs.shape().numel());
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
    if (!(lhs.is_contiguous() && rhs.is_contiguous())) {
        std::vector<size_t> idx(lhs.ndim(), 0);
        const size_t n = lhs.shape().numel();
        for (size_t t = 0; t < n; ++t) {
            const Shape s(idx);
            lhs[s] *= rhs[s];
            bump_index(idx, lhs.shape());
        }
        return;
    }
    T* a = lhs.data();
    ewise_mul_contig(a, a, rhs.data(), lhs.shape().numel());
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
    if (!(lhs.is_contiguous() && rhs.is_contiguous())) {
        std::vector<size_t> idx(lhs.ndim(), 0);
        const size_t n = lhs.shape().numel();
        for (size_t t = 0; t < n; ++t) {
            const Shape s(idx);
            lhs[s] /= rhs[s];
            bump_index(idx, lhs.shape());
        }
        return;
    }
    T* a = lhs.data();
    ewise_div_contig(a, a, rhs.data(), lhs.shape().numel());
}

template<typename T>
void Array<T>::_fast_neg(Array& lhs) {
    if (!lhs.is_contiguous()) {
        std::vector<size_t> idx(lhs.ndim(), 0);
        const size_t n = lhs.shape().numel();
        for (size_t t = 0; t < n; ++t) {
            const Shape s(idx);
            lhs[s] = -lhs[s];
            bump_index(idx, lhs.shape());
        }
        return;
    }
    T* a = lhs.data();
    ewise_neg_contig(a, a, lhs.shape().numel());
}

template<typename T>
void Array<T>::_fast_add(Array& lhs, T value) {
    if (!lhs.is_contiguous()) {
        std::vector<size_t> idx(lhs.ndim(), 0);
        const size_t n = lhs.shape().numel();
        for (size_t t = 0; t < n; ++t) {
            lhs[Shape(idx)] += value;
            bump_index(idx, lhs.shape());
        }
        return;
    }
    T* a = lhs.data();
    ewise_add_scalar_contig(a, a, value, lhs.shape().numel());
}

template<typename T>
void Array<T>::_fast_sub(Array& lhs, T value) {
    if (!lhs.is_contiguous()) {
        std::vector<size_t> idx(lhs.ndim(), 0);
        const size_t n = lhs.shape().numel();
        for (size_t t = 0; t < n; ++t) {
            lhs[Shape(idx)] -= value;
            bump_index(idx, lhs.shape());
        }
        return;
    }
    T* a = lhs.data();
    ewise_sub_scalar_contig(a, a, value, lhs.shape().numel());
}

template<typename T>
void Array<T>::_fast_mul(Array& lhs, T value) {
    if (!lhs.is_contiguous()) {
        std::vector<size_t> idx(lhs.ndim(), 0);
        const size_t n = lhs.shape().numel();
        for (size_t t = 0; t < n; ++t) {
            lhs[Shape(idx)] *= value;
            bump_index(idx, lhs.shape());
        }
        return;
    }
    T* a = lhs.data();
    ewise_mul_scalar_contig(a, a, value, lhs.shape().numel());
}

template<typename T>
void Array<T>::_fast_div(Array& lhs, T value) {
    if (!lhs.is_contiguous()) {
        std::vector<size_t> idx(lhs.ndim(), 0);
        const size_t n = lhs.shape().numel();
        for (size_t t = 0; t < n; ++t) {
            lhs[Shape(idx)] /= value;
            bump_index(idx, lhs.shape());
        }
        return;
    }
    T* a = lhs.data();
    ewise_div_scalar_contig(a, a, value, lhs.shape().numel());
}

#pragma endregion fast math operations

#pragma region public math API

template<typename T>
Array<T> Array<T>::matmul(const Array& other, bool parallel) const {
    const MatmulSpec spec = resolve_matmul(this->_shape, other._shape);
    Array<T> out(spec.out_shape);
    _fast_matmul(out, *this, other, parallel);
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
    if (!(this->is_contiguous() && other.is_contiguous())) {
        const size_t nd = this->ndim();
        std::vector<size_t> idx(nd, 0);
        for (size_t v = 0; v < out.shape().numel(); ++v) {
            T sum = 0;
            for (size_t k = 0; k < K; ++k) {
                idx[nd - 1] = k;
                sum += (*this)[Shape(idx)] * other[Shape(idx)];
            }
            out[v] = sum;
            idx[nd - 1] = 0;
            for (size_t d = nd - 1; d-- > 0; ) {
                ++idx[d];
                if (idx[d] < this->_shape[d]) {
                    break;
                }
                idx[d] = 0;
            }
        }
        return out;
    }
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
    const size_t nd = this->ndim();
    std::vector<size_t> idx(nd, 0);
    for (size_t v = 0; v < out.shape().numel(); ++v) {
        T sum = 0;
        for (size_t k = 0; k < K; ++k) {
            idx[nd - 1] = k;
            sum += (*this)[Shape(idx)] * other[Shape(idx)];
        }
        out[v] = sum;
        idx[nd - 1] = 0;
        for (size_t d = nd - 1; d-- > 0; ) {
            ++idx[d];
            if (idx[d] < this->_shape[d]) {
                break;
            }
            idx[d] = 0;
        }
    }
    return out;
}

template<typename T>
void Array<T>::_dot(const Array& other) {
    *this = this->dot(other);
}

template<typename T>
Array<T> Array<T>::cross(const Array& other) const {
    Array<T> out = this->contiguous();
    _fast_cross_product(out, other);
    return out;
}

template<typename T>
void Array<T>::_cross(const Array& other) {
    _fast_cross_product(*this, other);
}

template<typename T>
Array<T> Array<T>::operator+(const Array& other) const {
    if (other.shape().numel() == 1 && this->shape().numel() != 1) {
        return *this + other[0];
    }
    if (this->_shape != other.shape()) {
        throw std::runtime_error("add: shape mismatch");
    }
    Array<T> out(this->_shape);
    const size_t n = this->_shape.numel();
    if (this->is_contiguous() && other.is_contiguous()) {
        ewise_add_contig(out.data(), this->data(), other.data(), n);
        return out;
    }
    std::vector<size_t> idx(this->ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        const Shape s(idx);
        out[t] = (*this)[s] + other[s];
        bump_index(idx, this->_shape);
    }
    return out;
}

template<typename T>
Array<T> Array<T>::operator-(const Array& other) const {
    if (other.shape().numel() == 1 && this->shape().numel() != 1) {
        return *this - other[0];
    }
    if (this->_shape != other.shape()) {
        throw std::runtime_error("sub: shape mismatch");
    }
    Array<T> out(this->_shape);
    const size_t n = this->_shape.numel();
    if (this->is_contiguous() && other.is_contiguous()) {
        ewise_sub_contig(out.data(), this->data(), other.data(), n);
        return out;
    }
    std::vector<size_t> idx(this->ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        const Shape s(idx);
        out[t] = (*this)[s] - other[s];
        bump_index(idx, this->_shape);
    }
    return out;
}

template<typename T>
Array<T> Array<T>::operator*(const Array& other) const {
    if (other.shape().numel() == 1 && this->shape().numel() != 1) {
        return *this * other[0];
    }
    if (this->_shape != other.shape()) {
        throw std::runtime_error("mul: shape mismatch");
    }
    Array<T> out(this->_shape);
    const size_t n = this->_shape.numel();
    if (this->is_contiguous() && other.is_contiguous()) {
        ewise_mul_contig(out.data(), this->data(), other.data(), n);
        return out;
    }
    std::vector<size_t> idx(this->ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        const Shape s(idx);
        out[t] = (*this)[s] * other[s];
        bump_index(idx, this->_shape);
    }
    return out;
}

template<typename T>
Array<T> Array<T>::operator/(const Array& other) const {
    if (other.shape().numel() == 1 && this->shape().numel() != 1) {
        return *this / other[0];
    }
    if (this->_shape != other.shape()) {
        throw std::runtime_error("div: shape mismatch");
    }
    Array<T> out(this->_shape);
    const size_t n = this->_shape.numel();
    if (this->is_contiguous() && other.is_contiguous()) {
        ewise_div_contig(out.data(), this->data(), other.data(), n);
        return out;
    }
    std::vector<size_t> idx(this->ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        const Shape s(idx);
        out[t] = (*this)[s] / other[s];
        bump_index(idx, this->_shape);
    }
    return out;
}

template<typename T>
Array<T> Array<T>::operator+(T value) const {
    Array<T> out(this->_shape);
    const size_t n = this->_shape.numel();
    if (this->is_contiguous()) {
        ewise_add_scalar_contig(out.data(), this->data(), value, n);
        return out;
    }
    std::vector<size_t> idx(this->ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        out[t] = (*this)[Shape(idx)] + value;
        bump_index(idx, this->_shape);
    }
    return out;
}

template<typename T>
Array<T> Array<T>::operator-(T value) const {
    Array<T> out(this->_shape);
    const size_t n = this->_shape.numel();
    if (this->is_contiguous()) {
        ewise_sub_scalar_contig(out.data(), this->data(), value, n);
        return out;
    }
    std::vector<size_t> idx(this->ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        out[t] = (*this)[Shape(idx)] - value;
        bump_index(idx, this->_shape);
    }
    return out;
}

template<typename T>
Array<T> Array<T>::operator*(T value) const {
    Array<T> out(this->_shape);
    const size_t n = this->_shape.numel();
    if (this->is_contiguous()) {
        ewise_mul_scalar_contig(out.data(), this->data(), value, n);
        return out;
    }
    std::vector<size_t> idx(this->ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        out[t] = (*this)[Shape(idx)] * value;
        bump_index(idx, this->_shape);
    }
    return out;
}

template<typename T>
Array<T> Array<T>::operator/(T value) const {
    Array<T> out(this->_shape);
    const size_t n = this->_shape.numel();
    if (this->is_contiguous()) {
        ewise_div_scalar_contig(out.data(), this->data(), value, n);
        return out;
    }
    std::vector<size_t> idx(this->ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        out[t] = (*this)[Shape(idx)] / value;
        bump_index(idx, this->_shape);
    }
    return out;
}

template<typename T>
Array<T> Array<T>::operator-() const {
    Array<T> out(this->_shape);
    const size_t n = this->_shape.numel();
    if (this->is_contiguous()) {
        ewise_neg_contig(out.data(), this->data(), n);
        return out;
    }
    std::vector<size_t> idx(this->ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        out[t] = -(*this)[Shape(idx)];
        bump_index(idx, this->_shape);
    }
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

template<typename T>
Array<uint8_t> Array<T>::operator>(T value) const {
    return compare_value(*this, value, CmpOp::Gt);
}
template<typename T>
Array<uint8_t> Array<T>::operator<(T value) const {
    return compare_value(*this, value, CmpOp::Lt);
}
template<typename T>
Array<uint8_t> Array<T>::operator>=(T value) const {
    return compare_value(*this, value, CmpOp::Ge);
}
template<typename T>
Array<uint8_t> Array<T>::operator<=(T value) const {
    return compare_value(*this, value, CmpOp::Le);
}
template<typename T>
Array<uint8_t> Array<T>::operator==(T value) const {
    return compare_value(*this, value, CmpOp::Eq);
}
template<typename T>
Array<uint8_t> Array<T>::operator!=(T value) const {
    return compare_value(*this, value, CmpOp::Ne);
}
template<typename T>
Array<uint8_t> Array<T>::operator>(const Array& other) const {
    return compare_arrays(*this, other, CmpOp::Gt);
}
template<typename T>
Array<uint8_t> Array<T>::operator<(const Array& other) const {
    return compare_arrays(*this, other, CmpOp::Lt);
}
template<typename T>
Array<uint8_t> Array<T>::operator>=(const Array& other) const {
    return compare_arrays(*this, other, CmpOp::Ge);
}
template<typename T>
Array<uint8_t> Array<T>::operator<=(const Array& other) const {
    return compare_arrays(*this, other, CmpOp::Le);
}
template<typename T>
Array<uint8_t> Array<T>::operator==(const Array& other) const {
    return compare_arrays(*this, other, CmpOp::Eq);
}
template<typename T>
Array<uint8_t> Array<T>::operator!=(const Array& other) const {
    return compare_arrays(*this, other, CmpOp::Ne);
}

template<typename T>
size_t Array<T>::count_nonzero() const {
    const size_t n = this->_shape.numel();
    if (this->_is_contiguous) {
        const T* p = this->data();
        if constexpr (std::is_same_v<T, uint8_t>) {
            return count_bytes_nonzero(p, n);
        }
#ifdef __AVX2__
        if constexpr (std::is_same_v<T, float>) {
            size_t c = 0;
            size_t i = 0;
            const __m256 z = _mm256_setzero_ps();
            for (; i + 8 <= n; i += 8) {
                const __m256 eq = _mm256_cmp_ps(_mm256_loadu_ps(p + i), z, _CMP_EQ_OQ);
                c += 8 - popcnt32(static_cast<unsigned>(_mm256_movemask_ps(eq)));
            }
            for (; i < n; ++i) {
                c += (p[i] != 0.0f) ? 1 : 0;
            }
            return c;
        } else if constexpr (std::is_same_v<T, double>) {
            size_t c = 0;
            size_t i = 0;
            const __m256d z = _mm256_setzero_pd();
            for (; i + 4 <= n; i += 4) {
                const __m256d eq = _mm256_cmp_pd(_mm256_loadu_pd(p + i), z, _CMP_EQ_OQ);
                c += 4 - popcnt32(static_cast<unsigned>(_mm256_movemask_pd(eq)));
            }
            for (; i < n; ++i) {
                c += (p[i] != 0.0) ? 1 : 0;
            }
            return c;
        }
#endif
        size_t c = 0;
        for (size_t i = 0; i < n; ++i) {
            c += (p[i] != T{}) ? 1 : 0;
        }
        return c;
    }
    size_t c = 0;
    std::vector<size_t> idx(this->ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        if ((*this)[Shape(idx)] != T{}) {
            ++c;
        }
        bump_index(idx, this->_shape);
    }
    return c;
}

template<typename T>
Array<T> Array<T>::masked_select(const Array<uint8_t>& mask) const {
    check_mask_prefix(this->_shape, mask.shape());
    const size_t count = count_true(mask);
    Array<T> out(gathered_shape(this->_shape, mask.ndim(), count));
    if (count == 0) {
        return out;
    }
    const size_t mnd = mask.ndim();
    const size_t n_mask = mask.shape().numel();
    const size_t trail = trailing_product(this->_shape, mnd);
    if (this->_is_contiguous && mask.is_contiguous()) {
        const T* src = this->data();
        const uint8_t* m = mask.data();
        T* dst = out.data();
        size_t k = 0;
        if (trail == 1) {
            for (size_t i = 0; i < n_mask; ++i) {
                if (m[i]) {
                    dst[k++] = src[i];
                }
            }
        } else {
            const size_t nbytes = trail * sizeof(T);
            for (size_t i = 0; i < n_mask; ++i) {
                if (m[i]) {
                    std::memcpy(dst + k * trail, src + i * trail, nbytes);
                    ++k;
                }
            }
        }
        return out;
    }
    std::vector<size_t> idx(mnd, 0);
    size_t k = 0;
    for (size_t t = 0; t < n_mask; ++t) {
        if (mask[Shape(idx)]) {
            copy_slab(out, k, *this, idx);
            ++k;
        }
        bump_index(idx, mask.shape());
    }
    return out;
}

template<typename T>
void Array<T>::masked_fill(const Array<uint8_t>& mask, T value) {
    check_mask_prefix(this->_shape, mask.shape());
    const size_t mnd = mask.ndim();
    const size_t n_mask = mask.shape().numel();
    const size_t trail = trailing_product(this->_shape, mnd);
    if (this->_is_contiguous && mask.is_contiguous()) {
        T* data = this->data();
        const uint8_t* m = mask.data();
        if (trail == 1) {
#ifdef __AVX2__
            if constexpr (std::is_same_v<T, float>) {
                const __m256 vb = _mm256_set1_ps(value);
                const __m256i z = _mm256_setzero_si256();
                size_t i = 0;
                for (; i + 8 <= n_mask; i += 8) {
                    const __m128i m8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(m + i));
                    const __m256i m32 = _mm256_cvtepu8_epi32(m8);
                    const __m256 take = _mm256_castsi256_ps(
                        _mm256_andnot_si256(_mm256_cmpeq_epi32(m32, z), _mm256_set1_epi32(-1))
                    );
                    _mm256_storeu_ps(
                        data + i,
                        _mm256_blendv_ps(_mm256_loadu_ps(data + i), vb, take)
                    );
                }
                for (; i < n_mask; ++i) {
                    if (m[i]) {
                        data[i] = value;
                    }
                }
                return;
            } else if constexpr (std::is_same_v<T, double>) {
                const __m256d vb = _mm256_set1_pd(value);
                const __m128i z = _mm_setzero_si128();
                size_t i = 0;
                for (; i + 4 <= n_mask; i += 4) {
                    const __m128i m8 = _mm_cvtsi32_si128(
                        static_cast<int>(m[i]) | (static_cast<int>(m[i + 1]) << 8) |
                        (static_cast<int>(m[i + 2]) << 16) | (static_cast<int>(m[i + 3]) << 24)
                    );
                    const __m128i m32 = _mm_cvtepu8_epi32(m8);
                    const __m128i nz = _mm_andnot_si128(_mm_cmpeq_epi32(m32, z), _mm_set1_epi32(-1));
                    const __m256d take = _mm256_castsi256_pd(_mm256_cvtepi32_epi64(nz));
                    _mm256_storeu_pd(
                        data + i,
                        _mm256_blendv_pd(_mm256_loadu_pd(data + i), vb, take)
                    );
                }
                for (; i < n_mask; ++i) {
                    if (m[i]) {
                        data[i] = value;
                    }
                }
                return;
            }
#endif
            for (size_t i = 0; i < n_mask; ++i) {
                if (m[i]) {
                    data[i] = value;
                }
            }
            return;
        }
        for (size_t i = 0; i < n_mask; ++i) {
            if (m[i]) {
                T* slab = data + i * trail;
                for (size_t j = 0; j < trail; ++j) {
                    slab[j] = value;
                }
            }
        }
        return;
    }
    std::vector<size_t> idx(mnd, 0);
    for (size_t t = 0; t < n_mask; ++t) {
        if (mask[Shape(idx)]) {
            fill_slab(*this, idx, value);
        }
        bump_index(idx, mask.shape());
    }
}

template<typename T>
void Array<T>::masked_scatter(const Array<uint8_t>& mask, const Array& values) {
    check_mask_prefix(this->_shape, mask.shape());
    const size_t count = count_true(mask);
    const Shape expect = gathered_shape(this->_shape, mask.ndim(), count);
    if (values.shape() != expect) {
        throw std::runtime_error("mask: scatter values shape must match gathered shape");
    }
    const size_t mnd = mask.ndim();
    const size_t n_mask = mask.shape().numel();
    const size_t trail = trailing_product(this->_shape, mnd);
    if (this->_is_contiguous && mask.is_contiguous() && values.is_contiguous()) {
        T* dst = this->data();
        const T* src = values.data();
        const uint8_t* m = mask.data();
        size_t k = 0;
        if (trail == 1) {
            for (size_t i = 0; i < n_mask; ++i) {
                if (m[i]) {
                    dst[i] = src[k++];
                }
            }
        } else {
            const size_t nbytes = trail * sizeof(T);
            for (size_t i = 0; i < n_mask; ++i) {
                if (m[i]) {
                    std::memcpy(dst + i * trail, src + k * trail, nbytes);
                    ++k;
                }
            }
        }
        return;
    }
    std::vector<size_t> idx(mnd, 0);
    size_t k = 0;
    for (size_t t = 0; t < n_mask; ++t) {
        if (mask[Shape(idx)]) {
            write_slab(*this, idx, values, k);
            ++k;
        }
        bump_index(idx, mask.shape());
    }
}

Array<uint8_t> mask_and(const Array<uint8_t>& a, const Array<uint8_t>& b) {
#ifdef __AVX2__
    if (a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous()) {
        Array<uint8_t> out(a.shape());
        const size_t n = a.shape().numel();
        const uint8_t* pa = a.data();
        const uint8_t* pb = b.data();
        uint8_t* o = out.data();
        const __m256i z = _mm256_setzero_si256();
        const __m256i one = _mm256_set1_epi8(1);
        size_t i = 0;
        for (; i + 32 <= n; i += 32) {
            const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pa + i));
            const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb + i));
            const __m256i ba = _mm256_andnot_si256(_mm256_cmpeq_epi8(va, z), one);
            const __m256i bb = _mm256_andnot_si256(_mm256_cmpeq_epi8(vb, z), one);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + i), _mm256_and_si256(ba, bb));
        }
        for (; i < n; ++i) {
            o[i] = (pa[i] && pb[i]) ? 1 : 0;
        }
        return out;
    }
#endif
    return mask_zip(a, b, [](uint8_t x, uint8_t y) { return x && y; });
}
Array<uint8_t> mask_or(const Array<uint8_t>& a, const Array<uint8_t>& b) {
#ifdef __AVX2__
    if (a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous()) {
        Array<uint8_t> out(a.shape());
        const size_t n = a.shape().numel();
        const uint8_t* pa = a.data();
        const uint8_t* pb = b.data();
        uint8_t* o = out.data();
        const __m256i z = _mm256_setzero_si256();
        const __m256i one = _mm256_set1_epi8(1);
        size_t i = 0;
        for (; i + 32 <= n; i += 32) {
            const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pa + i));
            const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb + i));
            const __m256i ba = _mm256_andnot_si256(_mm256_cmpeq_epi8(va, z), one);
            const __m256i bb = _mm256_andnot_si256(_mm256_cmpeq_epi8(vb, z), one);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + i), _mm256_or_si256(ba, bb));
        }
        for (; i < n; ++i) {
            o[i] = (pa[i] || pb[i]) ? 1 : 0;
        }
        return out;
    }
#endif
    return mask_zip(a, b, [](uint8_t x, uint8_t y) { return x || y; });
}
Array<uint8_t> mask_xor(const Array<uint8_t>& a, const Array<uint8_t>& b) {
#ifdef __AVX2__
    if (a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous()) {
        Array<uint8_t> out(a.shape());
        const size_t n = a.shape().numel();
        const uint8_t* pa = a.data();
        const uint8_t* pb = b.data();
        uint8_t* o = out.data();
        const __m256i z = _mm256_setzero_si256();
        const __m256i one = _mm256_set1_epi8(1);
        size_t i = 0;
        for (; i + 32 <= n; i += 32) {
            const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pa + i));
            const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb + i));
            const __m256i ba = _mm256_andnot_si256(_mm256_cmpeq_epi8(va, z), one);
            const __m256i bb = _mm256_andnot_si256(_mm256_cmpeq_epi8(vb, z), one);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + i), _mm256_xor_si256(ba, bb));
        }
        for (; i < n; ++i) {
            o[i] = (bool(pa[i]) != bool(pb[i])) ? 1 : 0;
        }
        return out;
    }
#endif
    return mask_zip(a, b, [](uint8_t x, uint8_t y) { return bool(x) != bool(y); });
}
Array<uint8_t> mask_not(const Array<uint8_t>& a) {
    Array<uint8_t> out(a.shape());
    const size_t n = a.shape().numel();
    if (a.is_contiguous()) {
        const uint8_t* p = a.data();
        uint8_t* o = out.data();
#ifdef __AVX2__
        const __m256i z = _mm256_setzero_si256();
        const __m256i one = _mm256_set1_epi8(1);
        size_t i = 0;
        for (; i + 32 <= n; i += 32) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i));
            // 1 if zero, else 0
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(o + i),
                _mm256_and_si256(_mm256_cmpeq_epi8(v, z), one)
            );
        }
        for (; i < n; ++i) {
            o[i] = p[i] ? 0 : 1;
        }
#else
        for (size_t i = 0; i < n; ++i) {
            o[i] = p[i] ? 0 : 1;
        }
#endif
        return out;
    }
    std::vector<size_t> idx(a.ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        const Shape s(idx);
        out[s] = a[s] ? 0 : 1;
        bump_index(idx, a.shape());
    }
    return out;
}

#pragma endregion public math API

#pragma region shape / view API

template<typename T>
Array<T> Array<T>::operator[](const Slice& s) const {
    return (*this)[std::vector<Slice>{s}];
}

template<typename T>
Array<T> Array<T>::operator[](std::initializer_list<Slice> axes) const {
    return (*this)[std::vector<Slice>(axes)];
}

template<typename T>
Array<T> Array<T>::operator[](const std::vector<Slice>& axes) const {
    if (axes.size() > this->ndim()) {
        throw std::runtime_error("slice: more axes than ndim");
    }
    std::vector<size_t> new_shape;
    std::vector<size_t> new_strides;
    new_shape.reserve(this->ndim());
    new_strides.reserve(this->ndim());
    size_t offset = this->_offset;
    for (size_t d = 0; d < this->ndim(); ++d) {
        const Slice sl = (d < axes.size()) ? axes[d] : Slice();
        const Slice::Resolved r = sl.resolve(this->_shape[d]);
        offset += r.start * this->_strides[d];
        new_shape.push_back(r.length);
        new_strides.push_back(this->_strides[d] * r.step);
    }
    return Array<T>(this->_data, Shape(new_shape), Shape(new_strides), offset);
}

template<typename T>
Array<T> Array<T>::view(const Shape& view_shape) const {
    if (this->_shape.numel() != view_shape.numel()) {
        throw std::runtime_error("view: shape mismatch");
    }
    return Array<T>(this->_data, view_shape, view_shape.strides(), this->_offset);
}

template<typename T>
Array<T> Array<T>::reshape(const Shape& reshape_shape) const {
    if (this->_shape.numel() != reshape_shape.numel()) {
        throw std::runtime_error("reshape: shape mismatch");
    }
    return Array<T>(this->_data, reshape_shape, reshape_shape.strides(), this->_offset);
}

template<typename T>
Array<T> Array<T>::flatten() const {
    if (this->_shape.numel() == 0) {
        throw std::runtime_error("flatten: shape is empty");
    }
    return Array<T>(this->_data, Shape(this->_shape.numel()), Shape(this->_shape.numel()).strides(), this->_offset);
}

template<typename T>
Array<T> Array<T>::contiguous() const {
    return copy_to_contiguous(*this, this->_shape);
}

template<typename T>
void Array<T>::_contiguous() {
    if (this->_is_contiguous) {
        return;
    }
    *this = copy_to_contiguous(*this, this->_shape);
}

template<typename T>
Array<T> Array<T>::transpose() const {
    const size_t nd = this->ndim();
    if (nd <= 1) {
        return Array<T>(this->_data, this->_shape, this->_strides, this->_offset);
    }
    std::vector<size_t> new_shape(nd);
    std::vector<size_t> new_strides(nd);
    for (size_t i = 0; i < nd; ++i) {
        new_shape[i] = this->_shape[nd - 1 - i];
        new_strides[i] = this->_strides[nd - 1 - i];
    }
    return Array<T>(this->_data, Shape(new_shape), Shape(new_strides), this->_offset);
}

template<typename T>
Array<T> Array<T>::permute(const Shape& axes) const {
    const size_t nd = this->ndim();
    if (axes.ndim() != nd) {
        throw std::runtime_error("permute: axes rank must match array ndim");
    }
    std::vector<bool> seen(nd, false);
    std::vector<size_t> new_shape(nd);
    std::vector<size_t> new_strides(nd);
    for (size_t i = 0; i < nd; ++i) {
        const size_t ax = axes[i];
        if (ax >= nd) {
            throw std::runtime_error("permute: axis out of bounds");
        }
        if (seen[ax]) {
            throw std::runtime_error("permute: duplicate axis");
        }
        seen[ax] = true;
        new_shape[i] = this->_shape[ax];
        new_strides[i] = this->_strides[ax];
    }
    return Array<T>(this->_data, Shape(new_shape), Shape(new_strides), this->_offset);
}

template<typename T>
Array<T> Array<T>::squeeze(const size_t axis) const {
    if (axis >= this->_shape.ndim()) {
        throw std::runtime_error("squeeze: axis out of bounds");
    }
    if (this->_shape[axis] != 1) {
        throw std::runtime_error("squeeze: axis is not a singleton");
    }
    std::vector<size_t> new_shape;
    std::vector<size_t> new_strides;
    new_shape.reserve(this->_shape.ndim() - 1);
    new_strides.reserve(this->_shape.ndim() - 1);
    for (size_t i = 0; i < this->_shape.ndim(); ++i) {
        if (i == axis) {
            continue;
        }
        new_shape.push_back(this->_shape[i]);
        new_strides.push_back(this->_strides[i]);
    }
    return Array<T>(this->_data, Shape(new_shape), Shape(new_strides), this->_offset);
}

template<typename T>
Array<T> Array<T>::unsqueeze(const size_t axis) const {
    if (axis > this->_shape.ndim()) {
        throw std::runtime_error("unsqueeze: axis out of bounds");
    }
    std::vector<size_t> new_shape;
    std::vector<size_t> new_strides;
    new_shape.reserve(this->_shape.ndim() + 1);
    new_strides.reserve(this->_shape.ndim() + 1);
    for (size_t i = 0; i <= this->_shape.ndim(); ++i) {
        if (i == axis) {
            new_shape.push_back(1);
            new_strides.push_back(0);
        } else {
            const size_t src = i < axis ? i : i - 1;
            new_shape.push_back(this->_shape[src]);
            new_strides.push_back(this->_strides[src]);
        }
    }
    return Array<T>(this->_data, Shape(new_shape), Shape(new_strides), this->_offset);
}

template<typename T>
void Array<T>::_view(const Shape& new_shape) { *this = this->view(new_shape); }
template<typename T>
void Array<T>::_reshape(const Shape& new_shape) { *this = this->reshape(new_shape); }
template<typename T>
void Array<T>::_flatten() { *this = this->flatten(); }
template<typename T>
void Array<T>::_transpose() { *this = this->transpose(); }
template<typename T>
void Array<T>::_permute(const Shape& axes) { *this = this->permute(axes); }
template<typename T>
void Array<T>::_squeeze(const size_t axis) { *this = this->squeeze(axis); }
template<typename T>
void Array<T>::_unsqueeze(const size_t axis) { *this = this->unsqueeze(axis); }

#pragma endregion shape / view API

#pragma endregion math operations

template class Array<float>;
template class Array<double>;
template class Array<int>;
template class Array<uint8_t>;

} // namespace cthreads::linalg
