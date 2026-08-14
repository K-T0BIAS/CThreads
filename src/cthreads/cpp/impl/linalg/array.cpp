#include "../../headers/linalg/array.hpp"
#include "../../headers/linalg/shape.hpp"
#include "../../headers/linalg/tiling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef __AVX2__ // if available use the avx2 intrinsics for vectorization
#include <immintrin.h>
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
    Array<T> out(new_shape);
    const size_t n = new_shape.numel();
    if (n == 0) {
        return out;
    }
    const Shape& src_shape = src.shape();
    const size_t nd = src_shape.ndim();
    std::vector<size_t> idx(nd, 0);
    for (size_t flat = 0; flat < n; ++flat) {
        out[flat] = src[Shape(idx)];
        for (int d = static_cast<int>(nd) - 1; d >= 0; --d) {
            ++idx[static_cast<size_t>(d)];
            if (idx[static_cast<size_t>(d)] < src_shape[static_cast<size_t>(d)]) {
                break;
            }
            idx[static_cast<size_t>(d)] = 0;
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

size_t count_true(const Array<uint8_t>& mask) {
    const size_t n = mask.shape().numel();
    size_t c = 0;
    if (mask.is_contiguous()) {
        const uint8_t* p = mask.data();
        for (size_t i = 0; i < n; ++i) {
            c += p[i] ? 1 : 0;
        }
        return c;
    }
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

template<typename T, typename Pred>
Array<uint8_t> compare_walk(const Array<T>& a, Pred pred) {
    Array<uint8_t> out(a.shape());
    const size_t n = a.shape().numel();
    if (a.is_contiguous()) {
        const T* p = a.data();
        uint8_t* o = out.data();
        for (size_t i = 0; i < n; ++i) {
            o[i] = pred(p[i]) ? 1 : 0;
        }
        return out;
    }
    std::vector<size_t> idx(a.ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        const Shape s(idx);
        out[s] = pred(a[s]) ? 1 : 0;
        bump_index(idx, a.shape());
    }
    return out;
}

template<typename T, typename Pred>
Array<uint8_t> compare_walk(const Array<T>& a, const Array<T>& b, Pred pred) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("compare: shape mismatch");
    }
    Array<uint8_t> out(a.shape());
    const size_t n = a.shape().numel();
    if (a.is_contiguous() && b.is_contiguous()) {
        const T* pa = a.data();
        const T* pb = b.data();
        uint8_t* o = out.data();
        for (size_t i = 0; i < n; ++i) {
            o[i] = pred(pa[i], pb[i]) ? 1 : 0;
        }
        return out;
    }
    std::vector<size_t> idx(a.ndim(), 0);
    for (size_t t = 0; t < n; ++t) {
        const Shape s(idx);
        out[s] = pred(a[s], b[s]) ? 1 : 0;
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

/**
* Compute the matrix multiplication of two matrices (dense row-major batched)
* Uses AVX2 SIMD if available otherwise naive element wise loop
* The result is in place written to the out array
*
* #### args
* - out: output array
* - lhs: left hand side matrix
* - rhs: right hand side matrix
*
* #### returns
* - void
*
* #### throws
* - std::runtime_error: if the shapes are invalid
*/
template<typename T>
void Array<T>::_fast_matmul(Array& out, const Array& lhs, const Array& rhs) {
    // Dense row-major batched: lhs (..., M, K) @ rhs (..., K, N) -> out (..., M, N)
    if (!(lhs.is_contiguous() && rhs.is_contiguous() && out.is_contiguous())) {
        _fast_matmul_scalar(out, lhs, rhs);
        return;
    }
    const MatmulSpec spec = resolve_matmul(lhs.shape(), rhs.shape()); // create the spec for the matmul (May throw if invalid shapes)
    if (out.shape() != spec.out_shape) {
        throw std::runtime_error("matmul: incompatible output shape");
    }
    // get the dimensions of the matrices
    const size_t M = spec.M; // num rows in the lhs
    const size_t K = spec.K; // num cols in the lhs and num rows in the rhs
    const size_t N = spec.N; // num cols in the rhs
    const Shape row_shape(std::vector<size_t>{M, K}); // lhs matrix shape (M, K)

#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {  // float32
        std::vector<float> b_cols(K * N); 
        const bool pack_once = (spec.b_stride == 0); // check if theres a batch dim
        if (pack_once) { // batch dim exists and isnt 1 so broadast is not required
            const float* b0 = rhs.data();
            for (size_t j = 0; j < N; ++j) {
                float* col = b_cols.data() + j * K;
                for (size_t k = 0; k < K; ++k) {
                    col[k] = b0[k * N + j];
                }
            }
        } 
        
        // build the mem layout over the batch dims (takes the arrays inner data and copies into float* arrays)
        for (size_t bi = 0; bi < spec.batch; ++bi) {
            const float* a = lhs.data() + bi * spec.a_stride;
            const float* b = rhs.data() + bi * spec.b_stride;
            float* c = out.data() + bi * spec.c_stride;
            if (!pack_once) { // rhs has no batch dim so we need to broadcast here
                for (size_t j = 0; j < N; ++j) {
                    float* col = b_cols.data() + j * K;
                    for (size_t k = 0; k < K; ++k) {
                        col[k] = b[k * N + j];
                    }
                }
            }
            // pack the lhs into row wise tiles for cache locality
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
    T* a = lhs.data(); // lhs data pointer
    const T* b = rhs.data(); // rhs data pointer
    const size_t n = lhs.shape().numel(); // number of elements in the arrays

#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) { // float32
        size_t i = 0; // index
        for (; i + 8 <= n; i += 8) { // itter over 8 elements at a time
            __m256 va = _mm256_loadu_ps(a + i); // load 8 elements from a
            __m256 vb = _mm256_loadu_ps(b + i); // load 8 elements from b
            _mm256_storeu_ps(a + i, _mm256_add_ps(va, vb)); // add and store 8 elements back to a
        }
        for (; i < n; ++i) { // itter over the remaining elements
            a[i] += b[i]; // add the elements
        }
    } else if constexpr (std::is_same_v<T, double>) { // float64
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
    for (size_t i = 0; i < n; ++i) { // fallback: no AVX2 simple element wise loop
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
    Array<T> out = this->contiguous();
    _fast_add(out, other);
    return out;
}

template<typename T>
Array<T> Array<T>::operator-(const Array& other) const {
    Array<T> out = this->contiguous();
    _fast_sub(out, other);
    return out;
}

template<typename T>
Array<T> Array<T>::operator*(const Array& other) const {
    Array<T> out = this->contiguous();
    _fast_mul(out, other);
    return out;
}

template<typename T>
Array<T> Array<T>::operator/(const Array& other) const {
    Array<T> out = this->contiguous();
    _fast_div(out, other);
    return out;
}

template<typename T>
Array<T> Array<T>::operator+(T value) const {
    Array<T> out = this->contiguous();
    _fast_add(out, value);
    return out;
}

template<typename T>
Array<T> Array<T>::operator-(T value) const {
    Array<T> out = this->contiguous();
    _fast_sub(out, value);
    return out;
}

template<typename T>
Array<T> Array<T>::operator*(T value) const {
    Array<T> out = this->contiguous();
    _fast_mul(out, value);
    return out;
}

template<typename T>
Array<T> Array<T>::operator/(T value) const {
    Array<T> out = this->contiguous();
    _fast_div(out, value);
    return out;
}

template<typename T>
Array<T> Array<T>::operator-() const {
    Array<T> out = this->contiguous();
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

template<typename T>
Array<uint8_t> Array<T>::operator>(T value) const {
    return compare_walk(*this, [value](T x) { return x > value; });
}
template<typename T>
Array<uint8_t> Array<T>::operator<(T value) const {
    return compare_walk(*this, [value](T x) { return x < value; });
}
template<typename T>
Array<uint8_t> Array<T>::operator>=(T value) const {
    return compare_walk(*this, [value](T x) { return x >= value; });
}
template<typename T>
Array<uint8_t> Array<T>::operator<=(T value) const {
    return compare_walk(*this, [value](T x) { return x <= value; });
}
template<typename T>
Array<uint8_t> Array<T>::operator==(T value) const {
    return compare_walk(*this, [value](T x) { return x == value; });
}
template<typename T>
Array<uint8_t> Array<T>::operator!=(T value) const {
    return compare_walk(*this, [value](T x) { return x != value; });
}
template<typename T>
Array<uint8_t> Array<T>::operator>(const Array& other) const {
    return compare_walk(*this, other, [](T x, T y) { return x > y; });
}
template<typename T>
Array<uint8_t> Array<T>::operator<(const Array& other) const {
    return compare_walk(*this, other, [](T x, T y) { return x < y; });
}
template<typename T>
Array<uint8_t> Array<T>::operator>=(const Array& other) const {
    return compare_walk(*this, other, [](T x, T y) { return x >= y; });
}
template<typename T>
Array<uint8_t> Array<T>::operator<=(const Array& other) const {
    return compare_walk(*this, other, [](T x, T y) { return x <= y; });
}
template<typename T>
Array<uint8_t> Array<T>::operator==(const Array& other) const {
    return compare_walk(*this, other, [](T x, T y) { return x == y; });
}
template<typename T>
Array<uint8_t> Array<T>::operator!=(const Array& other) const {
    return compare_walk(*this, other, [](T x, T y) { return x != y; });
}

template<typename T>
size_t Array<T>::count_nonzero() const {
    const size_t n = this->_shape.numel();
    size_t c = 0;
    if (this->_is_contiguous) {
        const T* p = this->data();
        for (size_t i = 0; i < n; ++i) {
            c += (p[i] != T{}) ? 1 : 0;
        }
        return c;
    }
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
    std::vector<size_t> idx(mask.ndim(), 0);
    const size_t n = mask.shape().numel();
    size_t k = 0;
    for (size_t t = 0; t < n; ++t) {
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
    std::vector<size_t> idx(mask.ndim(), 0);
    const size_t n = mask.shape().numel();
    for (size_t t = 0; t < n; ++t) {
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
    std::vector<size_t> idx(mask.ndim(), 0);
    const size_t n = mask.shape().numel();
    size_t k = 0;
    for (size_t t = 0; t < n; ++t) {
        if (mask[Shape(idx)]) {
            write_slab(*this, idx, values, k);
            ++k;
        }
        bump_index(idx, mask.shape());
    }
}

Array<uint8_t> mask_and(const Array<uint8_t>& a, const Array<uint8_t>& b) {
    return mask_zip(a, b, [](uint8_t x, uint8_t y) { return x && y; });
}
Array<uint8_t> mask_or(const Array<uint8_t>& a, const Array<uint8_t>& b) {
    return mask_zip(a, b, [](uint8_t x, uint8_t y) { return x || y; });
}
Array<uint8_t> mask_xor(const Array<uint8_t>& a, const Array<uint8_t>& b) {
    return mask_zip(a, b, [](uint8_t x, uint8_t y) { return bool(x) != bool(y); });
}
Array<uint8_t> mask_not(const Array<uint8_t>& a) {
    Array<uint8_t> out(a.shape());
    const size_t n = a.shape().numel();
    if (a.is_contiguous()) {
        const uint8_t* p = a.data();
        uint8_t* o = out.data();
        for (size_t i = 0; i < n; ++i) {
            o[i] = p[i] ? 0 : 1;
        }
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
