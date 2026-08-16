#pragma once

#include "inner.hpp"
#include "shape.hpp"
#include "slice.hpp"
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

namespace cthreads::linalg {

    template<typename T>
    class Array {
        private:
            std::shared_ptr<Data<T>> _data; // shared flat storage (row-major / C-contiguous)
            Shape _shape;     // logical shape of this view
            Shape _strides;   // element strides; default = shape.strides() (last axis contiguous)
            bool _is_contiguous; // true iff _strides == _shape.strides() (AVX2 kernels require this)
            size_t _offset = 0; // element offset into _data (slices / dropped axes)

        public:
            // Allocates a dense row-major buffer: last dim is contiguous (vectors),
            // then trailing planes (matrices), etc.
            Array(const Shape& shape) :
                _shape(shape),
                _strides(shape.strides()),
                _is_contiguous(true),
                _offset(0)
            {
                size_t numel = shape.numel();
                this->_data = std::make_shared<Data<T>>(numel);
            }

            Array(std::shared_ptr<Data<T>> data, const Shape& shape) :
                _shape(shape),
                _strides(shape.strides()),
                _data(data),
                _is_contiguous(true),
                _offset(0) {}

            // View into existing storage with explicit strides (transpose / permute / slice).
            Array(std::shared_ptr<Data<T>> data, const Shape& shape, const Shape& strides, size_t offset = 0) :
                _shape(shape),
                _strides(strides),
                _data(data),
                _is_contiguous(strides == shape.strides()),
                _offset(offset) {}

            ~Array() = default;

            Array(const Array& other) :
                _shape(other._shape),
                _strides(other._strides),
                _data(other._data),
                _is_contiguous(other._is_contiguous),
                _offset(other._offset) {}

            Array(Array&& other) noexcept :
                _shape(std::move(other._shape)),
                _strides(std::move(other._strides)),
                _data(std::move(other._data)),
                _is_contiguous(other._is_contiguous),
                _offset(other._offset) {}

            Array& operator=(const Array& other) {
                this->_shape = other._shape;
                this->_strides = other._strides;
                this->_data = other._data;
                this->_is_contiguous = other._is_contiguous;
                this->_offset = other._offset;
                return *this;
            }

            Array& operator=(Array&& other) noexcept {
                this->_shape = std::move(other._shape);
                this->_strides = std::move(other._strides);
                this->_data = std::move(other._data);
                this->_is_contiguous = other._is_contiguous;
                this->_offset = other._offset;
                return *this;
            }

            size_t size() const { return this->_data->size(); }
            size_t ndim() const { return this->_shape.size(); }
            
            const Shape& shape() const { return this->_shape; }
            const Shape& strides() const { return this->_strides; }
            bool is_contiguous() const { return this->_is_contiguous; }
            size_t offset() const { return this->_offset; }

            T* data() { return this->_data->ptr() + this->_offset; }
            const T* data() const { return this->_data->ptr() + this->_offset; }

#pragma region indexing

            public:

            T& operator[](const Shape& index) {
                return (*this->_data)[this->get_index(index)];
            }
            const T& operator[](const Shape& index) const {
                return (*this->_data)[this->get_index(index)];
            }

            // Flat index relative to this view (data()[index]); valid when C-contiguous.
            T& operator[](size_t index) {
                return (*this->_data)[this->_offset + index];
            }
            const T& operator[](size_t index) const {
                return (*this->_data)[this->_offset + index];
            }

            // Zero-copy slice. Single Slice applies to axis 0; remaining axes stay full.
            Array operator[](const Slice& s) const;
            Array operator[](const std::vector<Slice>& axes) const;
            Array operator[](std::initializer_list<Slice> axes) const;

            // Boolean prefix mask (numpy-style, no broadcast):
            //   mask.shape == this.shape[:mask.ndim]
            //   result.shape == (count_true,) + this.shape[mask.ndim:]
            // Gather into a new contiguous buffer (not a view).
            Array masked_select(const Array<uint8_t>& mask) const;
            void masked_fill(const Array<uint8_t>& mask, T value);
            void masked_scatter(const Array<uint8_t>& mask, const Array& values);

            private: 

            // Flat offset using *stored* _strides and view _offset.
            size_t get_index(const Shape& index) const {
                if (index.size() != this->_shape.size()) {
                    throw std::runtime_error("index rank must match array ndim");
                }
                size_t flat = this->_offset;
                for (size_t i = 0; i < index.size(); ++i) {
                    flat += index[i] * this->_strides[i];
                }
                return flat;
            }

#pragma endregion indexing

#pragma region shape manipulation // zero-copy views (same data ptr); _fns mutate this

            public:

            Array view(const Shape& new_shape) const;
            Array reshape(const Shape& new_shape) const;
            Array flatten() const;
            Array transpose() const;
            Array permute(const Shape& new_shape) const;
            Array squeeze(const size_t axis) const;
            Array unsqueeze(const size_t axis) const;
            Array contiguous() const; // always a dense C-contiguous copy

            void _view(const Shape& new_shape);
            void _reshape(const Shape& new_shape);
            void _flatten();
            void _transpose();
            void _permute(const Shape& new_shape);
            void _squeeze(const size_t axis);
            void _unsqueeze(const size_t axis);
            void _contiguous(); // inplace: replace storage with a dense copy if needed

            private:

#pragma endregion shape manipulation

#pragma region math operations

            public:

            Array operator+(const Array& other) const; // elementwise addition between two arrays of equal shape-> interal fallback on _fast_add	
            Array operator-(const Array& other) const; // elementwise subtraction between two arrays of equal shape-> interal fallback on _fast_sub
            Array operator*(const Array& other) const; // elementwise multiplication between two arrays of equal shape-> interal fallback on _fast_mul
            Array operator/(const Array& other) const; // elementwise division between two arrays of equal shape-> interal fallback on _fast_div

            Array operator+(T value) const; // elementwise addition between an array and a scalar-> interal fallback on _fast_add
            Array operator-(T value) const; // elementwise subtraction between an array and a scalar-> interal fallback on _fast_sub
            Array operator*(T value) const; // elementwise multiplication between an array and a scalar-> interal fallback on _fast_mul
            Array operator/(T value) const; // elementwise division between an array and a scalar-> interal fallback on _fast_div

            Array operator-() const; // negation of an array-> interal fallback on _fast_neg

            Array matmul(const Array& other, bool parallel = false) const; // matrix multiplication; parallel=true panel-threads large GEMMs
            Array matmul_scalar(const Array& other) const; // elementwise/naive GEMM (bench baseline) (should not be used for the py side bindings, use matmul instead)
            Array dot(const Array& other) const;
            Array dot_scalar(const Array& other) const;    // elementwise/naive dot (bench baseline)
            Array cross(const Array& other) const;
            void _matmul(const Array& other);
            void _dot(const Array& other);
            void _cross(const Array& other);

            void _add(const Array& other);
            void _sub(const Array& other);
            void _mul(const Array& other);
            void _div(const Array& other);

            void _add(T value);
            void _sub(T value);
            void _mul(T value);
            void _div(T value);

            void _neg();

            Array<uint8_t> operator>(T value) const;
            Array<uint8_t> operator<(T value) const;
            Array<uint8_t> operator>=(T value) const;
            Array<uint8_t> operator<=(T value) const;
            Array<uint8_t> operator==(T value) const;
            Array<uint8_t> operator!=(T value) const;
            Array<uint8_t> operator>(const Array& other) const;
            Array<uint8_t> operator<(const Array& other) const;
            Array<uint8_t> operator>=(const Array& other) const;
            Array<uint8_t> operator<=(const Array& other) const;
            Array<uint8_t> operator==(const Array& other) const;
            Array<uint8_t> operator!=(const Array& other) const;

            size_t count_nonzero() const;

            private: // internal math mathods that handle the compute (the above use these) Implemented using avx2 SIMD if available otherwise naive element wise loops

            // Kernels take Array so shape/strides are available (needed for matmul tiling).
            static T _fast_inner_product(const Array& lhs, const Array& rhs);
            static T _fast_inner_product_scalar(const Array& lhs, const Array& rhs);
            static void _fast_cross_product(Array& lhs, const Array& rhs);
            static void _fast_matmul(Array& out, const Array& lhs, const Array& rhs, bool parallel = false);
            static void _fast_matmul_scalar(Array& out, const Array& lhs, const Array& rhs);

            static void _fast_add(Array& lhs, const Array& rhs);
            static void _fast_sub(Array& lhs, const Array& rhs);
            static void _fast_mul(Array& lhs, const Array& rhs);
            static void _fast_div(Array& lhs, const Array& rhs);
            static void _fast_add(Array& lhs, T value);
            static void _fast_sub(Array& lhs, T value);
            static void _fast_mul(Array& lhs, T value);
            static void _fast_div(Array& lhs, T value);
            static void _fast_neg(Array& lhs);

#pragma endregion math operations

    };

    Array<uint8_t> mask_and(const Array<uint8_t>& a, const Array<uint8_t>& b);
    Array<uint8_t> mask_or(const Array<uint8_t>& a, const Array<uint8_t>& b);
    Array<uint8_t> mask_xor(const Array<uint8_t>& a, const Array<uint8_t>& b);
    Array<uint8_t> mask_not(const Array<uint8_t>& a);
}
