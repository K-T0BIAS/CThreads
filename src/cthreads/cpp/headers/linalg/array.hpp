#pragma once

#include "inner.hpp"
#include "shape.hpp"
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

        public:
            // Allocates a dense row-major buffer: last dim is contiguous (vectors),
            // then trailing planes (matrices), etc.
            Array(const Shape& shape) :
                _shape(shape),
                _strides(shape.strides())
            {
                size_t numel = shape.numel();
                this->_data = std::make_shared<Data<T>>(numel);
            }

            ~Array() = default;

            Array(const Array& other) :
                _shape(other._shape),
                _strides(other._strides),
                _data(other._data) {}

            Array(Array&& other) noexcept :
                _shape(std::move(other._shape)),
                _strides(std::move(other._strides)),
                _data(std::move(other._data)) {}

            Array& operator=(const Array& other) {
                this->_shape = other._shape;
                this->_strides = other._strides;
                this->_data = other._data;
                return *this;
            }

            Array& operator=(Array&& other) noexcept {
                this->_shape = std::move(other._shape);
                this->_strides = std::move(other._strides);
                this->_data = std::move(other._data);
                return *this;
            }

            size_t size() const { return this->_data->size(); }
            size_t ndim() const { return this->_shape.size(); }
            
            const Shape& shape() const { return this->_shape; }
            const Shape& strides() const { return this->_strides; }

            T* data() { return this->_data->ptr(); }
            const T* data() const { return this->_data->ptr(); }

#pragma region indexing

            public:

            T& operator[](const Shape& index) {
                return (*this->_data)[this->get_index(index)];
            }
            const T& operator[](const Shape& index) const {
                return (*this->_data)[this->get_index(index)];
            }

            // Flat storage index (walks memory in row-major order when contiguous).
            T& operator[](size_t index) {
                return (*this->_data)[index];
            }
            const T& operator[](size_t index) const {
                return (*this->_data)[index];
            }

            private: 

            // Flat offset using *stored* _strides (correct for views / transposed layouts).
            size_t get_index(const Shape& index) const {
                if (index.size() != this->_shape.size()) {
                    throw std::runtime_error("index rank must match array ndim");
                }
                size_t flat = 0;
                for (size_t i = 0; i < index.size(); ++i) {
                    flat += index[i] * this->_strides[i];
                }
                return flat;
            }

#pragma endregion indexing

#pragma region shape manipulation // view, reshape, flatten, transpose (returns new array)

            public:

            Array view(const Shape& new_shape) const;
            Array reshape(const Shape& new_shape) const;
            Array flatten() const;
            Array transpose() const;
            Array permute(const Shape& new_shape) const;
            Array squeeze(const Shape& new_shape) const;
            Array unsqueeze(const Shape& new_shape) const;

            private:

#pragma endregion shape manipulation

#pragma region data manipulation // inplace oprations on this array

            public:

            void _view(const Shape& new_shape);
            void _reshape(const Shape& new_shape);
            void _flatten();
            void _transpose();
            void _permute(const Shape& new_shape);
            void _squeeze(const Shape& new_shape);
            void _unsqueeze(const Shape& new_shape);

            private:

#pragma endregion data manipulation

#pragma region math operations

            public:



            Array operator+(const Array& other) const;
            Array operator-(const Array& other) const;
            Array operator*(const Array& other) const;
            Array operator/(const Array& other) const;

            Array operator+(T value) const;
            Array operator-(T value) const;
            Array operator*(T value) const;
            Array operator/(T value) const;

            Array operator-() const;

            Array matmul(const Array& other) const;
            Array matmul_scalar(const Array& other) const; // elementwise/naive GEMM (bench baseline)
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

            private:

            // Kernels take Array so shape/strides are available (needed for matmul tiling).
            static T _fast_inner_product(const Array& lhs, const Array& rhs);
            static T _fast_inner_product_scalar(const Array& lhs, const Array& rhs);
            static void _fast_cross_product(Array& lhs, const Array& rhs);
            static void _fast_matmul(Array& out, const Array& lhs, const Array& rhs);
            static void _fast_matmul_scalar(Array& out, const Array& lhs, const Array& rhs);

            static void _fast_add(Array& lhs, const Array& rhs);
            static void _fast_sub(Array& lhs, const Array& rhs);
            static void _fast_mul(Array& lhs, const Array& rhs);
            static void _fast_div(Array& lhs, const Array& rhs);
            static void _fast_neg(Array& lhs);

#pragma endregion math operations

    };
}
