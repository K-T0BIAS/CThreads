#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>

namespace cthreads::linalg {

    class Shape {
        private:
            std::vector<size_t> _shape;
        public:
            Shape(const std::vector<size_t>& shape) : _shape(shape) {}
            Shape(size_t value) : _shape({value}) {}

            ~Shape() = default;
            
            Shape(const Shape& other) : _shape(other._shape) {}
            Shape(Shape&& other) noexcept : _shape(std::move(other._shape)) {}

#pragma region operators

            Shape& operator=(const Shape& other) {
                this->_shape = other._shape;
                return *this;
            }
            Shape& operator=(Shape&& other) noexcept {
                this->_shape = std::move(other._shape);
                return *this;
            }

            operator std::vector<size_t>() const { return this->_shape; }

            size_t operator[](size_t index) const { return this->_shape[index]; }
            size_t& operator[](size_t index) { return this->_shape[index]; }

            bool operator==(const Shape& other) const { return this->_shape == other._shape; }
            bool operator!=(const Shape& other) const { return this->_shape != other._shape; }

            Shape operator+(const Shape& other) const { 
                if (this->_shape.size() != other._shape.size()) {
                    throw std::runtime_error("Shape dimensions must match");
                }
                std::vector<size_t> result(this->_shape.size());
                for (size_t i = 0; i < this->_shape.size(); i++) {
                    result[i] = this->_shape[i] + other._shape[i];
                }
                return Shape(result);
            }
            Shape operator-(const Shape& other) const { 
                if (this->_shape.size() != other._shape.size()) {
                    throw std::runtime_error("Shape dimensions must match");
                }
                std::vector<size_t> result(this->_shape.size());
                for (size_t i = 0; i < this->_shape.size(); i++) {
                    result[i] = this->_shape[i] - other._shape[i];
                }
                return Shape(result);
            }
            Shape operator*(const Shape& other) const { 
                if (this->_shape.size() != other._shape.size()) {
                    throw std::runtime_error("Shape dimensions must match");
                }
                std::vector<size_t> result(this->_shape.size());
                for (size_t i = 0; i < this->_shape.size(); i++) {
                    result[i] = this->_shape[i] * other._shape[i];
                }
                return Shape(result);
            }
            Shape operator/(const Shape& other) const { 
                if (this->_shape.size() != other._shape.size()) {
                    throw std::runtime_error("Shape dimensions must match");
                }
                std::vector<size_t> result(this->_shape.size());
                for (size_t i = 0; i < this->_shape.size(); i++) {
                    result[i] = this->_shape[i] / other._shape[i];
                }
                return Shape(result);
            }

#pragma endregion operators

#pragma region properties

            size_t size() const { return this->_shape.size(); }

#pragma endregion properties

#pragma region helpers

            // C-contiguous / row-major element strides (last axis varies fastest).
            // shape [D0, D1, D2] -> strides [D1*D2, D2, 1]
            // e.g. 3x3x3 -> [9, 3, 1]:
            //   [i, :, :] = 9 contiguous elems (matrix / plane)
            //   [i, j, :] = 3 contiguous elems (vector / row)
            Shape strides() const { 
                if (this->_shape.empty()) {
                    return Shape(std::vector<size_t>{});
                }
                std::vector<size_t> strides(this->_shape.size());
                strides.back() = 1;
                for (size_t i = this->_shape.size(); i-- > 1; ) {
                    strides[i - 1] = strides[i] * this->_shape[i];
                }
                return Shape(strides);
            }

            // Flat offset for a full multi-index into a *contiguous* array of this shape.
            // Uses freshly computed row-major strides (not stored view strides).
            size_t get_index(const Shape& index) const {
                if (index.size() != this->_shape.size()) {
                    throw std::runtime_error("index rank must match shape ndim");
                }
                size_t flat_index = 0;
                Shape strides = this->strides();
                for (size_t i = 0; i < this->_shape.size(); i++) {
                    flat_index += index[i] * strides[i];
                }
                return flat_index;
            }

            size_t numel() const {
                if (this->_shape.size() == 0) {
                    return 0;
                }
                size_t numel = 1;
                for (size_t i = 0; i < this->_shape.size(); i++) {
                    numel *= this->_shape[i];
                }
                return numel;
            }

            size_t ndim() const { return this->_shape.size(); }


#pragma endregion helpers
        };
}