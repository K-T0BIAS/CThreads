#pragma once

#include <memory>
#include <vector>

namespace cthreads::linalg {

    template<typename T>
    class Data {
        private:
            std::shared_ptr<std::vector<T>> data;

        public:
            explicit Data(size_t size) : data(std::make_shared<std::vector<T>>(size)) {}
            ~Data() = default;
            size_t size() const { return this->data->size(); }

            T& operator[](size_t i) { return (*this->data)[i]; }
            const T& operator[](size_t i) const { return (*this->data)[i]; }

            T* ptr() { return this->data->data(); }
            const T* ptr() const { return this->data->data(); }
    };
}
