#pragma once

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "../headers/linalg/array.hpp"
#include "../headers/linalg/shape.hpp"
#include "../headers/linalg/slice.hpp"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

inline std::vector<size_t> shape_dims(const cthreads::linalg::Shape& s) {
    std::vector<size_t> out(s.ndim());
    for (size_t i = 0; i < s.ndim(); ++i) {
        out[i] = s[i];
    }
    return out;
}

inline cthreads::linalg::Slice slice_from_py(py::slice sl, size_t dim) {
    py::ssize_t start = 0;
    py::ssize_t stop = 0;
    py::ssize_t step = 0;
    py::ssize_t slicelength = 0;
    if (!sl.compute(static_cast<py::ssize_t>(dim), &start, &stop, &step, &slicelength)) {
        throw py::error_already_set();
    }
    if (step < 1) {
        throw std::runtime_error("slice: step must be >= 1");
    }
    return cthreads::linalg::Slice(
        static_cast<size_t>(start),
        static_cast<size_t>(stop),
        static_cast<size_t>(step)
    );
}

inline size_t norm_index(py::ssize_t i, size_t dim) {
    py::ssize_t ii = i;
    if (ii < 0) {
        ii += static_cast<py::ssize_t>(dim);
    }
    if (ii < 0 || static_cast<size_t>(ii) >= dim) {
        throw py::index_error("index out of range");
    }
    return static_cast<size_t>(ii);
}

template<typename T>
inline std::vector<size_t> infer_nested_shape(py::handle obj) {
    std::vector<size_t> sh;
    py::handle cur = obj;
    while (py::isinstance<py::sequence>(cur) && !py::isinstance<py::str>(cur)) {
        py::sequence seq = cur.cast<py::sequence>();
        sh.push_back(static_cast<size_t>(seq.size()));
        if (seq.size() == 0) {
            break;
        }
        cur = seq[0];
    }
    if (sh.empty()) {
        throw py::type_error("from_list: expected a nested sequence");
    }
    return sh;
}

template<typename T>
inline void fill_from_nested(
    cthreads::linalg::Array<T>& a,
    py::handle obj,
    std::vector<size_t>& idx,
    size_t dim
) {
    using cthreads::linalg::Shape;
    if (dim == a.ndim()) {
        a[Shape(idx)] = obj.cast<T>();
        return;
    }
    py::sequence seq = obj.cast<py::sequence>();
    if (static_cast<size_t>(seq.size()) != a.shape()[dim]) {
        throw py::value_error("from_list: nested length does not match shape");
    }
    for (size_t i = 0; i < static_cast<size_t>(seq.size()); ++i) {
        idx[dim] = i;
        fill_from_nested<T>(a, seq[i], idx, dim + 1);
    }
}

template<typename T>
inline py::object to_nested(
    const cthreads::linalg::Array<T>& a,
    std::vector<size_t>& idx,
    size_t dim
) {
    using cthreads::linalg::Shape;
    if (dim == a.ndim()) {
        if constexpr (std::is_same_v<T, uint8_t>) {
            return py::bool_(static_cast<bool>(a[Shape(idx)]));
        } else {
            return py::cast(a[Shape(idx)]);
        }
    }
    py::list out;
    for (size_t i = 0; i < a.shape()[dim]; ++i) {
        idx[dim] = i;
        out.append(to_nested<T>(a, idx, dim + 1));
    }
    return out;
}

template<typename T>
inline py::object array_getitem(cthreads::linalg::Array<T>& self, py::handle key) {
    using Arr = cthreads::linalg::Array<T>;
    using Shape = cthreads::linalg::Shape;
    using Slice = cthreads::linalg::Slice;

    if (py::isinstance<py::int_>(key)) {
        const size_t i = norm_index(key.cast<py::ssize_t>(), self.shape()[0]);
        if (self.ndim() == 1) {
            return py::cast(self[Shape(std::vector<size_t>{i})]);
        }
        Arr row = self[Slice(i, i + 1)];
        return py::cast(row.squeeze(0));
    }

    if (py::isinstance<py::slice>(key)) {
        return py::cast(self[slice_from_py(key.cast<py::slice>(), self.shape()[0])]);
    }

    if (py::isinstance<cthreads::linalg::Array<uint8_t>>(key)) {
        return py::cast(self.masked_select(key.cast<cthreads::linalg::Array<uint8_t>>()));
    }

    if (py::isinstance<py::tuple>(key) || py::isinstance<py::list>(key)) {
        py::sequence seq = key.cast<py::sequence>();
        const size_t n = static_cast<size_t>(seq.size());
        bool all_int = true;
        bool all_slice = true;
        for (size_t i = 0; i < n; ++i) {
            py::handle h = seq[i];
            const bool is_int = py::isinstance<py::int_>(h);
            const bool is_sl = py::isinstance<py::slice>(h);
            all_int = all_int && is_int;
            all_slice = all_slice && is_sl;
        }
        if (all_int && n == self.ndim()) {
            std::vector<size_t> idx(n);
            for (size_t i = 0; i < n; ++i) {
                idx[i] = norm_index(seq[i].cast<py::ssize_t>(), self.shape()[i]);
            }
            return py::cast(self[Shape(idx)]);
        }
        std::vector<Slice> axes;
        std::vector<size_t> squeeze_axes;
        axes.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            py::handle h = seq[i];
            if (py::isinstance<py::slice>(h)) {
                axes.push_back(slice_from_py(h.cast<py::slice>(), self.shape()[i]));
            } else if (py::isinstance<py::int_>(h)) {
                const size_t ii = norm_index(h.cast<py::ssize_t>(), self.shape()[i]);
                axes.push_back(Slice(ii, ii + 1));
                squeeze_axes.push_back(i);
            } else {
                throw py::type_error("array index must be int, slice, or a tuple of those");
            }
        }
        Arr view = self[axes];
        for (size_t k = squeeze_axes.size(); k-- > 0; ) {
            view = view.squeeze(squeeze_axes[k]);
        }
        return py::cast(std::move(view));
    }

    throw py::type_error("array index must be int, slice, or a tuple of those");
}

template<typename T>
inline void array_setitem(cthreads::linalg::Array<T>& self, py::handle key, py::handle value) {
    using Arr = cthreads::linalg::Array<T>;
    using Shape = cthreads::linalg::Shape;
    if (py::isinstance<cthreads::linalg::Array<uint8_t>>(key)) {
        auto mask = key.cast<cthreads::linalg::Array<uint8_t>>();
        if (py::isinstance<Arr>(value)) {
            self.masked_scatter(mask, value.cast<Arr>());
        } else {
            self.masked_fill(mask, value.cast<T>());
        }
        return;
    }
    if (py::isinstance<py::int_>(key) && self.ndim() == 1) {
        const size_t i = norm_index(key.cast<py::ssize_t>(), self.shape()[0]);
        self[Shape(std::vector<size_t>{i})] = value.cast<T>();
        return;
    }
    if (py::isinstance<py::tuple>(key) || py::isinstance<py::list>(key)) {
        py::sequence seq = key.cast<py::sequence>();
        if (static_cast<size_t>(seq.size()) == self.ndim()) {
            std::vector<size_t> idx(self.ndim());
            for (size_t i = 0; i < self.ndim(); ++i) {
                idx[i] = norm_index(seq[i].cast<py::ssize_t>(), self.shape()[i]);
            }
            self[Shape(idx)] = value.cast<T>();
            return;
        }
    }
    throw py::type_error("item assignment requires a full integer index or a boolean mask");
}

template<typename T>
inline void bind_array(py::module_& m, const char* name) {
    using Arr = cthreads::linalg::Array<T>;
    using Shape = cthreads::linalg::Shape;

    auto cls = py::class_<Arr>(m, name);
    cls.def(py::init([](const std::vector<size_t>& dims) {
            return Arr(Shape(dims));
        }), py::arg("shape"), "Allocate a C-contiguous array of the given shape.")
        .def_static(
            "from_list",
            [](py::object nested) {
                const std::vector<size_t> dims = infer_nested_shape<T>(nested);
                Arr a{Shape(dims)};
                std::vector<size_t> idx(dims.size(), 0);
                fill_from_nested<T>(a, nested, idx, 0);
                return a;
            },
            py::arg("data"),
            "Build from a nested Python list matching the array shape."
        )
        .def("to_list", [](const Arr& self) {
            std::vector<size_t> idx(self.ndim(), 0);
            return to_nested<T>(self, idx, 0);
        })
        .def_property_readonly("shape", [](const Arr& self) {
            return shape_dims(self.shape());
        })
        .def_property_readonly("strides", [](const Arr& self) {
            return shape_dims(self.strides());
        })
        .def_property_readonly("ndim", &Arr::ndim)
        .def_property_readonly("numel", [](const Arr& self) {
            return self.shape().numel();
        })
        .def_property_readonly("offset", &Arr::offset)
        .def("is_contiguous", &Arr::is_contiguous)
        .def("__len__", [](const Arr& self) {
            if (self.ndim() == 0) {
                return size_t{0};
            }
            return self.shape()[0];
        })
        .def("__repr__", [name](const Arr& self) {
            std::ostringstream os;
            os << name << "(shape=[";
            const auto dims = shape_dims(self.shape());
            for (size_t i = 0; i < dims.size(); ++i) {
                if (i) {
                    os << ", ";
                }
                os << dims[i];
            }
            os << "], contiguous=" << (self.is_contiguous() ? "True" : "False") << ")";
            return os.str();
        })
        .def("__bool__", [](const Arr&) -> bool {
            throw std::runtime_error("the truth value of an array is ambiguous; use .any() / .all() / .count()");
        })
        .def("__getitem__", &array_getitem<T>)
        .def("__setitem__", &array_setitem<T>)
        .def("view", [](const Arr& self, const std::vector<size_t>& dims) {
            return self.view(Shape(dims));
        })
        .def("reshape", [](const Arr& self, const std::vector<size_t>& dims) {
            return self.reshape(Shape(dims));
        })
        .def("flatten", &Arr::flatten)
        .def("transpose", &Arr::transpose)
        .def("permute", [](const Arr& self, const std::vector<size_t>& axes) {
            return self.permute(Shape(axes));
        })
        .def("squeeze", &Arr::squeeze, py::arg("axis"))
        .def("unsqueeze", &Arr::unsqueeze, py::arg("axis"))
        .def("contiguous", &Arr::contiguous)
        .def("_contiguous", &Arr::_contiguous)
        .def("_view", [](Arr& self, const std::vector<size_t>& dims) {
            self._view(Shape(dims));
        })
        .def("_reshape", [](Arr& self, const std::vector<size_t>& dims) {
            self._reshape(Shape(dims));
        })
        .def("_flatten", &Arr::_flatten)
        .def("_transpose", &Arr::_transpose)
        .def("_permute", [](Arr& self, const std::vector<size_t>& axes) {
            self._permute(Shape(axes));
        })
        .def("_squeeze", &Arr::_squeeze, py::arg("axis"))
        .def("_unsqueeze", &Arr::_unsqueeze, py::arg("axis"))
        .def("masked_select", &Arr::masked_select, py::arg("mask"))
        .def("masked_fill", &Arr::masked_fill, py::arg("mask"), py::arg("value"))
        .def("masked_scatter", &Arr::masked_scatter, py::arg("mask"), py::arg("values"))
        .def("count", &Arr::count_nonzero)
        .def("any", [](const Arr& self) { return self.count_nonzero() != 0; })
        .def("all", [](const Arr& self) { return self.count_nonzero() == self.shape().numel(); })
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self > py::self)
        .def(py::self < py::self)
        .def(py::self >= py::self)
        .def(py::self <= py::self)
        .def(py::self == T())
        .def(py::self != T())
        .def(py::self > T())
        .def(py::self < T())
        .def(py::self >= T())
        .def(py::self <= T());

    if constexpr (std::is_same_v<T, uint8_t>) {
        cls.def("__and__", &cthreads::linalg::mask_and)
            .def("__or__", &cthreads::linalg::mask_or)
            .def("__xor__", &cthreads::linalg::mask_xor)
            .def("__invert__", &cthreads::linalg::mask_not)
            .def("__rand__", &cthreads::linalg::mask_and)
            .def("__ror__", &cthreads::linalg::mask_or)
            .def("__rxor__", &cthreads::linalg::mask_xor);
    } else {
        cls.def("matmul", &Arr::matmul)
            .def("dot", &Arr::dot)
            .def("cross", &Arr::cross)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self * py::self)
            .def(py::self / py::self)
            .def(py::self + T())
            .def(py::self - T())
            .def(py::self * T())
            .def(py::self / T())
            .def(-py::self)
            .def("_add", py::overload_cast<const Arr&>(&Arr::_add))
            .def("_sub", py::overload_cast<const Arr&>(&Arr::_sub))
            .def("_mul", py::overload_cast<const Arr&>(&Arr::_mul))
            .def("_div", py::overload_cast<const Arr&>(&Arr::_div))
            .def("_add", py::overload_cast<T>(&Arr::_add))
            .def("_sub", py::overload_cast<T>(&Arr::_sub))
            .def("_mul", py::overload_cast<T>(&Arr::_mul))
            .def("_div", py::overload_cast<T>(&Arr::_div))
            .def("_neg", &Arr::_neg)
            .def("_matmul", &Arr::_matmul)
            .def("_dot", &Arr::_dot)
            .def("_cross", &Arr::_cross);
    }
    cls.attr("__cthreads_internal__") = true;
}

inline void bind_linalg(py::module_& parent) {
    py::module_ m = parent.def_submodule("linalg", "ND arrays (C++ linalg::Array)");
    m.attr("__cthreads_internal__") = true;

    auto shape_cls = py::class_<cthreads::linalg::Shape>(m, "Shape");
    shape_cls
        .def(py::init<const std::vector<size_t>&>(), py::arg("dims"))
        .def(py::init<size_t>(), py::arg("dim"))
        .def("__len__", &cthreads::linalg::Shape::size)
        .def("__getitem__", [](const cthreads::linalg::Shape& s, size_t i) {
            if (i >= s.ndim()) {
                throw py::index_error("Shape index out of range");
            }
            return s[i];
        })
        .def("ndim", &cthreads::linalg::Shape::ndim)
        .def("numel", &cthreads::linalg::Shape::numel)
        .def("strides", [](const cthreads::linalg::Shape& s) {
            return shape_dims(s.strides());
        })
        .def("__eq__", &cthreads::linalg::Shape::operator==)
        .def("__repr__", [](const cthreads::linalg::Shape& s) {
            std::ostringstream os;
            os << "Shape([";
            for (size_t i = 0; i < s.ndim(); ++i) {
                if (i) {
                    os << ", ";
                }
                os << s[i];
            }
            os << "])";
            return os.str();
        });
    shape_cls.attr("__cthreads_internal__") = true;

    auto slice_cls = py::class_<cthreads::linalg::Slice>(m, "Slice");
    slice_cls
        .def(py::init<>())
        .def(py::init<size_t>(), py::arg("stop"))
        .def(py::init<size_t, size_t, size_t>(),
             py::arg("start"), py::arg("stop"), py::arg("step") = 1)
        .def_readwrite("start", &cthreads::linalg::Slice::start)
        .def_readwrite("stop", &cthreads::linalg::Slice::stop)
        .def_readwrite("step", &cthreads::linalg::Slice::step)
        .def("__repr__", [](const cthreads::linalg::Slice& s) {
            std::ostringstream os;
            os << "Slice(" << s.start << ", ";
            if (s.stop == cthreads::linalg::Slice::npos) {
                os << "None";
            } else {
                os << s.stop;
            }
            os << ", " << s.step << ")";
            return os.str();
        });
    slice_cls.attr("__cthreads_internal__") = true;

    bind_array<uint8_t>(m, "ArrayBool");
    bind_array<float>(m, "ArrayF32");
    bind_array<double>(m, "ArrayF64");
    bind_array<int>(m, "ArrayI32");
}
