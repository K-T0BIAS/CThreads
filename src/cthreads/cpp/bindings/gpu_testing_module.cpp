// Copyright (c) 2026 Tobias Karusseit
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "gpu_testing_module.hpp"

#include "../gpu/testing/pack_roundtrip.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

py::bytes vector_to_bytes(const std::vector<std::uint8_t>& data) {
    return py::bytes(reinterpret_cast<const char*>(data.data()), static_cast<py::ssize_t>(data.size()));
}

} // namespace

void bind_gpu_testing(py::module_& gpu_parent) {
    py::module_ t = gpu_parent.def_submodule(
        "testing",
        "TEST-ONLY GpuPack helpers. Not a product API; do not use from library code."
    );

    t.def(
        "roundtrip_float_pack",
        [](const py::bytes& scalar, const std::vector<std::vector<float>>& lists) {
            const std::string raw = scalar;
            auto result = cthreads::gpu::testing::roundtrip_float_pack(
                raw.empty() ? nullptr : raw.data(),
                raw.size(),
                lists
            );
            return py::make_tuple(vector_to_bytes(result.scalars), py::cast(result.float_lists));
        },
        py::arg("scalar"),
        py::arg("float_lists"),
        "Upload/download scalar bytes + list[list[float]] through GpuPack (test-only)."
    );

    t.def(
        "roundtrip_int_pack",
        [](const py::bytes& scalar, const std::vector<std::vector<std::int32_t>>& lists) {
            const std::string raw = scalar;
            auto result = cthreads::gpu::testing::roundtrip_int_pack(
                raw.empty() ? nullptr : raw.data(),
                raw.size(),
                lists
            );
            return py::make_tuple(vector_to_bytes(result.scalars), py::cast(result.int_lists));
        },
        py::arg("scalar"),
        py::arg("int_lists"),
        "Upload/download scalar bytes + list[list[int]] through GpuPack (test-only)."
    );

    t.def(
        "probe_invalid_elem_bytes",
        &cthreads::gpu::testing::probe_invalid_elem_bytes,
        "Raises GpuInvalidArgument (zero elem_bytes). Test-only."
    );

    t.def(
        "probe_use_after_destroy_scalars",
        &cthreads::gpu::testing::probe_use_after_destroy_scalars,
        "Raises GpuUseAfterDestroy (upload into pack with no scalar buffer). Test-only."
    );
}
