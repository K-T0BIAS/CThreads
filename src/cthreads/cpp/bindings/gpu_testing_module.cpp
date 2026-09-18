// Copyright (c) 2026 Tobias Karusseit
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "gpu_testing_module.hpp"

#include "../gpu/testing/pack_roundtrip.hpp"
#include "../gpu/testing/shader_smoke.hpp"

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

    t.def(
        "smoke_create_entry",
        &cthreads::gpu::testing::smoke_create_entry,
        "create_entry + destroy for committed smoke SPIR-V. Test-only."
    );
    t.def(
        "smoke_update_descriptors",
        &cthreads::gpu::testing::smoke_update_descriptors,
        "create_entry + pack + update_descriptors smoke. Test-only."
    );
    t.def(
        "smoke_cache_register_and_get",
        &cthreads::gpu::testing::smoke_cache_register_and_get,
        "ShaderCache add/get/clear smoke. Test-only."
    );
    t.def(
        "probe_cache_duplicate_add",
        &cthreads::gpu::testing::probe_cache_duplicate_add,
        "Raises on duplicate ShaderCache add. Test-only."
    );
    t.def(
        "probe_update_empty_list_slot",
        &cthreads::gpu::testing::probe_update_empty_list_slot,
        "Raises GpuInvalidArgument for empty list descriptor. Test-only."
    );
    t.def(
        "probe_create_entry_zero_bindings",
        &cthreads::gpu::testing::probe_create_entry_zero_bindings,
        "Raises GpuInvalidArgument for binding_count 0. Test-only."
    );
    t.def(
        "clear_shader_cache",
        &cthreads::gpu::testing::clear_shader_cache,
        "Clear process ShaderCache. Test-only teardown."
    );
    t.def(
        "register_smoke_saxpy",
        &cthreads::gpu::testing::register_smoke_saxpy,
        "Register smoke saxpy SPIR-V in ShaderCache; returns symbol key. "
        "Does not launch - use _ext.gpu.launch_gpu_kernel. Test-only."
    );
}
