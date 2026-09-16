// Copyright (c) 2026 Tobias Karusseit
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "gpu_module.hpp"
#include "gpu_testing_module.hpp"

#include "../gpu/headers/context.hpp"
#include "../gpu/headers/compile_glsl.hpp"
#include "../gpu/headers/memory.hpp"
#include "../gpu/headers/module.hpp"
#include "../gpu/headers/shader_cache.hpp"
#include "../gpu/headers/state.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

void bind_gpu(py::module_& parent) {
    py::module_ g = parent.def_submodule(
        "gpu",
        "Vulkan GPU runtime (loader dynamically loaded at init)"
    );

    g.def(
        "available",
        &cthreads::gpu::available,
        "True if Vulkan loader + compute device initialized successfully."
    );

    g.def(
        "device_name",
        &cthreads::gpu::device_name,
        "GPU deviceName from Vulkan; calls init() (may raise)."
    );

    g.def(
        "init",
        &cthreads::gpu::init,
        "Explicitly initialize Vulkan context (optional; available/device_name also init)."
    );

    g.def(
        "shutdown",
        &cthreads::gpu::shutdown,
        "Destroy device/instance and unload the Vulkan loader."
    );

    // Product launch handle (mirror of CPU SpawnedKernel / Job surface).
    // join() always uses the process Context so Python never holds a Context&.
    py::class_<
        cthreads::gpu::SpawnedGpuKernel,
        std::shared_ptr<cthreads::gpu::SpawnedGpuKernel>>(g, "GpuJob")
        .def(
            "start",
            &cthreads::gpu::SpawnedGpuKernel::start,
            "No-op: work is submitted at launch_gpu_kernel time."
        )
        .def(
            "join",
            [](cthreads::gpu::SpawnedGpuKernel& self, bool download) {
                self.join(cthreads::gpu::context(), download);
            },
            py::arg("download") = true,
            "Wait for the GPU fence; if download=True, write ref lists back, then "
            "release inflight state."
        )
        .def(
            "done",
            &cthreads::gpu::SpawnedGpuKernel::done,
            "True after join (or failure) has marked the job complete."
        )
        .def(
            "wait",
            &cthreads::gpu::SpawnedGpuKernel::wait,
            py::call_guard<py::gil_scoped_release>(),
            "Block until done_flag; does not download. Prefer join()."
        );

    g.def(
        "compile_glsl",
        [](const std::string& source) {
            // glslang can take noticeable time; release the GIL while compiling.
            std::vector<std::uint32_t> words;
            {
                py::gil_scoped_release release;
                words = cthreads::gpu::compile_glsl_to_spirv(source);
            }
            const char* raw = reinterpret_cast<const char*>(words.data());
            const py::ssize_t nbytes =
                static_cast<py::ssize_t>(words.size() * sizeof(std::uint32_t));
            return py::bytes(raw, nbytes);
        },
        py::arg("source"),
        "Compile a GLSL compute shader string to SPIR-V bytes (vendored glslang). "
        "No external glslc / Vulkan SDK tools required."
    );

    g.def(
        "register_shader",
        [](const std::string& symbol, const py::bytes& spirv, uint32_t binding_count) {
            cthreads::gpu::init();
            const std::string raw = spirv;
            if (raw.empty() || (raw.size() % 4) != 0) {
                throw std::runtime_error(
                    "cthreads.gpu.GpuInvalidArgument: register_shader spirv must "
                    "be a non-empty multiple of 4 bytes");
            }
            if (binding_count == 0) {
                throw std::runtime_error(
                    "cthreads.gpu.GpuInvalidArgument: register_shader "
                    "binding_count must be >= 1");
            }
            // SPIR-V is a stream of little-endian uint32 words.
            std::vector<std::uint32_t> words(raw.size() / 4);
            std::memcpy(words.data(), raw.data(), raw.size());
            (void)cthreads::gpu::shader::ShaderRegistry::register_spirv(
                cthreads::gpu::context(),
                symbol,
                words.data(),
                words.size(),
                binding_count);
        },
        py::arg("symbol"),
        py::arg("spirv"),
        py::arg("binding_count"),
        "Create a compute pipeline from SPIR-V and insert it into ShaderCache. "
        "Sole product writer path (ShaderRegistry). Duplicate symbol throws."
    );

    g.def(
        "launch_gpu_kernel",
        &cthreads::gpu::launch_gpu_kernel,
        py::arg("meta"),
        py::arg("ordered_values"),
        "Marshal args from meta + ordered_values, submit compute, return GpuJob. "
        "Does not wait; call job.join() for fence wait and list writeback. "
        "Requires the kernel symbol to already be in ShaderCache."
    );

    // Named device-local buffer registry (singleton). Not a public DeviceBuffer:
    // Python only sees names + sizes; VkBuffer stays inside the registry.
    py::class_<
        cthreads::gpu::memory::GpuState,
        std::unique_ptr<
            cthreads::gpu::memory::GpuState,
            py::nodelete>>(g, "GpuState")
        .def_static(
            "instance",
            []() -> cthreads::gpu::memory::GpuState& {
                return cthreads::gpu::memory::GpuState::getInstance();
            },
            py::return_value_policy::reference,
            "Process-wide GpuState singleton."
        )
        .def(
            "contains",
            &cthreads::gpu::memory::GpuState::contains,
            py::arg("name"),
            "True if name is already registered."
        )
        .def(
            "size",
            &cthreads::gpu::memory::GpuState::size,
            "Number of registered buffers."
        )
        .def(
            "names",
            &cthreads::gpu::memory::GpuState::names,
            "Snapshot of registered names (order is not meaningful)."
        )
        .def(
            "add",
            [](cthreads::gpu::memory::GpuState& self,
               const std::string& name,
               std::uint64_t nbytes) {
                cthreads::gpu::init();
                auto buffer = cthreads::gpu::memory::create_buffer(
                    cthreads::gpu::context(),
                    static_cast<VkDeviceSize>(nbytes),
                    cthreads::gpu::memory::BufferKind::DeviceLocal
                );
                self.add(name, std::move(buffer));
            },
            py::arg("name"),
            py::arg("nbytes"),
            "Allocate a device-local buffer of nbytes and register it under name. "
            "Duplicate or empty names raise."
        )
        .def(
            "remove",
            [](cthreads::gpu::memory::GpuState& self, const std::string& name) {
                self.remove(cthreads::gpu::context(), name);
            },
            py::arg("name"),
            "Destroy and unregister name. Raises if unknown or in_use."
        )
        .def(
            "buffer_size",
            [](cthreads::gpu::memory::GpuState& self, const std::string& name) {
                return static_cast<std::uint64_t>(self.get(name).size);
            },
            py::arg("name"),
            "Byte size of the buffer registered under name."
        )
        .def(
            "is_in_use",
            &cthreads::gpu::memory::GpuState::is_in_use,
            py::arg("name"),
            "True if name is checked out for a launch."
        )
        .def(
            "mark_in_use",
            &cthreads::gpu::memory::GpuState::mark_in_use,
            py::arg("name"),
            "Check out name for a launch. Raises if unknown or already in_use."
        )
        .def(
            "release_in_use",
            &cthreads::gpu::memory::GpuState::release_in_use,
            py::arg("name"),
            "Clear in_use after a launch finishes. Raises if unknown or not in_use."
        )
        .def(
            "upload",
            [](cthreads::gpu::memory::GpuState& self,
               const std::string& name,
               const py::bytes& data) {
                cthreads::gpu::init();
                const std::string raw = data;
                self.upload(
                    cthreads::gpu::context(),
                    name,
                    raw.empty() ? nullptr : raw.data(),
                    static_cast<VkDeviceSize>(raw.size())
                );
            },
            py::arg("name"),
            py::arg("data"),
            "Upload host bytes into a registered device-local buffer (H2D)."
        )
        .def(
            "download",
            [](cthreads::gpu::memory::GpuState& self, const std::string& name) {
                cthreads::gpu::init();
                const std::uint64_t nbytes = static_cast<std::uint64_t>(
                    self.get(name).size);
                std::string raw(static_cast<size_t>(nbytes), '\0');
                self.download(
                    cthreads::gpu::context(),
                    name,
                    raw.empty() ? nullptr : raw.data(),
                    static_cast<VkDeviceSize>(nbytes)
                );
                return py::bytes(raw);
            },
            py::arg("name"),
            "Download a registered device-local buffer into bytes (D2H)."
        );

    // Test-only pack round-trips live in a separate submodule / translation unit
    // so product bindings stay small. Not re-exported by cthreads.gpu.
    bind_gpu_testing(g);
}
