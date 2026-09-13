// Copyright (c) 2026 Tobias Karusseit
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "gpu_module.hpp"
#include "gpu_testing_module.hpp"

#include "../gpu/headers/context.hpp"
#include "../gpu/headers/compile_glsl.hpp"
#include "../gpu/headers/module.hpp"
#include "../gpu/headers/shader_cache.hpp"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
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
            [](cthreads::gpu::SpawnedGpuKernel& self) {
                self.join(cthreads::gpu::context());
            },
            "Wait for the GPU fence, download ref lists, release inflight state."
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

    // Test-only pack round-trips live in a separate submodule / translation unit
    // so product bindings stay small. Not re-exported by cthreads.gpu.
    bind_gpu_testing(g);
}
