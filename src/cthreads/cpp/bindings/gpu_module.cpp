// Copyright (c) 2026 Tobias Karusseit
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "gpu_module.hpp"

#include "../gpu/headers/context.hpp"

#include <pybind11/pybind11.h>

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
}
