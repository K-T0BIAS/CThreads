// Copyright (c) 2026 Tobias Karusseit
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "../headers/compile_glsl.hpp"

#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <SPIRV/GlslangToSpv.h>

#include <mutex>
#include <stdexcept>
#include <string>

namespace cthreads::gpu {
namespace {

std::once_flag g_glslang_once;

void ensure_glslang_initialized() {
    std::call_once(g_glslang_once, []() {
        if (!glslang::InitializeProcess()) {
            throw std::runtime_error(
                "cthreads.gpu.VulkanInitFailed: glslang InitializeProcess failed");
        }
    });
}

} // namespace

std::vector<std::uint32_t> compile_glsl_to_spirv(const std::string& source) {
    if (source.empty()) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: compile_glsl source is empty");
    }

    ensure_glslang_initialized();

    const EShLanguage stage = EShLangCompute;
    glslang::TShader shader(stage);
    const char* strings[] = {source.c_str()};
    shader.setStrings(strings, 1);

    // Vulkan 1.0 / SPIR-V 1.0 is enough for our compute SSBOs + builtins.
    shader.setEnvInput(
        glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);

    const EShMessages messages =
        static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
    const TBuiltInResource* resources = GetDefaultResources();

    std::string log;
    if (!shader.parse(resources, 100, false, messages)) {
        log = shader.getInfoLog();
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GLSL parse failed:\n" + log);
    }

    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(messages)) {
        log = program.getInfoLog();
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GLSL link failed:\n" + log);
    }

    const glslang::TIntermediate* intermediate = program.getIntermediate(stage);
    if (intermediate == nullptr) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GLSL link produced no intermediate");
    }

    std::vector<std::uint32_t> spirv;
    spv::SpvBuildLogger logger;
    glslang::SpvOptions options;
    options.generateDebugInfo = false;
    options.disableOptimizer = true;
    options.optimizeSize = false;
    glslang::GlslangToSpv(*intermediate, spirv, &logger, &options);

    if (spirv.empty()) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GlslangToSpv produced empty SPIR-V: " +
            logger.getAllMessages());
    }
    return spirv;
}

} // namespace cthreads::gpu
