#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cthreads::gpu {

/**
 * Compile a GLSL compute shader string to SPIR-V words.
 *
 * Uses the vendored Khronos glslang library (same compiler engine shaderc
 * wraps). No external glslc / Vulkan SDK tools required at runtime.
 *
 * #### Args:
 * - source: std::string = full compute GLSL (`#version` + buffers + main)
 *
 * #### Returns
 * - std::vector<uint32_t> = SPIR-V words
 *
 * #### Raises
 * - std::runtime_error = parse/link failure (message includes glslang log)
 */
std::vector<std::uint32_t> compile_glsl_to_spirv(const std::string& source);

} // namespace cthreads::gpu
