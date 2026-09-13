# Third-party notices for native GLSL -> SPIR-V

When `CTHREADS_GPU=ON`, cthreads links **Khronos glslang** into `_ext` so
`compile_glsl` works without installing `glslc` or other Vulkan SDK tools.

glslang is the same compiler engine Google **shaderc** wraps. License texts
copied here at build time (see also the root `LICENSE` third-party section):

- `glslang-LICENSE*` — Khronos glslang (Apache-2.0 / BSD-style components)

End-user wheels that include the GPU extension must redistribute these notices
alongside the binary.
