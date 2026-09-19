# Copyright (c) 2026 Tobias Karusseit
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.18)

# Prefer pyproject metadata when driven by scikit-build-core.
if(DEFINED SKBUILD_PROJECT_NAME)
    project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)
else()
    project(cthreads LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enable locally (pick one shell):
#   cmd:        set CMAKE_ARGS=-DCTHREADS_GPU=ON
#   PowerShell: $env:CMAKE_ARGS="-DCTHREADS_GPU=ON"
#   either:     pip install -e . --config-settings=cmake.define.CTHREADS_GPU=ON
# Wipe build/ if toggling ON/OFF so CMake reconfigures (cached OFF sticks otherwise).
option(CTHREADS_GPU "Build Vulkan GPU support into _ext" OFF)

# --- Python + pybind11 -------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# --- Extension module: import cthreads._ext ----------------------------------
set(_CTHREADS_LINALG_SOURCES
    impl/linalg/array.cpp
    impl/linalg/tiling.cpp
)

pybind11_add_module(_ext MODULE
    bindings/module.cpp
    ${_CTHREADS_LINALG_SOURCES}
)

target_include_directories(_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/headers
)

# Enable AVX2/FMA so Array float/double kernels hit the SIMD path on supported CPUs.
if(MSVC)
    target_compile_options(_ext PRIVATE /arch:AVX2)
else()
    target_compile_options(_ext PRIVATE -mavx2 -mfma)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(_ext PRIVATE ${CMAKE_DL_LIBS})
endif()

# --- Standalone linalg micro-bench (dot + matmul, light/medium/heavy) (contact @K-T0BIAS for src) --------
set(_CTHREADS_BENCH_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../demo/bench/cpp/linalg_bench.cpp"
)
if(EXISTS "${_CTHREADS_BENCH_SRC}")
    add_executable(linalg_bench
        ${_CTHREADS_BENCH_SRC}
        ${_CTHREADS_LINALG_SOURCES}
    )
    target_include_directories(linalg_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/headers
    )
    if(MSVC)
        target_compile_options(linalg_bench PRIVATE /arch:AVX2 /O2)
    else()
        target_compile_options(linalg_bench PRIVATE -mavx2 -mfma -O3)
    endif()
    set_target_properties(linalg_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../../demo/bench/cpp"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../../../demo/bench/cpp"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../../../demo/bench/cpp"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/../../../demo/bench/cpp"
    )
    message(STATUS "linalg_bench: ${_CTHREADS_BENCH_SRC}")
endif()

# Python package dir (sibling of this cpp/ tree).
set(_cthreads_py_out "${CMAKE_CURRENT_SOURCE_DIR}/../python/cthreads")

# Bare cmake (no pip): build straight into the package tree.
if(NOT DEFINED SKBUILD)
    set_target_properties(_ext PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY          "${_cthreads_py_out}"
        RUNTIME_OUTPUT_DIRECTORY          "${_cthreads_py_out}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG    "${_cthreads_py_out}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE  "${_cthreads_py_out}"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_cthreads_py_out}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG    "${_cthreads_py_out}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE  "${_cthreads_py_out}"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_cthreads_py_out}"
    )
    message(STATUS "Ext out: ${_cthreads_py_out}")
endif()

# pip/scikit-build: keep a copy beside Python sources. Editable redirect loads
# the package from this tree; PathFinder then picks up _ext next to __init__.py.
if(DEFINED SKBUILD)
    add_custom_command(
        TARGET _ext POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_cthreads_py_out}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:_ext>"
            "${_cthreads_py_out}/$<TARGET_FILE_NAME:_ext>"
        COMMENT "Copy _ext into Python package directory"
    )
endif()

# Wheels: install into platlib/cthreads.
# Editable: install into NULL so we do NOT register a redirect mapping to a
# missing/ephemeral path (which breaks `import cthreads._ext` on Windows).
if(DEFINED SKBUILD_STATE AND SKBUILD_STATE STREQUAL "editable")
    install(TARGETS _ext
        LIBRARY DESTINATION "${SKBUILD_NULL_DIR}"
        RUNTIME DESTINATION "${SKBUILD_NULL_DIR}"
    )
else()
    install(TARGETS _ext
        LIBRARY DESTINATION cthreads
        RUNTIME DESTINATION cthreads
    )
endif()

# Kernel DLL build needs these next to the installed package (PyPI wheels).
# Editable installs resolve headers from the monorepo cpp/ tree via build.py.
# Layout: cthreads/_native/headers/... and cthreads/_native/runtime/sync_bridge.cpp
# so sync_bridge's `#include "../headers/sync/syncState.hpp"` stays valid.
if(DEFINED SKBUILD AND NOT (DEFINED SKBUILD_STATE AND SKBUILD_STATE STREQUAL "editable"))
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/headers/"
        DESTINATION cthreads/_native/headers
    )
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/runtime/sync_bridge.cpp"
        DESTINATION cthreads/_native/runtime
    )
endif()

message(STATUS "Python:  ${Python_EXECUTABLE} (${Python_VERSION})")
if(DEFINED SKBUILD_STATE)
    message(STATUS "SKBUILD_STATE: ${SKBUILD_STATE}")
endif()

# --- GPU ----

if(CTHREADS_GPU)
    find_package(Vulkan REQUIRED) # vulkan sdk for headers/includes
    message(STATUS "cthreads GPU: ON (Vulkan)")

    # Vendor Khronos glslang (GLSL -> SPIR-V). Same compiler engine shaderc uses.
    # Linked statically into _ext so end users need no glslc / extra SDK tools.
    include(FetchContent)
    set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
    set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
    set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
    set(ENABLE_SPVREMAPPER OFF CACHE BOOL "" FORCE)
    set(ENABLE_CTEST OFF CACHE BOOL "" FORCE)
    set(SKIP_GLSLANG_INSTALL ON CACHE BOOL "" FORCE)
    set(BUILD_EXTERNAL OFF CACHE BOOL "" FORCE)
    set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
    set(ENABLE_GLSLANG_JS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        glslang
        GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
        GIT_TAG        15.1.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(glslang)
    message(STATUS "cthreads GPU: vendored glslang for compile_glsl")

    target_sources(_ext PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/gpu/impl/context.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gpu/impl/memory.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gpu/impl/pack.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gpu/impl/shader_cache.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gpu/impl/state.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gpu/impl/shader.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gpu/impl/descriptors.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gpu/impl/module.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gpu/impl/compile_glsl.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/bindings/gpu_module.cpp"
    )

    # Local-only substrate smokes under gpu/testing/ (gitignored via testing/).
    # Product GPU builds (CI / wheels) do not need them. Python coverage of the
    # public path lives in repo tests/;
    set(_cthreads_gpu_testing_pack
        "${CMAKE_CURRENT_SOURCE_DIR}/gpu/testing/pack_roundtrip.cpp")
    set(_cthreads_gpu_testing_smoke
        "${CMAKE_CURRENT_SOURCE_DIR}/gpu/testing/shader_smoke.cpp")
    if(EXISTS "${_cthreads_gpu_testing_pack}" AND EXISTS "${_cthreads_gpu_testing_smoke}")
        message(STATUS "cthreads GPU: local gpu/testing helpers ON")
        target_sources(_ext PRIVATE
            "${_cthreads_gpu_testing_pack}"
            "${_cthreads_gpu_testing_smoke}"
            "${CMAKE_CURRENT_SOURCE_DIR}/bindings/gpu_testing_module.cpp"
        )
        target_compile_definitions(_ext PRIVATE CTHREADS_GPU_TESTING=1)
    else()
        message(STATUS "cthreads GPU: local gpu/testing helpers OFF (not in tree)")
    endif()

    target_include_directories(_ext PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/headers
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu
        ${Vulkan_INCLUDE_DIRS}
    )
    target_compile_definitions(_ext PRIVATE CTHREADS_WITH_GPU=1)
    target_link_libraries(_ext PRIVATE
        glslang
        SPIRV
        glslang-default-resource-limits
    )

    # Ship upstream license texts next to the Python package (Apache/BSD notices).
    set(_cthreads_gpu_notices_out
        "${_cthreads_py_out}/gpu/third_party_notices")
    set(_cthreads_gpu_notices_src
        "${CMAKE_CURRENT_SOURCE_DIR}/gpu/third_party_notices")
    file(MAKE_DIRECTORY "${_cthreads_gpu_notices_out}")
    if(EXISTS "${_cthreads_gpu_notices_src}/README.md")
        configure_file(
            "${_cthreads_gpu_notices_src}/README.md"
            "${_cthreads_gpu_notices_out}/README.md"
            COPYONLY
        )
    endif()
    if(DEFINED glslang_SOURCE_DIR)
        foreach(_lic IN ITEMS LICENSE.txt LICENSE.TXT LICENSE)
            if(EXISTS "${glslang_SOURCE_DIR}/${_lic}")
                configure_file(
                    "${glslang_SOURCE_DIR}/${_lic}"
                    "${_cthreads_gpu_notices_out}/glslang-${_lic}"
                    COPYONLY
                )
                break()
            endif()
        endforeach()
    endif()

    if(DEFINED SKBUILD AND NOT (DEFINED SKBUILD_STATE AND SKBUILD_STATE STREQUAL "editable"))
        install(DIRECTORY "${_cthreads_gpu_notices_out}/"
            DESTINATION cthreads/gpu/third_party_notices
            OPTIONAL
        )
    endif()
endif()
