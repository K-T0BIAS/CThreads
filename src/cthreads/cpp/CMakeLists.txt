cmake_minimum_required(VERSION 3.18)

project(cthreads LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

# --- Extension module: import cthreads / cthreads.sync -----------------------
# PYBIND11_MODULE(cthreads, ...) in bindings/module.cpp
pybind11_add_module(cthreads MODULE
    bindings/module.cpp
)

target_include_directories(cthreads PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/headers
)

# Dev-friendly output: drop the .pyd/.so next to the Python package tree so
# `PYTHONPATH=src/cthreads/python` (or adding that dir) can import `cthreads`.
set(_cthreads_py_out "${CMAKE_CURRENT_SOURCE_DIR}/../python")
set_target_properties(cthreads PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY          "${_cthreads_py_out}"
    RUNTIME_OUTPUT_DIRECTORY          "${_cthreads_py_out}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG    "${_cthreads_py_out}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE  "${_cthreads_py_out}"
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_cthreads_py_out}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG    "${_cthreads_py_out}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE  "${_cthreads_py_out}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_cthreads_py_out}"
)

install(TARGETS cthreads
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)

message(STATUS "Python:  ${Python_EXECUTABLE} (${Python_VERSION})")
message(STATUS "Output:  ${_cthreads_py_out}")
