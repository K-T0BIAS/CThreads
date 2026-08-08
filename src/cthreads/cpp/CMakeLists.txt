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

# --- Extension module: import cthreads._ext / cthreads.sync ------------------
# PYBIND11_MODULE(_ext, ...) in bindings/module.cpp
pybind11_add_module(_ext MODULE
    bindings/module.cpp
)

target_include_directories(_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/headers
)

# POSIX dynamic loader (dlopen/dlsym); Windows uses LoadLibrary from the SDK.
if(UNIX AND NOT APPLE)
    target_link_libraries(_ext PRIVATE ${CMAKE_DL_LIBS})
endif()

# Drop the .pyd/.so inside the cthreads package so `import cthreads` works
# with PYTHONPATH=src/cthreads/python.
set(_cthreads_py_out "${CMAKE_CURRENT_SOURCE_DIR}/../python/cthreads")
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

install(TARGETS _ext
    LIBRARY DESTINATION cthreads
    RUNTIME DESTINATION cthreads
)

message(STATUS "Python:  ${Python_EXECUTABLE} (${Python_VERSION})")
message(STATUS "Output:  ${_cthreads_py_out}")
