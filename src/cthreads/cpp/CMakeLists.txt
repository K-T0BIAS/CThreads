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
pybind11_add_module(_ext MODULE
    bindings/module.cpp
)

target_include_directories(_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/headers
)

if(UNIX AND NOT APPLE)
    target_link_libraries(_ext PRIVATE ${CMAKE_DL_LIBS})
endif()

# Python package dir (sibling of this cpp/ tree). Never use cpp/ as output/build.
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

message(STATUS "Python:  ${Python_EXECUTABLE} (${Python_VERSION})")
if(DEFINED SKBUILD_STATE)
    message(STATUS "SKBUILD_STATE: ${SKBUILD_STATE}")
endif()
