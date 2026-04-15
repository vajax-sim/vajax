cmake_minimum_required(VERSION 3.18...3.29)
project(dense_cuda_jax_cpp LANGUAGES CXX CUDA)

# C++17 required for XLA FFI headers
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find Python and nanobind
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

# Get jaxlib include path for XLA FFI headers
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import jaxlib; import os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE JAXLIB_INCLUDE_DIR
)
message(STATUS "jaxlib include dir: ${JAXLIB_INCLUDE_DIR}")

if(NOT EXISTS "${JAXLIB_INCLUDE_DIR}/xla/ffi/api/ffi.h")
    message(FATAL_ERROR "XLA FFI headers not found at ${JAXLIB_INCLUDE_DIR}")
endif()

# Target GPU architectures:
#   T4 (sm_75), A100 (sm_80), A10/3090 (sm_86), L4/4090 (sm_89), 5060/5090 (sm_120)
# Falls back to PTX JIT for unlisted architectures via the highest arch.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Check if nvcc supports sm_120 (requires CUDA 12.8+)
    execute_process(
        COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-arch
        OUTPUT_VARIABLE _nvcc_archs
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_nvcc_archs MATCHES "compute_120")
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;120")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
    endif()
endif()

# Build the Python extension module
# nanobind_add_module doesn't support CUDA directly, so we build the .cu as
# a static library and link it into the nanobind module.
add_library(dense_cuda_kernels STATIC dense_cuda_ffi.cu)
target_include_directories(dense_cuda_kernels PRIVATE ${JAXLIB_INCLUDE_DIR})
set_target_properties(dense_cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# The nanobind Python module (C++ wrapper with FFI handler + capsule export)
# Needs CUDA include dirs for cudaStream_t used in FFI PlatformStream binding.
nanobind_add_module(dense_cuda_jax_cpp STABLE_ABI
    dense_cuda_module.cpp
)

target_include_directories(dense_cuda_jax_cpp PRIVATE
    ${JAXLIB_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(dense_cuda_jax_cpp PRIVATE dense_cuda_kernels CUDA::cudart)

# Install target
install(TARGETS dense_cuda_jax_cpp LIBRARY DESTINATION .)
