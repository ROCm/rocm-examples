# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set(example_name hipsparselt_spmm)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "The hipSPARSELt examples are available on Linux only.")
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../Common/HipPlatform.cmake")
select_gpu_language()
enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()

# see https://github.com/ROCm/rocm-libraries/blob/40bbc75960836d1b9e0573606df9dd79800e412a/projects/hipsparselt/cmake/hipsparselt_supported_architectures.cmake#L10-L21
set(HIPSPARSELT_SUPPORTED_ARCH gfx942 gfx950)

foreach(ARCH ${CMAKE_HIP_ARCHITECTURES})
    if(NOT ARCH IN_LIST HIPSPARSELT_SUPPORTED_ARCH)
        message(FATAL_ERROR "hipSPARSELt does not support architecture: ${ARCH}, not building hipSPARSELt examples")
    endif()
endforeach()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(ROCM_PATH
        "$ENV{HIP_PATH}"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
else()
    set(ROCM_PATH
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

find_package(hip REQUIRED)
find_package(hipsparselt REQUIRED)

add_executable(${example_name} main.cpp)
# Make example runnable using CTest
add_test(NAME ${example_name} COMMAND ${example_name})

set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

set_target_properties(${example_name} PROPERTIES $<$<COMPILE_LANGUAGE:CUDA>:CUDA_EXTENSIONS OFF>
                                                 $<$<COMPILE_LANGUAGE:HIP>:HIP_EXTENSIONS OFF>)
target_compile_features(${example_name} PRIVATE cuda_std_17 hip_std_17)
target_include_directories(${example_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../../Common"
                                                   $<$<COMPILE_LANGUAGE:CUDA>:${ROCM_PATH}/include>)

# Link to example library
target_link_libraries(${example_name} PRIVATE hip::host roc::hipsparselt)

install(TARGETS ${example_name})