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

set(example_name hiptensor_contraction_bilinear_bf16_f32)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../Common/HipPlatform.cmake")
select_gpu_language()

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    message(FATAL_ERROR "hipTensor examples do not support the CUDA runtime.")
endif()

enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})

# see https://rocm.docs.amd.com/projects/hipTensor/en/latest/install/installation.html#gpu-support
set(HIPTENSOR_SUPPORTED_ARCH gfx908 gfx90a gfx942 gfx950)

foreach(ARCH ${CMAKE_HIP_ARCHITECTURES})
    if(NOT ARCH IN_LIST HIPTENSOR_SUPPORTED_ARCH)
        message(FATAL_ERROR "hipTensor does not support architecture: ${ARCH}, not building hipTensor examples")
    endif()
endforeach()

set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD 17)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_EXTENSIONS OFF)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD_REQUIRED ON)
select_hip_platform()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "hipTensor examples are only available on Linux")
    return()
else()
    set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

find_package(hiptensor REQUIRED)

add_executable(${example_name} main.cpp)
# Make example runnable using ctest
add_test(NAME ${example_name} COMMAND ${example_name})

# Link to example library
target_link_libraries(${example_name} PRIVATE hiptensor)

target_include_directories(
    ${example_name}
    PRIVATE "../../../../../Common" "../../../../../External"
)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${GPU_RUNTIME})

install(TARGETS ${example_name})
