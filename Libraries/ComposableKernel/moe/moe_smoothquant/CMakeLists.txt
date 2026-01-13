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

set(example_name ComposableKernel_ck_tile_moe_smoothquant)
set(example_script test_ck_tile_moe_smoothquant.sh)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Composable Kernel examples are supported only on Linux.")
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/HipPlatform.cmake")
select_gpu_language()
enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()

set(EXAMPLE_SUPPORTED_ARCH gfx908 gfx90a gfx942 gfx950)

# Filter CMAKE_HIP_ARCHITECTURES to only include supported architectures
set(FILTERED_HIP_ARCHITECTURES)
set(REMOVED_ARCHITECTURES)

foreach(ARCH ${CMAKE_HIP_ARCHITECTURES})
    if(ARCH IN_LIST EXAMPLE_SUPPORTED_ARCH)
        list(APPEND FILTERED_HIP_ARCHITECTURES ${ARCH})
    else()
        list(APPEND REMOVED_ARCHITECTURES ${ARCH})
    endif()
endforeach()

if(REMOVED_ARCHITECTURES)
    message(WARNING "CK Tile MoE-SmoothQuant example: Removing unsupported architectures: ${REMOVED_ARCHITECTURES}")
endif()

if(NOT FILTERED_HIP_ARCHITECTURES)
    message(STATUS "CK Tile MoE-SmoothQuant example: No supported architectures found. Not building CK Tile MoE-SmoothQuant example")
    return()
endif()

# Override CMAKE_HIP_ARCHITECTURES for this directory scope
set(CMAKE_HIP_ARCHITECTURES ${FILTERED_HIP_ARCHITECTURES})

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    message(STATUS "Composable Kernel examples are not supported for CUDA.")
    return()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(ROCM_ROOT
        "$ENV{HIP_PATH}"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
else()
    set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

find_package(composable_kernel REQUIRED)
include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/FindCKTile.cmake")

find_package(hip REQUIRED)

add_executable(${example_name}
    moe_smoothquant_bf16_n64_n128_instance.cpp
    moe_smoothquant_bf16_n256_instance.cpp
    moe_smoothquant_bf16_n512_instance.cpp
    moe_smoothquant_bf16_n768_instance.cpp
    moe_smoothquant_bf16_n1024_instance.cpp
    moe_smoothquant_bf16_n1536_instance.cpp
    moe_smoothquant_bf16_n2048_instance.cpp
    moe_smoothquant_bf16_n3072_instance.cpp
    moe_smoothquant_bf16_n4096_instance.cpp
    moe_smoothquant_bf16_n4096_tp_instance.cpp
    moe_smoothquant_fp16_n64_n128_instance.cpp
    moe_smoothquant_fp16_n256_instance.cpp
    moe_smoothquant_fp16_n512_instance.cpp
    moe_smoothquant_fp16_n768_instance.cpp
    moe_smoothquant_fp16_n1024_instance.cpp
    moe_smoothquant_fp16_n1536_instance.cpp
    moe_smoothquant_fp16_n2048_instance.cpp
    moe_smoothquant_fp16_n3072_instance.cpp
    moe_smoothquant_fp16_n4096_instance.cpp
    moe_smoothquant_fp16_n4096_tp_instance.cpp
    moe_smoothquant_fwd_api.cpp
    moe_smoothquant.cpp
)
add_test(${example_name} COMMAND "${CMAKE_CURRENT_LIST_DIR}/${example_script} $<TARGET_FILE:${example_name}>")
set_source_files_properties(
    moe_smoothquant_bf16_n64_n128_instance.cpp
    moe_smoothquant_bf16_n256_instance.cpp
    moe_smoothquant_bf16_n512_instance.cpp
    moe_smoothquant_bf16_n768_instance.cpp
    moe_smoothquant_bf16_n1024_instance.cpp
    moe_smoothquant_bf16_n1536_instance.cpp
    moe_smoothquant_bf16_n2048_instance.cpp
    moe_smoothquant_bf16_n3072_instance.cpp
    moe_smoothquant_bf16_n4096_instance.cpp
    moe_smoothquant_bf16_n4096_tp_instance.cpp
    moe_smoothquant_fp16_n64_n128_instance.cpp
    moe_smoothquant_fp16_n256_instance.cpp
    moe_smoothquant_fp16_n512_instance.cpp
    moe_smoothquant_fp16_n768_instance.cpp
    moe_smoothquant_fp16_n1024_instance.cpp
    moe_smoothquant_fp16_n1536_instance.cpp
    moe_smoothquant_fp16_n2048_instance.cpp
    moe_smoothquant_fp16_n3072_instance.cpp
    moe_smoothquant_fp16_n4096_instance.cpp
    moe_smoothquant_fp16_n4096_tp_instance.cpp
    moe_smoothquant_fwd_api.cpp
    moe_smoothquant.cpp
    PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE}
)
target_compile_features(${example_name} PRIVATE cxx_std_17 hip_std_17)
target_compile_options(${example_name} PRIVATE -Wno-undefined-func-template -Wno-float-equal)
target_link_libraries(${example_name} PRIVATE composable_kernel::ck_tile hip::host)
