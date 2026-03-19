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

set(example_name ComposableKernel_ck_tile_smoothquant)
set(example_name_basic ${example_name}_basic)
set(example_script test_ck_tile_smoothquant.sh)

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

include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/FilterHIPArchitectures.cmake")
filter_hip_architectures("CK Tile SmoothQuant example" "gfx908;gfx90a;gfx942;gfx950" SHOULD_SKIP)
if(SHOULD_SKIP)
    return()
endif()

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    message(STATUS "Composable Kernel examples are not supported for CUDA.")
    return()
endif()

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

find_package(composable_kernel REQUIRED)
include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/FindCKTile.cmake")

find_package(hip REQUIRED)

########################################################################################################################
# Main example with instances
########################################################################################################################
add_executable(${example_name}
    instances/smoothquant_bf16_n64_n128_instance.cpp
    instances/smoothquant_bf16_n256_instance.cpp
    instances/smoothquant_bf16_n512_instance.cpp
    instances/smoothquant_bf16_n768_instance.cpp
    instances/smoothquant_bf16_n1024_instance.cpp
    instances/smoothquant_bf16_n1536_instance.cpp
    instances/smoothquant_bf16_n2048_instance.cpp
    instances/smoothquant_bf16_n3072_instance.cpp
    instances/smoothquant_bf16_n4096_instance.cpp
    instances/smoothquant_bf16_n4096_tp_instance.cpp
    instances/smoothquant_fp16_n64_n128_instance.cpp
    instances/smoothquant_fp16_n256_instance.cpp
    instances/smoothquant_fp16_n512_instance.cpp
    instances/smoothquant_fp16_n768_instance.cpp
    instances/smoothquant_fp16_n1024_instance.cpp
    instances/smoothquant_fp16_n1536_instance.cpp
    instances/smoothquant_fp16_n2048_instance.cpp
    instances/smoothquant_fp16_n3072_instance.cpp
    instances/smoothquant_fp16_n4096_instance.cpp
    instances/smoothquant_fp16_n4096_tp_instance.cpp
    instances/smoothquant_fwd_api.cpp
    smoothquant.cpp
)
target_include_directories(${example_name} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
add_test(${example_name} COMMAND "${CMAKE_CURRENT_LIST_DIR}/${example_script}" "$<TARGET_FILE:${example_name}>")
set_source_files_properties(
    instances/smoothquant_bf16_n64_n128_instance.cpp
    instances/smoothquant_bf16_n256_instance.cpp
    instances/smoothquant_bf16_n512_instance.cpp
    instances/smoothquant_bf16_n768_instance.cpp
    instances/smoothquant_bf16_n1024_instance.cpp
    instances/smoothquant_bf16_n1536_instance.cpp
    instances/smoothquant_bf16_n2048_instance.cpp
    instances/smoothquant_bf16_n3072_instance.cpp
    instances/smoothquant_bf16_n4096_instance.cpp
    instances/smoothquant_bf16_n4096_tp_instance.cpp
    instances/smoothquant_fp16_n64_n128_instance.cpp
    instances/smoothquant_fp16_n256_instance.cpp
    instances/smoothquant_fp16_n512_instance.cpp
    instances/smoothquant_fp16_n768_instance.cpp
    instances/smoothquant_fp16_n1024_instance.cpp
    instances/smoothquant_fp16_n1536_instance.cpp
    instances/smoothquant_fp16_n2048_instance.cpp
    instances/smoothquant_fp16_n3072_instance.cpp
    instances/smoothquant_fp16_n4096_instance.cpp
    instances/smoothquant_fp16_n4096_tp_instance.cpp
    instances/smoothquant_fwd_api.cpp
    smoothquant.cpp
    PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE}
)
target_compile_features(${example_name} PRIVATE cxx_std_17 hip_std_17)
target_compile_options(${example_name} PRIVATE -Wno-undefined-func-template -Wno-float-equal)
target_link_libraries(${example_name} PRIVATE composable_kernel::ck_tile hip::host)

########################################################################################################################
# Basic example
########################################################################################################################
add_executable(${example_name_basic} example_smoothquant.cpp)
add_test(${example_name_basic} COMMAND "${CMAKE_CURRENT_LIST_DIR}/${example_script} $<TARGET_FILE:${example_name_basic}>")
set_source_files_properties(example_smoothquant.cpp PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})
target_compile_features(${example_name_basic} PRIVATE cxx_std_17 hip_std_17)
target_compile_options(${example_name_basic} PRIVATE -Wno-undefined-func-template -Wno-float-equal)
target_link_libraries(${example_name_basic} PRIVATE composable_kernel::ck_tile hip::host)
