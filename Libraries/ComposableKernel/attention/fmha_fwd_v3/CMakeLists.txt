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

set(example_name ComposableKernel_ck_tile_fmha_fwd_v3)
set(example_script test_ck_tile_fmha_fwd_v3.sh)

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
filter_hip_architectures("CK Tile FMHA v3 examples" "gfx950" SHOULD_SKIP)
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

add_executable(${example_name}
    example_fmha_fwd_v3.cpp
    fmha_fwd_v3.cpp
    instances/fmha_fwd_v3_d128_bf16_mask.cpp
    instances/fmha_fwd_v3_d128_bf16_nmask.cpp
    instances/fmha_fwd_v3_d128_fp16_mask.cpp
    instances/fmha_fwd_v3_d128_fp16_nmask.cpp
)

set_source_files_properties(
    example_fmha_fwd_v3.cpp
    fmha_fwd_v3.cpp
    instances/fmha_fwd_v3_d128_bf16_mask.cpp
    instances/fmha_fwd_v3_d128_bf16_nmask.cpp
    instances/fmha_fwd_v3_d128_fp16_mask.cpp
    instances/fmha_fwd_v3_d128_fp16_nmask.cpp
    PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE}
)

target_compile_features(${example_name} PRIVATE cxx_std_17 hip_std_17)
target_compile_options(${example_name} PRIVATE
  -fgpu-flush-denormals-to-zero
  -Wno-undefined-func-template
  --save-temps
)
target_include_directories(${example_name} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(${example_name} PRIVATE composable_kernel::ck_tile hip::host)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mllvm --amdgpu-disable-packed-fp32=1" HAS_DISABLE_PACKED_FP32)
if(HAS_DISABLE_PACKED_FP32)
  target_compile_definitions(${example_name} PRIVATE CK_TILE_DISABLE_PACKED_FP32=1)
  target_compile_options(${example_name} PRIVATE -mllvm --amdgpu-disable-packed-fp32=1)
endif()

add_test(${example_name} COMMAND "${CMAKE_CURRENT_LIST_DIR}/${example_script}" "$<TARGET_FILE:${example_name}>")
