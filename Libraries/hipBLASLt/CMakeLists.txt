# MIT License
#
# Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.25.2 FATAL_ERROR)
project(hipBLASLt_examples LANGUAGES CXX)
include(CTest)

file(RELATIVE_PATH folder_bin ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${folder_bin})

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "hipBLASLt examples are only available on Linux")
    return()
else()
    set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

find_package(hipblaslt)
if(NOT hipblaslt_FOUND)
    message(STATUS "hipBLASLt could not be found, not building hipBLASLt examples")
    return(1)
endif()

add_subdirectory(ext_op_amax)
add_subdirectory(ext_op_layernorm)
add_subdirectory(gemm)
add_subdirectory(gemm_alphavec_ext)
add_subdirectory(gemm_amax)
add_subdirectory(gemm_amax_ext)
add_subdirectory(gemm_amax_with_scale)
add_subdirectory(gemm_amax_with_scale_ext)
add_subdirectory(gemm_attr_tciA_tciB)
add_subdirectory(gemm_batched)
add_subdirectory(gemm_batched_ext)
add_subdirectory(gemm_bgradb)
add_subdirectory(gemm_bias)
add_subdirectory(gemm_bias_ext)
add_subdirectory(gemm_bias_swizzle_a_ext)
add_subdirectory(gemm_clamp_bias)
add_subdirectory(gemm_dgelu_bgradb)
add_subdirectory(gemm_dgelu_bgradb_ext)
add_subdirectory(gemm_ext)
add_subdirectory(gemm_ext_bgradb)
add_subdirectory(gemm_gelu_aux_bias)
add_subdirectory(gemm_gelu_aux_bias_ext)
add_subdirectory(gemm_get_algo_by_index_ext)
add_subdirectory(gemm_get_all_algos)
add_subdirectory(gemm_get_all_algos_ext)
add_subdirectory(gemm_is_tuned_ext)
add_subdirectory(gemm_mix_precision)
add_subdirectory(gemm_mix_precision_ext)
add_subdirectory(gemm_mix_precision_with_amax_ext)
add_subdirectory(gemm_swish_bias)
add_subdirectory(gemm_swizzle_a)
add_subdirectory(gemm_swizzle_a_scale_a_b_vector)
add_subdirectory(gemm_tuning_splitk_ext)
add_subdirectory(gemm_tuning_wgm_ext)
add_subdirectory(gemm_with_scale_a_b)
add_subdirectory(gemm_with_scale_a_b_ext)
add_subdirectory(gemm_with_scale_a_b_vector)
add_subdirectory(gemm_with_TF32)
add_subdirectory(groupedgemm_ext)
add_subdirectory(groupedgemm_fixed_mk_ext)
# Disabled until fix is upstreamed in later ROCm version
# add_subdirectory(groupedgemm_get_all_algos_ext)
add_subdirectory(weight_swizzle_padding)
