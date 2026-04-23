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

set(example_name ComposableKernel_ck_tile_grouped_convolution)
set(example_fwd ${example_name}_fwd)
set(example_bwd ${example_name}_bwd)

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
filter_hip_architectures("CK Tile grouped convolution example" "gfx908;gfx90a;gfx942;gfx950" SHOULD_SKIP)
if(SHOULD_SKIP)
    return()
endif()

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    message(STATUS "Composable Kernel examples are not supported for CUDA.")
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/ROCmPath.cmake")

find_package(composable_kernel REQUIRED)
include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/FindCKTile.cmake")

find_package(hip REQUIRED)

add_executable(${example_fwd} grouped_convolution_forward.cpp)
add_test(${example_fwd} COMMAND ${example_fwd})
set_source_files_properties(grouped_convolution_forward.cpp PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})
target_compile_features(${example_fwd} PRIVATE cxx_std_17 hip_std_17)
target_compile_options(${example_fwd} PRIVATE -mllvm -enable-noalias-to-md-conversion=0)
target_link_libraries(${example_fwd} PRIVATE composable_kernel::ck_tile hip::host)

add_executable(${example_bwd} grouped_convolution_backward_weight.cpp)
add_test(${example_bwd} COMMAND ${example_bwd})
set_source_files_properties(grouped_convolution_backward_weight.cpp PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})
target_compile_features(${example_bwd} PRIVATE cxx_std_17 hip_std_17)
target_compile_options(${example_bwd} PRIVATE -mllvm -enable-noalias-to-md-conversion=0)
target_link_libraries(${example_bwd} PRIVATE composable_kernel::ck_tile hip::host)

# Additional convolution variants
set(example_fwd_large ${example_name}_fwd_large_tensor)
set(example_fwd_bias_clamp ${example_name}_fwd_bias_clamp)
set(example_bwd_two_stage ${example_name}_bwd_weight_two_stage)
set(example_bwd_data ${example_name}_bwd_data)

add_executable(${example_fwd_large} grouped_convolution_forward_large_tensor.cpp)
set_source_files_properties(grouped_convolution_forward_large_tensor.cpp PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})
target_compile_features(${example_fwd_large} PRIVATE cxx_std_17 hip_std_17)
target_compile_options(${example_fwd_large} PRIVATE -mllvm -enable-noalias-to-md-conversion=0)
target_link_libraries(${example_fwd_large} PRIVATE composable_kernel::ck_tile hip::host)

add_executable(${example_fwd_bias_clamp} grouped_convolution_forward_bias_clamp.cpp)
set_source_files_properties(grouped_convolution_forward_bias_clamp.cpp PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})
target_compile_features(${example_fwd_bias_clamp} PRIVATE cxx_std_17 hip_std_17)
target_compile_options(${example_fwd_bias_clamp} PRIVATE -mllvm -enable-noalias-to-md-conversion=0)
target_link_libraries(${example_fwd_bias_clamp} PRIVATE composable_kernel::ck_tile hip::host)

add_executable(${example_bwd_two_stage} grouped_convolution_backward_weight_two_stage.cpp)
set_source_files_properties(grouped_convolution_backward_weight_two_stage.cpp PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})
target_compile_features(${example_bwd_two_stage} PRIVATE cxx_std_17 hip_std_17)
target_compile_options(${example_bwd_two_stage} PRIVATE -mllvm -enable-noalias-to-md-conversion=0)
target_link_libraries(${example_bwd_two_stage} PRIVATE composable_kernel::ck_tile hip::host)

add_executable(${example_bwd_data} grouped_convolution_backward_data.cpp)
set_source_files_properties(grouped_convolution_backward_data.cpp PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})
target_compile_features(${example_bwd_data} PRIVATE cxx_std_17 hip_std_17)
target_compile_options(${example_bwd_data} PRIVATE -mllvm -enable-noalias-to-md-conversion=0)
target_link_libraries(${example_bwd_data} PRIVATE composable_kernel::ck_tile hip::host)
