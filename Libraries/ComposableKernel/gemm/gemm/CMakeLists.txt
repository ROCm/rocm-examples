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

set(example_name ComposableKernel_ck_tile_gemm)
set(example_basic ${example_name}_basic)
set(example_basic_script test_ck_tile_gemm_basic.sh)
set(example_universal ${example_name}_universal)
set(example_universal_script test_ck_tile_gemm_universal.sh)
set(example_weight_preshuffle ${example_name}_weight_preshuffle)

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
filter_hip_architectures("CK Tile GEMM example" "gfx908;gfx90a;gfx942;gfx950" SHOULD_SKIP)
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

set_source_files_properties(
    gemm_basic.cpp
    gemm_weight_preshuffle.cpp
    universal_gemm.cpp
    PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE}
)

add_executable(${example_basic} gemm_basic.cpp)
add_test(${example_basic} COMMAND "${CMAKE_CURRENT_LIST_DIR}/${example_basic_script} $<TARGET_FILE:${example_basic}>")
target_compile_features(${example_basic} PRIVATE cxx_std_17 hip_std_17)
target_compile_options(${example_basic} PRIVATE -mllvm -enable-noalias-to-md-conversion=0)
target_link_libraries(${example_basic} PRIVATE composable_kernel::ck_tile hip::host)

add_executable(${example_universal} universal_gemm.cpp)
add_test(${example_universal}
         COMMAND "${CMAKE_CURRENT_LIST_DIR}/${example_universal_script} $<TARGET_FILE:${example_universal}>"
)
target_compile_features(${example_universal} PRIVATE cxx_std_17 hip_std_17)
target_compile_options(${example_universal} PRIVATE -mllvm -enable-noalias-to-md-conversion=0)
target_link_libraries(${example_universal} PRIVATE composable_kernel::ck_tile hip::host)

add_executable(${example_weight_preshuffle} gemm_weight_preshuffle.cpp)
add_test(
    ${example_weight_preshuffle}
    COMMAND "${CMAKE_CURRENT_LIST_DIR}/${example_basic_script} $<TARGET_FILE:${example_weight_preshuffle}>"
)
target_compile_features(${example_weight_preshuffle} PRIVATE cxx_std_17 hip_std_17)
target_compile_options(
    ${example_weight_preshuffle}
    PRIVATE
    -Wno-unused-local-typedef
    -Wno-gnu-line-marker
    --save-temps
    "SHELL:-mllvm -greedy-reverse-local-assignment=1 -mllvm --slp-threshold=-32 -mllvm -enable-noalias-to-md-conversion=0"
)
target_link_libraries(${example_weight_preshuffle} PRIVATE composable_kernel::ck_tile hip::host)

# Additional executables for splitk two-stage
set(example_splitk_reduce ${example_name}_splitk_reduce)
set(example_splitk_two_stage ${example_name}_splitk_two_stage)

add_executable(${example_splitk_reduce} gemm_splitk_two_stage_reduce.cpp)
target_compile_features(${example_splitk_reduce} PRIVATE cxx_std_17 hip_std_17)
target_compile_options(${example_splitk_reduce} PRIVATE -mllvm -enable-noalias-to-md-conversion=0)
target_link_libraries(${example_splitk_reduce} PRIVATE composable_kernel::ck_tile hip::host)
set_source_files_properties(gemm_splitk_two_stage_reduce.cpp PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

add_executable(${example_splitk_two_stage} gemm_splitk_two_stage.cpp)
target_compile_features(${example_splitk_two_stage} PRIVATE cxx_std_17 hip_std_17)
target_compile_options(${example_splitk_two_stage} PRIVATE -mllvm -enable-noalias-to-md-conversion=0)
target_link_libraries(${example_splitk_two_stage} PRIVATE composable_kernel::ck_tile hip::host)
set_source_files_properties(gemm_splitk_two_stage.cpp PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

