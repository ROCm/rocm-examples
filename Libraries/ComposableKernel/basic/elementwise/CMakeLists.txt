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

set(example_name ComposableKernel_ck_tile_elementwise)
set(example_name_add_4d ${example_name}_add_4d)
set(example_name_transpose ${example_name}_transpose)
set(example_name_unary ${example_name}_unary)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Composable Kernel examples are supported only on Linux.")
    return(1)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/HipPlatform.cmake")
select_gpu_language()
enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    message(STATUS "Composable Kernel examples are not supported for CUDA.")
    return(1)
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

add_executable(${example_name} elementwise_example.cpp)
add_test(${example_name} COMMAND ${example_name})
set_source_files_properties(elementwise_example.cpp PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})
target_link_libraries(${example_name} PRIVATE composable_kernel::ck_tile)

add_executable(${example_name_add_4d} elementwise_example_add_4d.cpp)
add_test(${example_name_add_4d} COMMAND ${example_name_add_4d})
set_source_files_properties(elementwise_example_add_4d.cpp PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})
target_link_libraries(${example_name_add_4d} PRIVATE composable_kernel::ck_tile)

add_executable(${example_name_transpose} elementwise_example_transpose.cpp)
add_test(${example_name_transpose} COMMAND ${example_name_transpose})
set_source_files_properties(elementwise_example_transpose.cpp PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})
target_link_libraries(${example_name_transpose} PRIVATE composable_kernel::ck_tile)

add_executable(${example_name_unary} elementwise_example_unary.cpp)
add_test(${example_name_unary} COMMAND ${example_name_unary})
set_source_files_properties(elementwise_example_unary.cpp PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})
target_link_libraries(${example_name_unary} PRIVATE composable_kernel::ck_tile)