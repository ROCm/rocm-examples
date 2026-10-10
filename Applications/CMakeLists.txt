# MIT License
#
# Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(Applications LANGUAGES CXX)
include(CTest)

file(RELATIVE_PATH folder_bin ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${folder_bin})

include("${CMAKE_CURRENT_LIST_DIR}/../Common/FindFilesystem.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../Common/HipPlatform.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../Common/ROCmPath.cmake")
select_gpu_language()
# select_hip_platform requires the language to be enabled
enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()

find_package(glfw3 QUIET)

add_subdirectory(bitonic_sort)
add_subdirectory(convolution)
add_subdirectory(floyd_warshall)
add_subdirectory(histogram)
add_subdirectory(prefix_sum)
add_subdirectory(optical_flow)

if(ROCM_EXAMPLES_HIP_PLATFORM STREQUAL "amd")
    add_subdirectory(pqc_trustflow_rocm)
endif()

if(NOT CXX_FS_HEADER_FOUND)
    message(WARNING "filesystem not found, not building fdtd example")
else()
    add_subdirectory(fdtd)
endif()

if(NOT (CMAKE_SYSTEM_NAME MATCHES "Windows" AND ROCM_EXAMPLES_HIP_PLATFORM STREQUAL "nvidia"))
    add_subdirectory(monte_carlo_pi)
endif()

if(NOT glfw3_FOUND)
    message(WARNING "GLFW not found, not building sobel filter example")
else()
    add_subdirectory(sobel_filter)
endif()
