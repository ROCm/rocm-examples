# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
project(hipThreads_examples LANGUAGES CXX HIP)
include(CTest)

file(RELATIVE_PATH folder_bin ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${folder_bin})

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    message(STATUS "hipThreads examples do not support the CUDA runtime")
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../Common/ROCmPath.cmake")
list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} /opt/rocm)

find_package(hipthreads QUIET)
if(NOT hipthreads_FOUND)
    message(WARNING "hipThreads could not be found, not building hipThreads examples")
    return()
endif()

find_package(rocprim QUIET CONFIG)
if(NOT rocprim_FOUND)
    message(WARNING "rocPRIM could not be found, not building hipThreads examples")
    return()
endif()

find_package(rocthrust QUIET CONFIG)
if(NOT rocthrust_FOUND)
    message(WARNING "rocThrust could not be found, not building hipThreads examples")
    return()
endif()

add_subdirectory(saxpy)
add_subdirectory(in_one_weekend_raytracer)
add_subdirectory(llama3c)
add_subdirectory(sparse_mat_mul)
