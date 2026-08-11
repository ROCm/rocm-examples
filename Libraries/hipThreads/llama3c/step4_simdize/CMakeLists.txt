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

set(example_name hipthreads_llama3c_step4_simdize)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX HIP)

set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    message(STATUS "hipThreads examples do not support the CUDA runtime")
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/ROCmPath.cmake")
if(NOT CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH "$ENV{ROCM_PATH};/opt/rocm")
endif()

find_package(hipthreads REQUIRED)
find_package(rocprim REQUIRED CONFIG)
find_package(rocthrust REQUIRED CONFIG)

set_source_files_properties(llama3.cxx PROPERTIES LANGUAGE HIP)

add_executable(${example_name} llama3.cxx)
target_compile_options(${example_name} PRIVATE -fgpu-rdc)
target_link_options(${example_name} PRIVATE -fgpu-rdc)
target_link_libraries(${example_name} PRIVATE hipthreads::hipthreads roc::rocthrust)

target_include_directories(${example_name} PRIVATE "../../../../Common")

install(TARGETS ${example_name})
