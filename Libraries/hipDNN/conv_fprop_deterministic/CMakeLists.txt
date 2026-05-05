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

set(example_name hipdnn_conv_fprop_deterministic)

cmake_minimum_required(VERSION 3.25.2 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    message(STATUS "hipDNN examples do not support the CUDA runtime")
    return()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include("${CMAKE_CURRENT_LIST_DIR}/../../../Common/ROCmPath.cmake")

find_package(hipdnn_frontend REQUIRED CONFIG)
find_package(hipdnn_test_sdk REQUIRED CONFIG)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

add_executable(${example_name} main.cpp)
add_test(NAME ${example_name} COMMAND ${example_name} --verify-cpu)

target_link_libraries(${example_name} PRIVATE
    hip::host
    Threads::Threads
    hipdnn_frontend
    hipdnn_test_sdk
)

target_compile_options(${example_name} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/GR->
)

target_include_directories(${example_name} PRIVATE
    "../../../Common"
    "../../../External"
)

install(TARGETS ${example_name})
