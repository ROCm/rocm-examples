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

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(hipTensor_examples LANGUAGES CXX)
include(CTest)

file(RELATIVE_PATH folder_bin ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${folder_bin})

include("${CMAKE_CURRENT_LIST_DIR}/../../Common/ROCmPath.cmake")

find_package(hiptensor QUIET)
if(NOT hiptensor_FOUND)
    message(
        WARNING
        "hipTensor could not be found, not building hipTensor examples"
    )
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../Common/FilterHIPArchitectures.cmake")
hiptensor_get_supported_architectures(HIPTENSOR_SUPPORTED_ARCHS)
filter_hip_architectures("hipTensor" "${HIPTENSOR_SUPPORTED_ARCHS}" SHOULD_SKIP)
if(SHOULD_SKIP)
    return()
endif()

add_subdirectory(contraction)
add_subdirectory(elementwise)
add_subdirectory(reduction)
