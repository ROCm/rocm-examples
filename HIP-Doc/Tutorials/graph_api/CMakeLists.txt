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

cmake_minimum_required(VERSION 3.21)

project(hip_graph_api_tutorial LANGUAGES CXX)

# Check for general C++20 support
list(FIND CMAKE_CXX_COMPILE_FEATURES cxx_std_20 COMPILER_KNOWS_CXX20)
if(COMPILER_KNOWS_CXX20 EQUAL -1)
    message(WARNING "The C++ compiler does not support C++20. Not building the HIP Graph API tutorial examples.")
    return()
endif()

# Check for C++20 features we use in the tutorial
include(CheckIncludeFileCXX)
set(CMAKE_REQUIRED_QUIET TRUE)
check_include_file_cxx(numbers HAVE_NUMBERS)
if(NOT HAVE_NUMBERS)
    message(WARNING "<numbers> not found. Not building the HIP Graph API tutorial examples.")
    return()
endif()

check_include_file_cxx(source_location HAVE_SOURCE_LOC)
if(NOT HAVE_SOURCE_LOC)
    message(WARNING "<source_location> not found. Not building the HIP Graph API tutorial examples.")
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../Common/HipPlatform.cmake")
select_gpu_language()
enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()

include("${CMAKE_CURRENT_LIST_DIR}/../../../Common/ROCmPath.cmake")

add_subdirectory(src)
