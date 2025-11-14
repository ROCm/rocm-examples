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

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(Tools LANGUAGES CXX)
include(CTest)

file(RELATIVE_PATH folder_bin ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${folder_bin})

# The HIP tools are supported neither on Windows nor on CUDA.
if(NOT (CMAKE_SYSTEM_NAME MATCHES "Windows" AND "${GPU_RUNTIME}" STREQUAL "CUDA"))

    find_package(Python3 3.3 QUIET)
    if(NOT Python3_FOUND)
        message(STATUS "Python >= 3.3 not found. Skipping rocprof-compute examples.")
    else()
        add_subdirectory(rocprof-compute)
    endif()

    find_package(rocprofiler-systems COMPONENTS user QUIET)
    if(NOT rocprofiler-systems_FOUND)
        message(STATUS "rocprofiler-systems not found. Skipping rocprof-systems examples.")
    else()
        add_subdirectory(rocprof-systems)
    endif()

    find_package(rocprofiler-sdk-roctx QUIET)
    if(NOT rocprofiler-sdk-roctx_FOUND)
        message(STATUS "rocprofiler-sdk-roctx not found. Skipping rocprofv3 examples.")
    else()
        add_subdirectory(rocprofv3)
    endif()
endif()
