# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
project(hipSOLVER_examples LANGUAGES CXX)
include(CTest)

file(RELATIVE_PATH folder_bin ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${folder_bin})

if(WIN32)
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

find_package(hipsolver)
if(NOT hipsolver_FOUND)
    message(
        STATUS
        "hipSOLVER could not be found, not building hipSOLVER examples"
    )
    return(1)
endif()

add_subdirectory(gels)
add_subdirectory(geqrf)
add_subdirectory(gesvd)
add_subdirectory(getrf)
add_subdirectory(potrf)
add_subdirectory(syevd)
add_subdirectory(syevdx)
add_subdirectory(syevj)
add_subdirectory(syevj_batched)
add_subdirectory(sygvd)

# this example currently does not work with CUDA
# https://github.com/ROCmSoftwarePlatform/hipSOLVER/issues/152
if(NOT ("${GPU_RUNTIME}" STREQUAL "CUDA"))
    add_subdirectory(sygvj)
endif()
