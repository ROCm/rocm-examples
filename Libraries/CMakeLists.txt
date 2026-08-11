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
project(Libraries LANGUAGES CXX)
include(CTest)

include("${CMAKE_CURRENT_LIST_DIR}/../Common/ROCmPath.cmake")

file(RELATIVE_PATH folder_bin ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${folder_bin})

option(ROCM_EXAMPLES_ENABLE_CK "Enable Composable Kernel examples (disabled by default)" OFF)
option(ROCM_EXAMPLES_ENABLE_ROCDECODE "Enable rocDecode examples (enabled by default)" ON)
option(ROCM_EXAMPLES_ENABLE_HIPDNN "Enable hipDNN examples (enabled by default)" ON)

# CMake configuration files for CUDA versions of HIP libraries are not yet
# included under the HIP SDK for Windows.
if(
    NOT (CMAKE_SYSTEM_NAME MATCHES "Windows" AND "${ROCM_EXAMPLES_GPU_LANGUAGE}" STREQUAL "CUDA")
)
    if(ROCM_EXAMPLES_ENABLE_CK)
        add_subdirectory(ComposableKernel)
    else()
        message(WARNING "Composable Kernel examples disabled, not building Composable Kernel examples.")
    endif()
    add_subdirectory(hipBLAS)
    if(CMAKE_VERSION VERSION_LESS "3.25.2")
        message(WARNING "CMake version < 3.25.2, not building hipBLASLt examples.")
    else()
        add_subdirectory(hipBLASLt)
    endif()
    add_subdirectory(hipCUB)
    if(CMAKE_VERSION VERSION_LESS "3.25.2")
        message(WARNING "CMake version < 3.25.2, not building hipDNN examples.")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13")
        message(WARNING "Current GCC version is ${CMAKE_CXX_COMPILER_VERSION} which is < 13, not building hipDNN examples.")
    else()
        if (ROCM_EXAMPLES_ENABLE_HIPDNN)
            add_subdirectory(hipDNN)
        else()
            message(STATUS "hipDNN examples disabled, not building hipDNN examples.")
        endif()
    endif()
    add_subdirectory(hipFFT)
    add_subdirectory(hipRAND)
    add_subdirectory(hipThreads)
    add_subdirectory(hipSOLVER)
    add_subdirectory(hipSPARSE)
    add_subdirectory(hipSPARSELt)
    add_subdirectory(hipTensor)
    if(CMAKE_VERSION VERSION_LESS "3.25.2")
        message(WARNING "CMake version < 3.25.2, not building MIGraphX examples.")
    else()
        add_subdirectory(MIGraphX)
    endif()
    add_subdirectory(MIVisionX)
    add_subdirectory(RCCL)
    add_subdirectory(rocAL)
    add_subdirectory(rocALUTION)
    add_subdirectory(rocBLAS)
    add_subdirectory(rocCV)

    if(ROCM_EXAMPLES_ENABLE_ROCDECODE)
        add_subdirectory(rocDecode)
    else()
        message(STATUS "rocDecode examples disabled, not building rocDecode examples.")
    endif()

    add_subdirectory(rocJPEG)
    add_subdirectory(rocFFT)
    add_subdirectory(rocPRIM)
    add_subdirectory(rocProfiler-SDK)
    add_subdirectory(rocRAND)
    add_subdirectory(rocSOLVER)
    add_subdirectory(rocSPARSE)
    add_subdirectory(rocThrust)
    add_subdirectory(rocWMMA)
    add_subdirectory(RPP)
endif()
