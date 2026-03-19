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
project(rocWMMA_examples LANGUAGES CXX)

file(RELATIVE_PATH folder_bin ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${folder_bin})

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    message(STATUS "rocWMMA examples do not support the CUDA runtime")
    return()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(ROCM_PATH
        "$ENV{HIP_PATH}"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
else()
    set(ROCM_PATH
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

find_package(hip)
# Try to find rocWMMA as a package first
find_package(rocwmma QUIET)

# Remove OpenMP dependency from rocwmma on Windows
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    if(TARGET roc::rocwmma)
        get_target_property(ROCWMMA_LINK_LIBS roc::rocwmma INTERFACE_LINK_LIBRARIES)
        if(ROCWMMA_LINK_LIBS)
            list(REMOVE_ITEM ROCWMMA_LINK_LIBS OpenMP::OpenMP_CXX)
            list(REMOVE_ITEM ROCWMMA_LINK_LIBS OpenMP::omp)
            set_target_properties(roc::rocwmma PROPERTIES
                INTERFACE_LINK_LIBRARIES "${ROCWMMA_LINK_LIBS}"
            )
        endif()
    elseif(TARGET rocwmma)
        get_target_property(ROCWMMA_LINK_LIBS rocwmma INTERFACE_LINK_LIBRARIES)
        if(ROCWMMA_LINK_LIBS)
            list(REMOVE_ITEM ROCWMMA_LINK_LIBS OpenMP::OpenMP_CXX)
            list(REMOVE_ITEM ROCWMMA_LINK_LIBS OpenMP::omp)
            set_target_properties(rocwmma PROPERTIES
                INTERFACE_LINK_LIBRARIES "${ROCWMMA_LINK_LIBS}"
            )
        endif()
    endif()
endif()

if(TARGET roc::rocwmma)
    set(ROCWMMA_TARGET roc::rocwmma)
elseif(TARGET rocwmma)
    # Fallback for legacy naming
    set(ROCWMMA_TARGET rocwmma)
endif()

if(NOT ROCWMMA_TARGET AND NOT rocwmma_FOUND)
    # If package not found, try to find headers manually
    find_path(ROCWMMA_INCLUDE_DIR
        NAMES rocwmma/rocwmma.hpp
        PATHS ${ROCM_PATH}/include
        DOC "rocWMMA include directory"
    )

    if(NOT ROCWMMA_INCLUDE_DIR)
        message(WARNING "rocWMMA headers not found. Not building rocWMMA examples. Please install rocWMMA or set ROCWMMA_INCLUDE_DIR")
        return()
    endif()

    # Create imported target for header-only library
    add_library(rocwmma INTERFACE)
    target_include_directories(rocwmma INTERFACE ${ROCWMMA_INCLUDE_DIR})
    set(ROCWMMA_TARGET rocwmma)
endif()

add_subdirectory(hiprtc_gemm)
add_subdirectory(perf_dgemm)
add_subdirectory(perf_hgemm)
add_subdirectory(perf_i8gemm)
add_subdirectory(perf_sgemm)
add_subdirectory(simple_dgemm)
add_subdirectory(simple_dgemv)
add_subdirectory(simple_dlrm)
add_subdirectory(simple_hgemm)
add_subdirectory(simple_sgemm)
add_subdirectory(simple_sgemv)
