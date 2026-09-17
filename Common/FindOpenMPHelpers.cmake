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

include(${CMAKE_CURRENT_LIST_DIR}/ROCmPath.cmake)

# Determine LLVM host triple as libomptarget might be installed there.
function(find_llvm_host_triple RESULT_VAR)
    find_program(
        amdclang_EXECUTABLE
        NAMES amdclang
        HINTS ${ROCM_PATH}
        PATHS /opt/rocm
        PATH_SUFFIXES bin llvm/bin lib/llvm/bin
        REQUIRED
    )
    if(amdclang_EXECUTABLE)
        execute_process(
            COMMAND "${amdclang_EXECUTABLE}" --print-target-triple
            OUTPUT_VARIABLE _llvm_host_triple
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _llvm_host_triple_result
        )
        if(NOT _llvm_host_triple_result EQUAL 0)
            message(
                FATAL_ERROR
                "Failed to query LLVM host triple from ${amdclang_EXECUTABLE}; "
                "libomp discovery for LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON may not work."
            )
        endif()
    else()
        message(
            WARNING
            "Could not find amdclang to determine the LLVM host triple; "
            "libomp discovery for LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON may not work."
            )
    endif()
    set("${RESULT_VAR}" "${_llvm_host_triple}" PARENT_SCOPE)
endfunction()

function(find_omp_library RESULT_VAR)
    # Try to locate openmp-config.cmake in ROCm install
    find_package(OpenMP CONFIG PATHS ${ROCM_PATH}/lib/llvm/lib/cmake)
    if (OpenMP_FOUND)
        set("${RESULT_VAR}" OpenMP::OpenMP_CXX PARENT_SCOPE)
        message(STATUS "OpenMP found with openmp-config.cmake")
    # Fallback to find_library with LLVM_HOST_TRIPLE included. This covers the per target runtime directory.
    else()
        find_llvm_host_triple(LLVM_HOST_TRIPLE)
        find_library(_omp_lib
            NAMES omp
            PATHS
                "${ROCM_PATH}/lib/llvm/lib"
                "${ROCM_PATH}/lib"
            PATH_SUFFIXES lib lib64 ${LLVM_HOST_TRIPLE}
            REQUIRED
        )
        set("${RESULT_VAR}" "${_omp_lib}" PARENT_SCOPE)
        message(STATUS "OpenMP found with find_library at ${_omp_lib}")
    endif()
endfunction()
