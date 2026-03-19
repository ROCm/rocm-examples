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
project(rocALUTION_examples LANGUAGES CXX)
include(CTest)

file(RELATIVE_PATH folder_bin ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${folder_bin})

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "rocALUTION examples are only available on Linux")
    return()
else()
    set(ROCM_PATH
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

find_package(rocalution QUIET)
if(NOT rocalution_FOUND)
    message(
        WARNING
        "rocALUTION could not be found, not building rocALUTION examples"
    )
    return()
endif()

add_subdirectory(amg)
add_subdirectory(sa_amg)
add_subdirectory(ua_amg)
add_subdirectory(as_precond)
add_subdirectory(async)
add_subdirectory(benchmark)
add_subdirectory(bicgstab)
add_subdirectory(block_precond)
add_subdirectory(cg_amg)
add_subdirectory(cg_rsamg)
add_subdirectory(cg_saamg)
add_subdirectory(cg)
add_subdirectory(cmk)
add_subdirectory(complex)
add_subdirectory(direct)
add_subdirectory(fcg)
add_subdirectory(fixed_point)
add_subdirectory(gmres)
add_subdirectory(fgmres)
add_subdirectory(idr)
add_subdirectory(key)
add_subdirectory(me_preconditioner)
add_subdirectory(mixed_precision)
add_subdirectory(power_method)
add_subdirectory(simple_spmv)
add_subdirectory(stencil)
add_subdirectory(tns)
add_subdirectory(var_precond)
add_subdirectory(itsolve)
