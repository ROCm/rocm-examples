# MIT License
#
# Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
project(rocrand_simple_distributions_cpp LANGUAGES CXX)

set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
set(GPU_RUNTIMES "HIP" "CUDA")
set_property(CACHE GPU_RUNTIME PROPERTY STRINGS ${GPU_RUNTIMES})

if(NOT "${GPU_RUNTIME}" IN_LIST GPU_RUNTIMES)
    message(FATAL_ERROR "Only the following values are accepted for GPU_RUNTIME: ${GPU_RUNTIMES}")
endif()

if(GPU_RUNTIME STREQUAL "CUDA")
    set(LANG "CUDA")
    enable_language(CUDA)
else()
    set(LANG "CXX")
endif()

set(CMAKE_${LANG}_STANDARD 17)
set(CMAKE_${LANG}_EXTENSIONS OFF)
set(CMAKE_${LANG}_STANDARD_REQUIRED ON)

if(NOT CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH "/opt/rocm")
endif()

find_package(rocrand REQUIRED)

add_executable(rocrand_simple_distributions_cpp main.cpp)
add_test(rocrand_simple_distributions_cpp rocrand_simple_distributions_cpp)

if(GPU_RUNTIME STREQUAL "CUDA")
    target_link_libraries(rocrand_simple_distributions_cpp PRIVATE roc::rocrand)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
else()
    target_link_libraries(rocrand_simple_distributions_cpp roc::rocrand hip::host)
endif()

target_include_directories(rocrand_simple_distributions_cpp PRIVATE "../../../Common")
