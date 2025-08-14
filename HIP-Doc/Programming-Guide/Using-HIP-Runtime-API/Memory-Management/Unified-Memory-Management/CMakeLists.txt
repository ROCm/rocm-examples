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
project(HIP-Doc-Programming-Guide-Using-HIP-Runtime-API-Memory-Management-Unified-Memory-Management
        LANGUAGES CXX)

add_subdirectory(data_prefetching)
add_subdirectory(dynamic_unified_memory)
add_subdirectory(explicit_memory)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "The memory range attributes example is available only on Linux.")
else()
    add_subdirectory(memory_range_attributes)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "The standard unified memory example is available only on Linux.")
elseif(NOT DEFINED ENV{HSA_XNACK})
    message(STATUS "Environment variable HSA_XNACK not set, not building standard unified memory example.")
elseif(NOT ENV{HSA_XNACK} EQUAL 1)
    message(STATUS "Environment variable HSA_XNACK not set to 1, not building standard unified memory example.")
else()
    add_subdirectory(standard_unified_memory)
endif()

add_subdirectory(static_unified_memory)
add_subdirectory(unified_memory_advice)
