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
project(HIP-Programming-Guide LANGUAGES CXX)
include(CTest)

include("${CMAKE_CURRENT_LIST_DIR}/../Common/HipPlatform.cmake")
select_gpu_language()
enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()

add_subdirectory(add_kernel)

if(NOT ${ROCM_EXAMPLES_GPU_LANGUAGE} STREQUAL "CUDA")
    message(STATUS "The block reduction example is available only for CUDA.")
else()
    add_subdirectory(block_reduction)
endif()

add_subdirectory(call_stack_management)
add_subdirectory(complex_math)
add_subdirectory(constant_memory)
add_subdirectory(data_prefetching)
add_subdirectory(device_recursion)
add_subdirectory(dynamic_shared_memory)
add_subdirectory(dynamic_unified_memory)
add_subdirectory(error_handling)
add_subdirectory(explicit_copy)
add_subdirectory(explicit_memory)
add_subdirectory(kernel_memory_allocation)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Low-precision floating-point examples are available only on Linux.")
elseif(${ROCM_EXAMPLES_GPU_LANGUAGE} STREQUAL "CUDA")
    message(STATUS "Low-precision floating-point examples are available only for the HIP runtime.")
else()
    add_subdirectory(low_precision_float_fp8)
    add_subdirectory(low_precision_float_fp16)
endif()

add_subdirectory(math)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "The memory pool examples are available only on Linux.")
else()
    add_subdirectory(memory_pool)
    add_subdirectory(memory_pool_interprocess_handle)
    add_subdirectory(memory_pool_interprocess_pointer)
    add_subdirectory(memory_pool_resource_usage_statistics)
    add_subdirectory(memory_pool_threshold)
    add_subdirectory(memory_pool_trim)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "The memory range attributes example is available only on Linux.")
else()
    add_subdirectory(memory_range_attributes)
endif()

add_subdirectory(ordinary_memory_allocation)
add_subdirectory(pageable_host_memory)
add_subdirectory(pinned_host_memory)
add_subdirectory(simple_device_query)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "The standard unified memory example is available only on Linux.")
elseif(NOT DEFINED ENV{HSA_XNACK})
    message(STATUS "Environment variable HSA_XNACK not set, not building standard unified memory example.")
elseif(NOT ENV{HSA_XNACK} EQUAL 1)
    message(STATUS "Environment variable HSA_XNACK not set to 1, not building standard unified memory example.")
else()
    add_subdirectory(standard_unified_memory)
endif()

add_subdirectory(static_shared_memory)
add_subdirectory(static_unified_memory)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "The stream ordered memory allocation example is available only on Linux.")
else()
    add_subdirectory(stream_ordered_memory_allocation)
endif()

add_subdirectory(unified_memory_advice)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "The virtual memory example is available only on Linux.")
elseif(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    message(STATUS "The virtual memory example is currently not supported for CUDA.")
else()
    add_subdirectory(virtual_memory)
endif()
