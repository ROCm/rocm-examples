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

set(example_name hip_memory_pool_interprocess_handle)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(FATAL_ERROR "This example is supported only on Linux.")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../../Common/HipPlatform.cmake")
select_gpu_language()

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "HIP")
    set(USED_LANGUAGE "CXX")
    find_package(hip REQUIRED)
else()
    set(USED_LANGUAGE "CUDA")
endif()

enable_language(${USED_LANGUAGE})
select_hip_platform()

set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

###############################################################################
# Exporter target
###############################################################################

add_executable(${example_name}_exporter exporter.cpp)
# Make example runnable using ctest
add_test(NAME ${example_name}_exporter COMMAND ${example_name}_exporter)

set_source_files_properties(exporter.cpp PROPERTIES LANGUAGE ${USED_LANGUAGE})

set_target_properties(${example_name}_exporter PROPERTIES $<$<COMPILE_LANGUAGE:CUDA>:CUDA_EXTENSIONS OFF>
                                                          $<$<COMPILE_LANGUAGE:CXX>:CXX_EXTENSIONS OFF>)
target_compile_features(${example_name}_exporter PRIVATE cuda_std_17 cxx_std_17)
target_include_directories(${example_name}_exporter PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${ROCM_ROOT}/include>)
target_link_libraries(${example_name}_exporter PRIVATE $<$<NOT:$<LINK_LANGUAGE:CUDA>>:hip::host>)

###############################################################################
# Importer target
###############################################################################

add_executable(${example_name}_importer importer.cpp)
# Make example runnable using ctest
add_test(NAME ${example_name}_importer COMMAND ${example_name}_importer)

set_source_files_properties(importer.cpp PROPERTIES LANGUAGE ${USED_LANGUAGE})

set_target_properties(${example_name}_importer PROPERTIES $<$<COMPILE_LANGUAGE:CUDA>:CUDA_EXTENSIONS OFF>
                                                          $<$<COMPILE_LANGUAGE:CXX>:CXX_EXTENSIONS OFF>)
target_compile_features(${example_name}_importer PRIVATE cuda_std_17 cxx_std_17)
target_include_directories(${example_name}_importer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${ROCM_ROOT}/include>)
target_link_libraries(${example_name}_importer PRIVATE $<$<NOT:$<LINK_LANGUAGE:CUDA>>:hip::host>)

# Tell CTest that these tests belong to the same group and hope that they will execute concurrently.
set_tests_properties(${example_name}_exporter ${example_name}_importer PROPERTIES FIXTURES_REQUIRED ${example_name})

install(TARGETS ${example_name}_exporter ${example_name}_importer)
