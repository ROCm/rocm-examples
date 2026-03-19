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

set(example_name hip_low_precision_float_fp8)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(FATAL_ERROR "This example is not supported on Windows.")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/HipPlatform.cmake")
select_gpu_language()

enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()

set(ROCM_PATH "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")

list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

add_executable(${example_name} main.hip)

# Make example runnable using ctest
add_test(NAME ${example_name} COMMAND ${example_name})

set_source_files_properties(main.hip PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

set_target_properties(${example_name} PROPERTIES $<$<COMPILE_LANGUAGE:CUDA>:CUDA_EXTENSIONS OFF>
                                                 $<$<COMPILE_LANGUAGE:HIP>:HIP_EXTENSIONS OFF>)
target_compile_features(${example_name} PRIVATE cuda_std_17 hip_std_17)
target_include_directories(${example_name} PRIVATE ${include_dirs} $<$<COMPILE_LANGUAGE:CUDA>:${ROCM_PATH}/include>)

install(TARGETS ${example_name})
