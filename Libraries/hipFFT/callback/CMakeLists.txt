# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

set(example_name hipfft_callback)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(hipfft_callback LANGUAGES CXX)

# Set default build type to Release
# to workaround a compiler issue when building with -O0
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../Common/ROCmPath.cmake")

find_package(hipfft REQUIRED)
find_package(hip REQUIRED)
find_package(hiprtc REQUIRED)

add_executable(${example_name} main.cpp)
# Make example runnable using ctest
add_test(NAME ${example_name} COMMAND ${example_name})

set(include_dirs "../../../Common" "../../../External")

target_link_libraries(${example_name} PRIVATE hip::hipfft hip::host)
target_include_directories(${example_name} PRIVATE ${include_dirs})

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${example_name} PRIVATE CUDA::cudart CUDA::nvrtc)
  # on CUDA, RTC callback code needs to be told where the CUDA includes are
  list( GET CUDAToolkit_INCLUDE_DIRS 0 CUDA_INCLUDE_DIR )
  target_compile_definitions( ${example_name} PRIVATE __HIP_PLATFORM_NVIDIA__ CUDA_INCLUDE_DIR=\"${CUDA_INCLUDE_DIR}\" )
else()
  target_link_libraries(${example_name} PRIVATE  hiprtc::hiprtc)
endif()

install(TARGETS ${example_name})
