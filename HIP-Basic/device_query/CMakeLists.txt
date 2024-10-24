# MIT License
#
# Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

set(example_name hip_device_query)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
set(GPU_RUNTIMES "HIP" "CUDA")
set_property(CACHE GPU_RUNTIME PROPERTY STRINGS ${GPU_RUNTIMES})

if(NOT "${GPU_RUNTIME}" IN_LIST GPU_RUNTIMES)
    set(ERROR_MESSAGE
        "GPU_RUNTIME is set to \"${GPU_RUNTIME}\".\nGPU_RUNTIME must be either HIP or CUDA."
    )
    message(FATAL_ERROR ${ERROR_MESSAGE})
endif()

if(GPU_RUNTIME STREQUAL "HIP")
    # This example does not contain device code, thereby it can be compiled with any conforming C++ compiler.
    set(USED_LANGUAGE "CXX")
else()
    set(USED_LANGUAGE "CUDA")
    enable_language(${USED_LANGUAGE})
endif()

set(CMAKE_${USED_LANGUAGE}_STANDARD 17)
set(CMAKE_${USED_LANGUAGE}_EXTENSIONS OFF)
set(CMAKE_${USED_LANGUAGE}_STANDARD_REQUIRED ON)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(ROCM_ROOT
        "$ENV{HIP_PATH}"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
else()
    set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

add_executable(${example_name} main.cpp)
# Make example runnable using ctest
add_test(NAME ${example_name} COMMAND ${example_name})

set(include_dirs "../../Common" "../../External")
# For examples targeting NVIDIA, include the HIP header directory.
if(GPU_RUNTIME STREQUAL "CUDA")
    list(APPEND include_dirs "${ROCM_ROOT}/include")
endif()

target_include_directories(${example_name} PRIVATE ${include_dirs})
set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${USED_LANGUAGE})

if(GPU_RUNTIME STREQUAL "HIP")
    # When an ordinary C++ compiler is used (on the ROCm backend),
    # it must be set up to link to the HIP runtime.
    target_compile_definitions(${example_name} PRIVATE __HIP_PLATFORM_AMD__)
    find_package(HIP)
    target_link_libraries(${example_name} PRIVATE hip::host)
endif()

install(TARGETS ${example_name})
