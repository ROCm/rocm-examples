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

set(example_name hip_hipify)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(
        FATAL_ERROR
        "The hipify example currently does not support Windows."
    )
endif()

find_package(Perl REQUIRED)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../Common/HipPlatform.cmake")
select_gpu_language()
enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()

set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD 17)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_EXTENSIONS OFF)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD_REQUIRED ON)

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

# find hipify-perl
if(EXISTS "${ROCM_PATH}/libexec/hipify/hipify-perl")
    set(HIPIFY_PERL ${ROCM_PATH}/libexec/hipify/hipify-perl)
else()
    set(HIPIFY_PERL ${ROCM_PATH}/bin/hipify-perl)
endif()

# create main.hip
add_custom_command(
    OUTPUT main.hip
    COMMAND
        ${PERL_EXECUTABLE} ${HIPIFY_PERL} ${CMAKE_CURRENT_SOURCE_DIR}/main.cu -o
        ${CMAKE_CURRENT_BINARY_DIR}/main.hip
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
    VERBATIM
)

add_executable(${example_name} ${CMAKE_CURRENT_BINARY_DIR}/main.hip)
# Make example runnable using ctest
add_test(NAME ${example_name} COMMAND ${example_name})

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    set(include_dirs "${ROCM_PATH}/include")
endif()

target_include_directories(${example_name} PRIVATE ${include_dirs})
set_source_files_properties(main.hip PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

install(TARGETS ${example_name})
