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

set(example_name rocal_video)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    message(STATUS "rocAL examples do not support the CUDA runtime")
    return()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "rocAL examples are only available on Linux")
    return()
else()
    set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

find_library(ROCAL_LIBRARY
    NAMES rocal
    PATHS"${ROCM_ROOT}/lib"
    PATH_SUFFIXES lib lib64
    REQUIRED
)

find_package(OpenCV REQUIRED)

add_executable(${example_name} main.cpp)
# Make example runnable using ctest
add_test(NAME ${example_name} COMMAND ${example_name})

# Link to rocAL library
target_link_libraries(${example_name} PRIVATE ${ROCAL_LIBRARY} ${OpenCV_LIBS})

# Include directories
target_include_directories(
    ${example_name}
    PRIVATE "../../../External" ${ROCM_ROOT}/include ${ROCM_ROOT}/include/rocal ${OpenCV_INCLUDE_DIRS}
)

# Set example data directory as compile definition
target_compile_definitions(
    ${example_name}
    PRIVATE EXAMPLE_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data"
)

install(TARGETS ${example_name})
