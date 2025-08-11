# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

set(example_name hip_hello_world_cuda)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

include("${CMAKE_CURRENT_LIST_DIR}/../../Common/HipPlatform.cmake")
select_gpu_language()

project(${example_name} LANGUAGES ${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()
verify_hip_platform(PLATFORMS "nvidia")

set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD 17)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_EXTENSIONS OFF)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD_REQUIRED ON)

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

add_executable(${example_name} main.hip)

# Make example runnable using ctest
add_test(NAME ${example_name} COMMAND ${example_name})

# Make the HIP runtime headers accessible
target_include_directories(
    ${example_name}
    PRIVATE "${ROCM_ROOT}/include" "${CMAKE_CURRENT_SOURCE_DIR}/../../Common"
)

# Set up the compilation language for the source file.
# Usually this can be deduced from the file extension, but not in the case of .hip.
set_source_files_properties(main.hip PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

install(TARGETS ${example_name})
