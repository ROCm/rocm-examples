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

set(example_name hiptensor_contraction_bilinear_f32_bf16)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../Common/HipPlatform.cmake")
select_gpu_language()

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    message(FATAL_ERROR "hipTensor examples do not support the CUDA runtime.")
endif()

enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../Common/ROCmPath.cmake")

find_package(hiptensor REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../Common/FilterHIPArchitectures.cmake")
filter_hip_architectures("hipTensor" "gfx908;gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx11-generic;gfx1200;gfx1201;gfx1250;gfx12-generic" SHOULD_SKIP)
if(SHOULD_SKIP)
    return()
endif()

set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD 17)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_EXTENSIONS OFF)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD_REQUIRED ON)
select_hip_platform()

add_executable(${example_name} main.cpp)
# Make example runnable using ctest
add_test(NAME ${example_name} COMMAND ${example_name})

# Link to example library
target_link_libraries(${example_name} PRIVATE hiptensor)

target_include_directories(
    ${example_name}
    PRIVATE "../../../../../Common" "../../../../../External"
)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

install(TARGETS ${example_name})

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    install(IMPORTED_RUNTIME_ARTIFACTS hiptensor)
endif()
