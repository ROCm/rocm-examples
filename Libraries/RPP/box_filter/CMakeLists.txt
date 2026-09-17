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

set(example_name rpp_box_filter)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

include("../../../Common/HipPlatform.cmake")
select_gpu_language()

enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD 17)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_EXTENSIONS OFF)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD_REQUIRED ON)
select_hip_platform()
verify_hip_platform(PLATFORMS "amd")

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "RPP examples are only available on Linux")
    return()
endif()

include("../../../Common/ROCmPath.cmake")

find_package(OpenCV REQUIRED)

# Locate libomp.so with helper
include("../../../Common/FindOpenMPHelpers.cmake")
find_omp_library(OMP_LIBRARY)

add_executable(${example_name} main.cpp)
add_test(NAME ${example_name} COMMAND ${example_name})

target_link_libraries(${example_name} PRIVATE
    -lrpp
    ${OpenCV_LIBS}
    ${OMP_LIBRARY}
)

target_compile_options(${example_name} PRIVATE -fopenmp)
target_link_options(${example_name} PRIVATE -fopenmp)

target_compile_definitions(
    ${example_name}
    PRIVATE EXAMPLE_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/images" GPU_SUPPORT=1 RPP_BACKEND_HIP=1
)

target_include_directories(
    ${example_name}
    PRIVATE "../../../Common" "../../../External" ${OpenCV_INCLUDE_DIRS} "${ROCM_PATH}/include/rpp"
)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

install(TARGETS ${example_name})
