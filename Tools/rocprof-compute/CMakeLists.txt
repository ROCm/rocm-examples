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

set(example_name rocprof-compute)

set(example_basic ${example_name}_basic)
set(example_basic_script ${example_basic}.sh)

set(example_advanced ${example_name}_advanced)
set(example_advanced_script ${example_advanced}.sh)

set(example_occupancy ${example_name}_occupancy)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

set(ROCPROF_COMPUTE_SUPPORTED_ARCH gfx908 gfx90a gfx940 gfx941 gfx942 gfx950)

foreach(ARCH ${CMAKE_HIP_ARCHITECTURES})
    if(NOT ARCH IN_LIST ROCPROF_COMPUTE_SUPPORTED_ARCH)
        message(STATUS "rocprof-compute does not support architecture ${ARCH}. Not building rocprof-compute examples")
        return()
    endif()
endforeach()

include("${CMAKE_CURRENT_LIST_DIR}/../../Common/HipPlatform.cmake")
select_gpu_language()
enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    message(STATUS "rocprof-compute does not support CUDA accelerators. Not building rocprof-compute examples.")
    return()
endif()

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

find_package(hip REQUIRED)

# This example requires a Python environment with the dependencies installed from /opt/rocm/libexec/rocprofiler-compute/requirements.txt
find_package(Python3 3.3 REQUIRED COMPONENTS Interpreter)

add_executable(${example_occupancy} occupancy.hip)
set_source_files_properties(occupancy.hip PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})
set_target_properties(${example_occupancy} PROPERTIES $<$<COMPILE_LANGUAGE:CXX>:CXX_EXTENSIONS OFF>
                                                     $<$<COMPILE_LANGUAGE:HIP>:HIP_EXTENSIONS OFF>)
target_compile_features(${example_occupancy} PRIVATE cxx_std_17 hip_std_17)
target_link_libraries(${example_occupancy} PRIVATE hip::host)

# We will work from the executable output directory
set(example_work_dir $<TARGET_FILE_DIR:${example_occupancy}>)

# Copy the scripts
add_custom_command(
    TARGET ${example_occupancy}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/${example_basic_script}" ${example_work_dir}
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/${example_advanced_script}" ${example_work_dir}
    BYPRODUCTS ${example_basic_script} ${example_advanced_script}
    WORKING_DIRECTORY ${example_work_dir}
    COMMENT "Copying ${example_name} scripts to ${example_work_dir}"
)

# Make examples runnable using ctest
add_test(NAME ${example_basic} COMMAND ${example_basic_script} WORKING_DIRECTORY ${example_work_dir})
add_test(NAME ${example_advanced} COMMAND ${example_advanced_script} WORKING_DIRECTORY ${example_work_dir})

install(TARGETS ${example_occupancy})
install(PROGRAMS ${example_basic_script} ${example_advanced_script} TYPE BIN)
