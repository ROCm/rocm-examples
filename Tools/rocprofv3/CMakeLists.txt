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

set(example_name rocprofv3)

set(example_basic ${example_name}-basic)
set(example_basic_script ${example_basic}.sh)

set(example_advanced ${example_name}-advanced)
set(example_advanced_script ${example_advanced}.sh)

set(example_workload ${example_name}-matmul)
set(example_workload_roctx ${example_workload}-roctx)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../Common/HipPlatform.cmake")
select_gpu_language()
enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    message(STATUS "${example_name} does not support CUDA accelerators. Not building ${example_name} examples.")
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../Common/ROCmPath.cmake")

find_package(hip REQUIRED)
find_package(rocprofiler-sdk-roctx REQUIRED)

set_source_files_properties(main.hip PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

add_executable(${example_workload} main.hip)
set_target_properties(${example_workload} PROPERTIES $<$<COMPILE_LANGUAGE:CXX>:CXX_EXTENSIONS OFF>
                                                     $<$<COMPILE_LANGUAGE:HIP>:HIP_EXTENSIONS OFF>)
target_compile_features(${example_workload} PRIVATE cxx_std_17 hip_std_17)
target_include_directories(${example_workload} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../Common/
    ${CMAKE_CURRENT_LIST_DIR}/../../External/
)
target_link_libraries(${example_workload} PRIVATE hip::host)

add_executable(${example_workload_roctx} main.hip)
set_target_properties(${example_workload_roctx} PROPERTIES $<$<COMPILE_LANGUAGE:CXX>:CXX_EXTENSIONS OFF>
                                                             $<$<COMPILE_LANGUAGE:HIP>:HIP_EXTENSIONS OFF>)
target_compile_definitions(${example_workload_roctx} PRIVATE EXAMPLE_ROCTX)
target_compile_features(${example_workload_roctx} PRIVATE cxx_std_17 hip_std_17)
target_include_directories(${example_workload_roctx} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../Common/
    ${CMAKE_CURRENT_LIST_DIR}/../../External/
)
target_link_libraries(${example_workload_roctx} PRIVATE
    hip::host
    rocprofiler-sdk-roctx::rocprofiler-sdk-roctx
)

# We will work from the executable output directory
set(example_work_dir $<TARGET_FILE_DIR:${example_workload}>)

# Copy the scripts
add_custom_command(
    TARGET ${example_workload}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/${example_basic_script}" ${example_work_dir}
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/${example_advanced_script}" ${example_work_dir}
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/kernel_filter.yml" ${example_work_dir}
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/wavefront_stats.yml" ${example_work_dir}
    BYPRODUCTS ${example_basic_script} ${example_advanced_script} kernel_filter.yml wavefront_stats.yml
    WORKING_DIRECTORY ${example_work_dir}
    COMMENT "Copying ${example_name} scripts to ${example_work_dir}"
)

# Make examples runnable using ctest
add_test(NAME ${example_basic} COMMAND ${example_basic_script} WORKING_DIRECTORY ${example_work_dir})
add_test(NAME ${example_advanced} COMMAND ${example_advanced_script} WORKING_DIRECTORY ${example_work_dir})

install(TARGETS ${example_workload} ${example_workload_roctx})
install(PROGRAMS ${example_basic_script} ${example_advanced_script} TYPE BIN)
install(FILES kernel_filter.yml wavefront_stats.yml TYPE BIN)
