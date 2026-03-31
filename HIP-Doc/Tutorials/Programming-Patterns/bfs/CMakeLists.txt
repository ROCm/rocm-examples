# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

set(example_name hip_bfs)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/HipPlatform.cmake")
select_gpu_language()
enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    # The example uses CUDA-compatible HIP runtime functions
    find_package(CUDAToolkit REQUIRED)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/ROCmPath.cmake")

# Build the main BFS application
add_executable(${example_name} main.hip)
set_source_files_properties(main.hip PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

set_target_properties(${example_name} PROPERTIES 
    $<$<COMPILE_LANGUAGE:CUDA>:CUDA_EXTENSIONS OFF>
    $<$<COMPILE_LANGUAGE:HIP>:HIP_EXTENSIONS OFF>
)
target_compile_features(${example_name} PRIVATE cuda_std_17 hip_std_17)
target_include_directories(${example_name} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<$<COMPILE_LANGUAGE:CUDA>:${ROCM_PATH}/include>
)
target_link_libraries(${example_name} PRIVATE $<$<LINK_LANGUAGE:CUDA>:CUDA::cuda_driver>)

# Build the graph generator utility
add_executable(graphgen graphgen.cpp)
target_compile_features(graphgen PRIVATE cxx_std_17)

set(example_work_dir $<TARGET_FILE_DIR:${example_name}>)

# Copy test image to executable output directory
add_custom_command(TARGET ${example_name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/graph4096.txt ${example_work_dir}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/graph65536.txt ${example_work_dir}
    WORKING_DIRECTORY ${example_work_dir}
    COMMENT "Copying test graphs to ${example_work_dir}"
)

# Make example runnable using ctest
add_test(NAME ${example_name} COMMAND ${example_name} graph4096.txt WORKING_DIRECTORY ${example_work_dir})

install(TARGETS ${example_name} graphgen)
install(FILES graph4096.txt graph65536.txt DESTINATION ${CMAKE_INSTALL_PREFIX})
