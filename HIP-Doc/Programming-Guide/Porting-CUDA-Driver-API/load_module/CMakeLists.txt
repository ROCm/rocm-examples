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

set(example_name hip_load_module)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/HipPlatform.cmake")
select_gpu_language()
enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()

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

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "HIP")
    set(USED_LANGUAGE "CXX")
    find_package(hip REQUIRED)
else()
    set(USED_LANGUAGE "CUDA")
    find_package(CUDAToolkit REQUIRED) # We must explicitly link the CUDA driver library.
endif()

set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${USED_LANGUAGE})
set_source_files_properties(vcpy.hip PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

set(example_module_prefix "${example_name}_")
set(example_module_suffix "vcpy_isa")
set(example_module "${example_module_prefix}${example_module_suffix}")
add_library(${example_module} OBJECT vcpy.hip)
target_compile_features(${example_module} PRIVATE cuda_std_17 hip_std_17)
target_compile_options(${example_module} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--cuda-device-only>)
target_include_directories(${example_module} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${ROCM_ROOT}/include>)
if(${ROCM_EXAMPLES_GPU_LANGUAGE} STREQUAL "CUDA")
    set_property(TARGET ${example_module} PROPERTY CUDA_PTX_COMPILATION ON)
endif()

add_executable(${example_name} main.cpp)
# Make example runnable using ctest
add_test(NAME ${example_name} COMMAND ${example_name} WORKING_DIRECTORY $<TARGET_FILE_DIR:${example_name}>)

add_dependencies(${example_name} ${example_module})
set_target_properties(${example_name} PROPERTIES $<$<COMPILE_LANGUAGE:CUDA>:CUDA_EXTENSIONS OFF>
                                                 $<$<COMPILE_LANGUAGE:CXX>:CXX_EXTENSIONS OFF>)
target_compile_features(${example_name} PRIVATE cuda_std_17 cxx_std_17)
target_include_directories(${example_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${ROCM_ROOT}/include>)
target_link_libraries(${example_name} PRIVATE $<$<LINK_LANGUAGE:CUDA>:CUDA::cuda_driver>)
if(USED_LANGUAGE STREQUAL "CXX")
    target_link_libraries(${example_name} PRIVATE hip::host)
endif()

# Copy object file to executable output directory
if(${ROCM_EXAMPLES_GPU_LANGUAGE} STREQUAL "HIP")
    set(example_module_hsaco "$<TARGET_FILE_DIR:${example_name}>/${example_module_suffix}.hsaco")
    message(STATUS "Target path: ${example_module_hsaco}")
    add_custom_command(TARGET ${example_name} POST_BUILD
                       COMMAND ${CMAKE_COMMAND}
                       ARGS -E copy $<TARGET_OBJECTS:${example_module}> ${example_module_hsaco}
                       COMMAND_EXPAND_LISTS
                       COMMENT "Copying ${example_module_hsaco}."
    )
    install(FILES ${example_module_hsaco} TYPE BIN)
endif()

if(${ROCM_EXAMPLES_GPU_LANGUAGE} STREQUAL "CUDA")
    set(example_module_ptx "$<TARGET_FILE_DIR:${example_name}>/${example_module_suffix}.ptx")
    add_custom_command(TARGET ${example_name} POST_BUILD
                       COMMAND ${CMAKE_COMMAND}
                       ARGS -E copy $<TARGET_OBJECTS:${example_module}> ${example_module_ptx}
                       COMMAND_EXPAND_LISTS
                       COMMENT "Copying ${example_module_ptx}.")
    install(FILES ${example_module_ptx} TYPE BIN)
endif()

install(TARGETS ${example_name})
