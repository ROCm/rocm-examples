# MIT License
#
# Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

set(example_name hip_module_api)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../Common/FindFilesystem.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../../Common/HipPlatform.cmake")
select_gpu_language()

enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD 17)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_EXTENSIONS OFF)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD_REQUIRED ON)
select_hip_platform()
verify_hip_platform(PLATFORMS "amd")

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

add_executable(${example_name} main.hip)

# This is a mild hack. CMake (like most build systems) don't play well with targeting multiple
# architectures in the same build. This example builds an executable for the host system and a
# device code object for potentially multiple device architectures. Code objects need not and
# _should not_ be linked using the host linker. We're creating an OBJECT library, because in
# CMake terms, OBJECT libraries don't involve a linking step, but in reality, we'll also add
# the `--cuda-device-only` compile option, which changes the target triplet of the compiler
# from host to device code.
add_library(${example_name}_module OBJECT module.hip)

# Ideally we'd need a non-existent CheckHIPCompilerFlag module before assuming this flag works.
target_compile_options(${example_name}_module PRIVATE --cuda-device-only)

# Copy the code object next to the executable
add_custom_command(
    TARGET ${example_name}
    POST_BUILD
    COMMAND "${CMAKE_COMMAND}"
    ARGS
        -E copy $<TARGET_OBJECTS:${example_name}_module>
        $<TARGET_FILE_DIR:${example_name}>/module.co
    COMMAND_EXPAND_LISTS
    COMMENT "Copying HIP code object for ${example_name}"
)

add_dependencies(${example_name} ${example_name}_module)
# Make example runnable using ctest
add_test(NAME ${example_name} COMMAND ${example_name})

set(include_dirs "../../Common" "../../External")
target_include_directories(${example_name} PRIVATE ${include_dirs})
target_link_libraries(${example_name} PRIVATE ${CXX_FS_LIBRARY})
set_source_files_properties(main.hip PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

install(TARGETS ${example_name})
