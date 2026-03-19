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

set(example_name hip_vulkan_interop_mipmap)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../Common/HipPlatform.cmake")
select_gpu_language()
enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()

set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD 17)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_EXTENSIONS OFF)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD_REQUIRED ON)

if(WIN32)
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

add_executable(${example_name} main.hip vulkan_utils.hip)

set(include_dirs "../../Common")
if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    list(APPEND include_dirs "${ROCM_PATH}/include")
endif()

find_package(Vulkan REQUIRED COMPONENTS glslangValidator)
find_package(glfw3 REQUIRED)

set(SHADER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

add_custom_command(
    OUTPUT "${SHADER_BINARY_DIR}/texture.vert.spv.h"
    COMMAND
        ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V100 --vn texture_vert -o
        "${SHADER_BINARY_DIR}/texture.vert.spv.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/texture.vert"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/texture.vert"
    COMMENT "Compiling vertex shader"
)

add_custom_command(
    OUTPUT "${SHADER_BINARY_DIR}/texture.frag.spv.h"
    COMMAND
        ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V100 --vn texture_frag -o
        "${SHADER_BINARY_DIR}/texture.frag.spv.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/texture.frag"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/texture.frag"
    COMMENT "Compiling fragment shader"
)

add_custom_target(
    texture_shaders
    DEPENDS
        "${SHADER_BINARY_DIR}/texture.vert.spv.h"
        "${SHADER_BINARY_DIR}/texture.frag.spv.h"
)

list(APPEND include_dirs ${SHADER_BINARY_DIR})
add_dependencies(${example_name} texture_shaders)

target_link_libraries(${example_name} PRIVATE Vulkan::Headers)
target_link_libraries(${example_name} PRIVATE glfw)
target_include_directories(${example_name} PRIVATE ${include_dirs})

set_source_files_properties(main.hip PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})
set_source_files_properties(vulkan_utils.hip PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

install(TARGETS ${example_name})
