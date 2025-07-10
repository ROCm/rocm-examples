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

set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
set(GPU_RUNTIMES "HIP" "CUDA")
set_property(CACHE GPU_RUNTIME PROPERTY STRINGS ${GPU_RUNTIMES})

if(NOT "${GPU_RUNTIME}" IN_LIST GPU_RUNTIMES)
    set(ERROR_MESSAGE
        "GPU_RUNTIME is set to \"${GPU_RUNTIME}\".\nGPU_RUNTIME must be either HIP or CUDA."
    )
    message(FATAL_ERROR ${ERROR_MESSAGE})
endif()

enable_language(${GPU_RUNTIME})
set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)

if(WIN32)
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

add_executable(${example_name} main.hip vulkan_utils.hip)

set(include_dirs "../../Common")
if(GPU_RUNTIME STREQUAL "CUDA")
    list(APPEND include_dirs "${ROCM_ROOT}/include")
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

set_source_files_properties(main.hip PROPERTIES LANGUAGE ${GPU_RUNTIME})
set_source_files_properties(vulkan_utils.hip PROPERTIES LANGUAGE ${GPU_RUNTIME})

install(TARGETS ${example_name})
