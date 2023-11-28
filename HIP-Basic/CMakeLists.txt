# MIT License
#
# Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(HIP-Basic LANGUAGES CXX)

# Only supported on HIP (not CUDA)
if(NOT "${GPU_RUNTIME}" STREQUAL "CUDA")
    add_subdirectory(assembly_to_executable)
    add_subdirectory(llvm_ir_to_executable)
    add_subdirectory(module_api)
endif()

if(WIN32 AND NOT "${GPU_RUNTIME}" STREQUAL "CUDA")
    add_subdirectory(cooperative_groups)
endif()

add_subdirectory(bandwidth)
add_subdirectory(bit_extract)
add_subdirectory(device_query)
add_subdirectory(device_globals)
add_subdirectory(dynamic_shared)
add_subdirectory(events)
add_subdirectory(gpu_arch)
add_subdirectory(hello_world)
if(NOT WIN32)
    find_package(Perl)

    if(Perl_FOUND)
        add_subdirectory(hipify)
    else()
        message("Perl not found, not building hipify example")
    endif()
endif()
add_subdirectory(inline_assembly)
add_subdirectory(matrix_multiplication)
add_subdirectory(moving_average)
add_subdirectory(multi_gpu_data_transfer)
add_subdirectory(occupancy)
add_subdirectory(runtime_compilation)
add_subdirectory(saxpy)
add_subdirectory(shared_memory)
# We cannot build the static library on Windows as the HIP SDK does not include CMAKE_AR.
if(NOT WIN32 AND NOT "${GPU_RUNTIME}" STREQUAL "HIP")
    add_subdirectory(static_host_library)
endif()
add_subdirectory(streams)
add_subdirectory(texture_management)
add_subdirectory(warp_shuffle)

find_package(glfw3)
find_package(Vulkan COMPONENTS glslangValidator)

if(NOT WIN32)
    if(NOT glfw3_FOUND)
        message("GLFW not found, not building OpenGL interop example")
    else()
        add_subdirectory(opengl_interop)
    endif()
endif()

if(NOT glfw3_FOUND)
    message("GLFW not found, not building Vulkan interop example")
elseif(NOT Vulkan_FOUND)
    message("Vulkan not found, not building Vulkan interop example")
else()
    add_subdirectory(vulkan_interop)
endif()
