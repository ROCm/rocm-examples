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

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(HIP-Basic LANGUAGES CXX)
include(CTest)

include("${CMAKE_CURRENT_LIST_DIR}/../Common/FindFilesystem.cmake")

# ROCm installation path
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(ROCM_ROOT
        "$ENV{HIP_PATH}"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
    set(ROCM_LLVM_ROOT ${ROCM_ROOT})
    set(PREFIX_LLVM_ROOT ${CMAKE_INSTALL_PREFIX})
else()
    set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
    set(ROCM_LLVM_ROOT "${ROCM_ROOT}/llvm")
    set(PREFIX_LLVM_ROOT "${CMAKE_INSTALL_PREFIX}/llvm")
endif()

file(RELATIVE_PATH folder_bin ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${folder_bin})

# Only supported on HIP (not CUDA)
if(NOT "${GPU_RUNTIME}" STREQUAL "CUDA")
    # Make sure the dependencies can be found before building.
    find_program(
        LLVM_DIS_COMMAND
        llvm-dis
        PATH_SUFFIXES bin
        PATHS ${ROCM_LLVM_ROOT} ${PREFIX_LLVM_ROOT}
        NO_DEFAULT_PATH
    )
    find_program(
        OFFLOAD_BUNDLER_COMMAND
        clang-offload-bundler
        PATH_SUFFIXES bin
        PATHS ${ROCM_LLVM_ROOT} ${PREFIX_LLVM_ROOT}
        NO_DEFAULT_PATH
    )
    find_program(
        LLVM_MC_COMMAND
        llvm-mc
        PATH_SUFFIXES bin
        PATHS ${ROCM_LLVM_ROOT} ${PREFIX_LLVM_ROOT}
        NO_DEFAULT_PATH
    )

    if(LLVM_DIS_COMMAND AND OFFLOAD_BUNDLER_COMMAND AND LLVM_MC_COMMAND)
        add_subdirectory(llvm_ir_to_executable)
    else()
        message(
            "'llvm-dis', 'llvm-mc', or 'clang-offload-bundler' not found, not building assembly example."
        )
        if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
            message(
                "    These tools can be found in the 'rocm-llvm-dev' package."
            )
        endif()
    endif()

    if(OFFLOAD_BUNDLER_COMMAND AND LLVM_MC_COMMAND)
        add_subdirectory(assembly_to_executable)
    else()
        message(
            "'llvm-mc' or 'clang-offload-bundler' not found, not building LLVM IR example."
        )
        if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
            message(
                "    These tools can be found in the 'rocm-llvm-dev' package."
            )
        endif()
    endif()

    if(NOT CXX_FS_HEADER_FOUND)
        message("filesystem not found, not building module API example")
    else()
        add_subdirectory(module_api)
    endif()
endif()

if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
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
if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
    find_package(Perl)

    if(Perl_FOUND)
        add_subdirectory(hipify)
    else()
        message("Perl not found, not building hipify example")
    endif()
endif()
if("${GPU_RUNTIME}" STREQUAL "CUDA")
    add_subdirectory(hello_world_cuda)
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
if(
    NOT CMAKE_SYSTEM_NAME MATCHES "Windows"
    AND NOT "${GPU_RUNTIME}" STREQUAL "HIP"
)
    add_subdirectory(static_host_library)
endif()
add_subdirectory(streams)
add_subdirectory(texture_management)
add_subdirectory(warp_shuffle)

find_package(glfw3)
find_package(Vulkan COMPONENTS glslangValidator)

if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
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
elseif(NOT Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
    message("GlslangValidator not found, not building Vulkan interop example")
else()
    add_subdirectory(vulkan_interop)
    if(WIN32)
        add_subdirectory(vulkan_interop_mipmap)
    endif()
endif()
