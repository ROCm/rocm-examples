# MIT License
#
# Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")

# Only supported on HIP (not CUDA)
if(NOT "${GPU_RUNTIME}" STREQUAL "HIP")
    set(ERROR_MESSAGE "GPU_RUNTIME is set to \"${GPU_RUNTIME}\".\nGPU_RUNTIME must be HIP.")
    message(FATAL_ERROR ${ERROR_MESSAGE})
endif()

enable_language(${GPU_RUNTIME})
set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)

set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
if(NOT CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH "${ROCM_ROOT}")
endif()

set(offload_archs ${CMAKE_HIP_ARCHITECTURES})
list(TRANSFORM offload_archs PREPEND "--offload-arch=")

set(module ${CMAKE_CURRENT_BINARY_DIR}/module.co)
set(module_sources ${CMAKE_CURRENT_SOURCE_DIR}/module.hip)
if(CMAKE_BUILD_TYPE EQUAL "Debug")
    set(module_flags ${CMAKE_HIP_FLAGS} ${CMAKE_HIP_FLAGS_DEBUG})
elseif(CMAKE_BUILD_TYPE EQUAL "Release")
    set(module_flags ${CMAKE_HIP_FLAGS} ${CMAKE_HIP_FLAGS_RELEASE})
elseif(CMAKE_BUILD_TYPE EQUAL "MinSizeRel")
    set(module_flags ${CMAKE_HIP_FLAGS} ${CMAKE_HIP_FLAGS_MINSIZEREL})
elseif(CMAKE_BUILD_TYPE EQUAL "RelWithDebInfo")
    set(module_flags ${CMAKE_HIP_FLAGS} ${CMAKE_HIP_FLAGS_RELWITHDEBINFO})
endif()

add_custom_command(
    OUTPUT ${module}
    COMMAND ${CMAKE_HIP_COMPILER} ${module_flags} ${module_sources} ${offload_archs} --cuda-device-only -o ${module}
    DEPENDS ${module_sources}
    COMMENT "Compiling HIP code object module.co"
)

add_custom_target(module ALL DEPENDS ${module})

add_executable(${example_name} main.hip)
# Make example runnable using ctest
add_test(${example_name} ${example_name})

set(include_dirs "../../Common")
target_include_directories(${example_name} PRIVATE ${include_dirs})
set_source_files_properties(main.hip PROPERTIES LANGUAGE ${GPU_RUNTIME})
