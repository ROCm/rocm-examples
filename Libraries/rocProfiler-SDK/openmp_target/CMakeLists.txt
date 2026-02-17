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

set(example_name rocprofiler-sdk_openmp_target)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

if(GPU_RUNTIME STREQUAL "CUDA")
    message(STATUS "rocProfiler-SDK examples do not support the CUDA runtime")
    return()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "rocProfiler-SDK examples are only available on Linux")
    return()
else()
    set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
endif()

project(${example_name} LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../Common/FindFilesystem.cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

find_package(rocprofiler-sdk REQUIRED)
find_package(rocprofiler-sdk-roctx REQUIRED)
find_package(OpenMP COMPONENTS CXX REQUIRED)
find_package(Threads REQUIRED)

# Find the rocprofiler launcher
find_program(
    ROCPROFILER_SDK_COMPILE_LAUNCHER
    NAMES rocprofiler-sdk-launch-compiler
    HINTS ${ROCM_ROOT}
    PATHS ${ROCM_ROOT}
    PATH_SUFFIXES libexec/rocprofiler-sdk
    REQUIRED
)

# Find amdclang++
find_program(
    AMDCLANG_COMPILER
    NAMES amdclang++
    HINTS ${ROCM_ROOT}
    PATHS ${ROCM_ROOT}
    PATH_SUFFIXES bin llvm/bin
    REQUIRED
)

# Default GPU targets
set(OPENMP_GPU_TARGETS "gfx906" "gfx908" "gfx90a" "gfx942" "gfx1100"
    CACHE STRING "GPU targets to compile for")

# Create client library
add_library(${example_name}_client SHARED client.cpp)

target_link_libraries(${example_name}_client PRIVATE
    rocprofiler-sdk::rocprofiler-sdk
    ${CXX_FS_LIBRARY}
)

target_include_directories(${example_name}_client PRIVATE
    ../../../Common
)

target_compile_options(${example_name}_client PRIVATE
    -D__HIP_PLATFORM_AMD__
)

# Create executable
add_executable(${example_name} main.cpp)

target_link_libraries(${example_name} PRIVATE
    Threads::Threads
    rocprofiler-sdk-roctx::rocprofiler-sdk-roctx
    ${example_name}_client
    OpenMP::OpenMP_CXX
    ${CXX_FS_LIBRARY}
)

target_include_directories(${example_name} PRIVATE
    ../../../Common
    ../../../External
)

# Add compile options
target_compile_options(${example_name} PRIVATE
    -D__HIP_PLATFORM_AMD__
)

# Add link options - do not remove the -fopenmp flag. It is still required
# despite us linking to OpenMP::OpenMP_CXX above because we're using the
# rocProfiler-SDK compiler wrapper and CMake fails to add the necessary OpenMP
# flags to the link step.
target_link_options(${example_name} PRIVATE -fopenmp -Wl,-rpath,${ROCM_ROOT}/lib)

# Add offload architecture flags for each GPU target
foreach(_TARGET ${OPENMP_GPU_TARGETS})
    target_compile_options(${example_name} PRIVATE --offload-arch=${_TARGET})
    target_link_options(${example_name} PRIVATE --offload-arch=${_TARGET})
endforeach()

# Use the rocprofiler launcher with amdclang++ for this target only
set_property(
    TARGET ${example_name}
    PROPERTY RULE_LAUNCH_COMPILE
    "${ROCPROFILER_SDK_COMPILE_LAUNCHER} ${AMDCLANG_COMPILER} ${CMAKE_CXX_COMPILER}"
)
set_property(
    TARGET ${example_name}
    PROPERTY RULE_LAUNCH_LINK
    "${ROCPROFILER_SDK_COMPILE_LAUNCHER} ${AMDCLANG_COMPILER} ${CMAKE_CXX_COMPILER}"
)

# Make example runnable using ctest
add_test(NAME ${example_name} COMMAND ${example_name})

install(TARGETS ${example_name})
