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

set(example_name rocwmma_simple_dgemv)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX HIP)

include("${CMAKE_CURRENT_LIST_DIR}/../../../Common/FilterHIPArchitectures.cmake")
filter_hip_architectures("rocWMMA simple_dgemv example" "gfx908;gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201" SHOULD_SKIP)
if(SHOULD_SKIP)
    return()
endif()

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    message(STATUS "rocWMMA examples do not support the CUDA runtime")
    return()
endif()

set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

include("${CMAKE_CURRENT_LIST_DIR}/../../../Common/ROCmPath.cmake")

find_package(hip REQUIRED)
# Try to find rocWMMA as a package first
find_package(rocwmma QUIET)

# Remove unnecessary OpenMP dependency from rocwmma
if(TARGET roc::rocwmma)
    get_target_property(ROCWMMA_LINK_LIBS roc::rocwmma INTERFACE_LINK_LIBRARIES)
    if(ROCWMMA_LINK_LIBS)
        list(REMOVE_ITEM ROCWMMA_LINK_LIBS OpenMP::OpenMP_CXX)
        list(REMOVE_ITEM ROCWMMA_LINK_LIBS OpenMP::omp)
        set_target_properties(roc::rocwmma PROPERTIES
            INTERFACE_LINK_LIBRARIES "${ROCWMMA_LINK_LIBS}"
        )
    endif()
    set(ROCWMMA_TARGET roc::rocwmma)
elseif(TARGET rocwmma)
    get_target_property(ROCWMMA_LINK_LIBS rocwmma INTERFACE_LINK_LIBRARIES)
    if(ROCWMMA_LINK_LIBS)
        list(REMOVE_ITEM ROCWMMA_LINK_LIBS OpenMP::OpenMP_CXX)
        list(REMOVE_ITEM ROCWMMA_LINK_LIBS OpenMP::omp)
        set_target_properties(rocwmma PROPERTIES
            INTERFACE_LINK_LIBRARIES "${ROCWMMA_LINK_LIBS}"
        )
    endif()
    set(ROCWMMA_TARGET rocwmma)
endif()

if(NOT ROCWMMA_TARGET AND NOT rocwmma_FOUND)
    # If package not found, try to find headers manually
    find_path(ROCWMMA_INCLUDE_DIR
        NAMES rocwmma/rocwmma.hpp
        PATHS ${ROCM_PATH}/include
        DOC "rocWMMA include directory"
    )

    if(NOT ROCWMMA_INCLUDE_DIR)
        message(FATAL_ERROR "rocWMMA headers not found. Please install rocWMMA or set ROCWMMA_INCLUDE_DIR")
    endif()

    # Create imported target for header-only library
    add_library(rocwmma INTERFACE)
    target_include_directories(rocwmma INTERFACE ${ROCWMMA_INCLUDE_DIR})
    set(ROCWMMA_TARGET rocwmma)
endif()

add_executable(${example_name} main.hip)
# Make example runnable using ctest
add_test(NAME ${example_name} COMMAND ${example_name})

# Link to example library
target_link_libraries(${example_name} PRIVATE ${ROCWMMA_TARGET})

target_include_directories(
    ${example_name}
    PRIVATE "../../../Common" "../../../External"
)

install(TARGETS ${example_name})
