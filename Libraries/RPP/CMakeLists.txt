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
project(RPP_examples LANGUAGES CXX)
include(CTest)

file(RELATIVE_PATH folder_bin ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${folder_bin})

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "RPP examples are only available on Linux")
    return()
else()
    set(ROCM_PATH
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

# Try to find RPP using find_package first
find_package(rpp QUIET)

# If not found via find_package, manually locate the library
if(NOT rpp_FOUND)
    message(STATUS "RPP config not found, attempting to locate library manually")

    # Search for the RPP library
    find_library(RPP_LIBRARY
        NAMES rpp librpp
        PATHS ${ROCM_PATH}/lib
        DOC "Path to RPP library"
    )

    # Search for the RPP header
    find_path(RPP_INCLUDE_DIR
        NAMES rpp.h rppi.h
        PATHS
            ${ROCM_PATH}/include
            ${ROCM_PATH}/include/rpp
        DOC "Path to RPP include directory"
    )

    # Check if both library and header were found
    if(RPP_LIBRARY AND RPP_INCLUDE_DIR)
        message(STATUS "Found RPP library: ${RPP_LIBRARY}")
        message(STATUS "Found RPP include: ${RPP_INCLUDE_DIR}")

        # Create an imported target for RPP
        add_library(rpp SHARED IMPORTED)
        set_target_properties(rpp PROPERTIES
            IMPORTED_LOCATION "${RPP_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${RPP_INCLUDE_DIR}"
        )

        set(rpp_FOUND TRUE)
    else()
        message(WARNING "RPP could not be found, not building RPP examples")
        if(NOT RPP_LIBRARY)
            message(STATUS "  - RPP library not found")
        endif()
        if(NOT RPP_INCLUDE_DIR)
            message(STATUS "  - RPP headers not found")
        endif()
        return()
    endif()
endif()

add_subdirectory(box_filter)
add_subdirectory(brightness)
add_subdirectory(contrast)
add_subdirectory(flip)
add_subdirectory(gamma_correction)
add_subdirectory(resize)
