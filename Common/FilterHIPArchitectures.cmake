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

cmake_minimum_required(VERSION 3.21)

include_guard(GLOBAL)

# Filters CMAKE_HIP_ARCHITECTURES to only include supported architectures
#
# This function filters the CMAKE_HIP_ARCHITECTURES list to remove unsupported
# architectures for a specific example. If any architectures are removed, a
# WARNING message is printed. If no supported architectures remain, the function
# sets a variable indicating that the example should be skipped.
#
# Parameters:
#   EXAMPLE_NAME - The name of the example (for messages)
#   SUPPORTED_ARCHS - List of supported GPU architectures (e.g., gfx908 gfx90a gfx942)
#   SHOULD_SKIP_VAR - Output variable name that will be set to TRUE if no supported
#                     architectures remain, FALSE otherwise
#
# Example usage:
#   filter_hip_architectures("MyExample" "gfx908;gfx90a;gfx942" SHOULD_SKIP)
#   if(SHOULD_SKIP)
#       return()
#   endif()
#
function(filter_hip_architectures EXAMPLE_NAME SUPPORTED_ARCHS SHOULD_SKIP_VAR)
    set(FILTERED_HIP_ARCHITECTURES)
    set(REMOVED_ARCHITECTURES)

    foreach(ARCH ${CMAKE_HIP_ARCHITECTURES})
        if(ARCH IN_LIST SUPPORTED_ARCHS)
            list(APPEND FILTERED_HIP_ARCHITECTURES ${ARCH})
        else()
            list(APPEND REMOVED_ARCHITECTURES ${ARCH})
        endif()
    endforeach()

    if(REMOVED_ARCHITECTURES)
        message(WARNING "${EXAMPLE_NAME}: Removing unsupported architectures: ${REMOVED_ARCHITECTURES}")
    endif()

    if(NOT FILTERED_HIP_ARCHITECTURES)
        message(STATUS "${EXAMPLE_NAME}: No supported architectures found. Not building ${EXAMPLE_NAME}")
        set(${SHOULD_SKIP_VAR} TRUE PARENT_SCOPE)
    else()
        # Override CMAKE_HIP_ARCHITECTURES for this directory scope
        set(CMAKE_HIP_ARCHITECTURES ${FILTERED_HIP_ARCHITECTURES} PARENT_SCOPE)
        set(${SHOULD_SKIP_VAR} FALSE PARENT_SCOPE)
    endif()
endfunction()
