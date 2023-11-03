# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.21...3.27)

include_guard(GLOBAL)

set(ROCM_EXAMPLES_WERROR_DEV OFF CACHE BOOL "Treat CMake developer warnings as errors.")

function(select_gpu_language)
    if (DEFINED CACHE{GPU_RUNTIME} AND NOT "${ROCM_EXAMPLES_GPU_RUNTIME_DISABLE_WARN}")
        message(DEPRECATION "GPU_RUNTIME is deprecated, please use ROCM_EXAMPLES_GPU_LANGUAGE to set"
                            " the GPU language to be used.")
        set(ROCM_EXAMPLES_GPU_RUNTIME_DISABLE_WARN ON CACHE INTERNAL "")
        set(ROCM_EXAMPLES_GPU_LANGUAGE "$CACHE{GPU_RUNTIME}" CACHE STRING "Switches between HIP and CUDA" FORCE)
    else()
        set(ROCM_EXAMPLES_GPU_LANGUAGE "HIP" CACHE STRING "Switches between HIP and CUDA")
    endif()

    # For backward compatibility with examples not yet updated
    set(GPU_RUNTIME ${ROCM_EXAMPLES_GPU_LANGUAGE} PARENT_SCOPE)

    set(GPU_LANGUAGES "HIP" "CUDA")
    set_property(CACHE ROCM_EXAMPLES_GPU_LANGUAGE PROPERTY STRINGS "${GPU_LANGUAGES}")
endfunction()

# Must be called after select_gpu_language() and enabling that language
# (`enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})`)
function(select_hip_platform)
    if (NOT DEFINED ROCM_EXAMPLES_GPU_LANGUAGE)
        message(FATAL_ERROR "ROCM_EXAMPLES_GPU_LANGUAGE is not defined, select_gpu_language should"
                            " be called before select_hip_platform")
    endif()

    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    if (NOT "${ROCM_EXAMPLES_GPU_LANGUAGE}" IN_LIST LANGUAGES)
        message(FATAL_ERROR "ROCM_EXAMPLES_GPU_LANGUAGE (${ROCM_EXAMPLES_GPU_LANGUAGE}) must be"
                            " enabled before calling select_hip_platform")
    endif()

    # Newest CMake with HIP support for NVIDIA platform
    if("${ROCM_EXAMPLES_GPU_LANGUAGE}" STREQUAL "HIP" AND DEFINED CMAKE_HIP_PLATFORM)
        set(ROCM_EXAMPLES_HIP_PLATFORM "${CMAKE_HIP_PLATFORM}" CACHE INTERNAL "")
        if("${CMAKE_HIP_PLATFORM}" STREQUAL "nvidia" AND NOT "${ROCM_EXAMPLES_NV_PLATFORM_DISABLE_WARN}")
            message(WARNING "CMake HIP language targeting the NVIDIA platform detected,"
                            " this is an experimental feature with incomplete support.\n"
                            "If you encounter any errors please revert to the CUDA Language by "
                            " setting ROCM_EXAMPLES_GPU_LANGUAGE=CUDA.")
            set(ROCM_EXAMPLES_NV_PLATFORM_DISABLE_WARN ON CACHE INTERNAL "")
        endif()
        return()
    endif()

    # Legacy support for older cmake using the CUDA language on the NVIDIA platform
    if("${ROCM_EXAMPLES_GPU_LANGUAGE}" STREQUAL "HIP")
        set(ROCM_EXAMPLES_HIP_PLATFORM "amd" CACHE INTERNAL "")
    elseif("${ROCM_EXAMPLES_GPU_LANGUAGE}" STREQUAL "CUDA")
        set(ROCM_EXAMPLES_HIP_PLATFORM "nvidia" CACHE INTERNAL "")
    else()
        message(FATAL_ERROR "ROCM_EXAMPLES_GPU_LANGUAGE is set to:\n"
                            " ${ROCM_EXAMPLES_GPU_LANGUAGE}\n"
                            "ROCM_EXAMPLES_GPU_LANGUAGE must be either HIP or CUDA.")
    endif()
endfunction()

function(verify_hip_platform)
    cmake_parse_arguments(PARSE_ARGV 0 "PAR" "" "" "PLATFORMS")

    if(NOT "${ROCM_EXAMPLES_HIP_PLATFORM}" IN_LIST PAR_PLATFORMS)
        string(REPLACE ";" "\n " PAR_PLATFORMS "${PAR_PLATFORMS}")
        message(FATAL_ERROR "The ${PROJECT_NAME} example(s) don't support the HIP platform:\n "
                            " ${ROCM_EXAMPLES_HIP_PLATFORM}\n"
                            "the supported HIP platforms are:\n"
                            " ${PAR_PLATFORMS}\n")
    endif()
endfunction()

function(warn_gpu_runtime_used VARIABLE ACCESS VALUE CURRENT_LIST_FILE STACK)
    if("${CMAKE_CURRENT_FUNCTION_LIST_FILE}" IN_LIST STACK)
        return()
    endif()

    set(MESSAGE_TYPE "AUTHOR_WARNING")
    if("${ROCM_EXAMPLES_WERROR_DEV}")
        set(MESSAGE_TYPE "SEND_ERROR")
    endif()

    message(${MESSAGE_TYPE} "${VARIABLE} is deprecated, please call select_hip_platform(), then use"
                            " ROCM_EXAMPLES_GPU_LANGUAGE for the GPU language"
                            " and ROCM_EXAMPLES_HIP_PLATFORM for the HIP platform.")
endfunction()

variable_watch(GPU_RUNTIME warn_gpu_runtime_used)
