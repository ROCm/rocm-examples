# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

project(reduction_benchmarks LANGUAGES CXX)

if("${ROCM_EXAMPLES_GPU_LANGUAGE}" STREQUAL "CUDA")
    cmake_minimum_required(VERSION 3.25.2)
    # Allow __device__ or __host__ annotated lambdas
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
else()
    cmake_minimum_required(VERSION 3.21)
    # Add -fPIE flag to compiler.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../Common/HipPlatform.cmake")
select_gpu_language()
enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()

set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD 20)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_EXTENSIONS OFF)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD_REQUIRED ON)

include("${CMAKE_CURRENT_LIST_DIR}/../../../Common/ROCmPath.cmake")

set(include_dirs "../../Common" "../../External")
list(APPEND include_dirs "${PROJECT_SOURCE_DIR}")

if("${ROCM_EXAMPLES_GPU_LANGUAGE}" STREQUAL "CUDA")
    # For examples targeting NVIDIA, include the HIP header directory.
    list(APPEND include_dirs "${ROCM_PATH}/include")

    # Some CUDA versions have issues when compiling for C++20 on Windows, check if we are using those
    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        find_package(CUDA)
        if(CUDA_FOUND)
            set(FAULTY_CUDA_VERSION "12.5")
            if(CUDA_VERSION VERSION_LESS_EQUAL ${FAULTY_CUDA_VERSION})
                message(
                    WARNING
                    "CUDA version ${CUDA_VERSION} has issues when compiling for C++20. Not building reduction benchmarks."
                )
                return()
            endif()
        else()
            message(
                STATUS
                "CUDA Toolkit not found. Not building reduction benchmarks."
            )
            return()
        endif()
    endif()
endif()

# libstdc++ Parallel STL on Ubuntu 20.04 requires explicit linking to TBB
if(CMAKE_HOST_APPLE)
    # TODO: The macro name used to detect TBB dependence differs on MacOS
    set(STL_DEPENDS_ON_TBB ON)
else()
    include(CMakePrintHelpers)
    include(CheckCXXSymbolExists)
    set(CMAKE_REQUIRED_FLAGS "-std=c++17")
    check_cxx_symbol_exists(
        __PSTL_PAR_BACKEND_TBB
        "cstddef"
        STL_DEPENDS_ON_TBB0
    )
    if(NOT STL_DEPENDS_ON_TBB0)
        check_cxx_symbol_exists(
            _PSTL_PAR_BACKEND_TBB
            "cstddef"
            STL_DEPENDS_ON_TBB1
        )
    endif()
endif()
if(STL_DEPENDS_ON_TBB0 OR STL_DEPENDS_ON_TBB1)
    find_package(TBB REQUIRED)
endif()

find_package(benchmark CONFIG QUIET)
if(NOT TARGET benchmark::benchmark)
    message(STATUS "Google Benchmark not found. Fetching...")
    option(
        BENCHMARK_ENABLE_TESTING
        "Enable testing of the benchmark library."
        OFF
    )
    option(BENCHMARK_ENABLE_INSTALL "Enable installation of benchmark." OFF)
    option(BENCHMARK_ENABLE_WERROR "Build Release candidates with -Werror." OFF)
    include(FetchContent)
    FetchContent_Declare(
        googlebench
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.6.1
    )
    FetchContent_MakeAvailable(googlebench)
    if(NOT TARGET benchmark::benchmark)
        add_library(benchmark::benchmark ALIAS benchmark)
    endif()
else()
    find_package(benchmark CONFIG REQUIRED)
endif()

foreach(VER RANGE 0 10)
    set(Sources v${VER}.hip)
    set(TargetName reduction_benchmark_v${VER})

    add_executable(${TargetName} ${Sources})

    target_include_directories(${TargetName} PRIVATE ${include_dirs})

    set_target_properties(${TargetName} PROPERTIES LINKER_LANGUAGE CXX)

    target_link_libraries(
        ${TargetName}
        PRIVATE
            Reduction
            benchmark::benchmark
            $<$<TARGET_EXISTS:TBB::tbb>:TBB::tbb>
    )

    if("${ROCM_EXAMPLES_GPU_LANGUAGE}" STREQUAL "CUDA")
        set_source_files_properties(
            ${Sources}
            PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE}
        )
        set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ROCM_PATH}/cmake)
        find_package(HIP MODULE REQUIRED)
        target_include_directories(
            ${TargetName}
            PRIVATE "${HIP_ROOT_DIR}/include"
        )
    endif()
endforeach()
