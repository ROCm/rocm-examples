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

project(reduction_examples LANGUAGES CXX)

if("${GPU_RUNTIME}" STREQUAL "CUDA")
    cmake_minimum_required(VERSION 3.25.2)
    # Allow __device__ or __host__ annotated lambdas
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
else()
    cmake_minimum_required(VERSION 3.21)
    # Add -fPIE flag to compiler.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

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
set(CMAKE_${GPU_RUNTIME}_STANDARD 20)
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

set(include_dirs "../../Common")
list(APPEND include_dirs "${PROJECT_SOURCE_DIR}")

if("${GPU_RUNTIME}" STREQUAL "CUDA")
    # For examples targeting NVIDIA, include the HIP header directory.
    list(APPEND include_dirs "${ROCM_ROOT}/include")

    # Some CUDA versions have issues when compiling for C++20 on Windows, check if we are using those
    if(WIN32)
        find_package(CUDA)
        if(CUDA_FOUND)
            set(FAULTY_CUDA_VERSION "12.5")
            if(CUDA_VERSION VERSION_LESS_EQUAL ${FAULTY_CUDA_VERSION})
                message(
                    WARNING
                    "CUDA version ${CUDA_VERSION} has issues when compiling for C++20. Not building reduction examples."
                )
                return()
            endif()
        else()
            message(
                STATUS
                "CUDA Toolkit not found. Not building reduction examples."
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

foreach(VER RANGE 1 9)
    set(Sources v${VER}.hip)
    set(TargetName reduction_example_v${VER})

    add_executable(${TargetName} ${Sources})

    target_include_directories(${TargetName} PRIVATE ${include_dirs})

    set_target_properties(${TargetName} PROPERTIES LINKER_LANGUAGE CXX)

    if("${GPU_RUNTIME}" STREQUAL "CUDA")
        set_target_properties(${TargetName} PROPERTIES CUDA_STANDARD 20)
    endif()

    target_link_libraries(
        ${TargetName}
        PRIVATE Reduction $<$<TARGET_EXISTS:TBB::tbb>:TBB::tbb>
    )

    set_source_files_properties(${Sources} PROPERTIES LANGUAGE ${GPU_RUNTIME})

    install(TARGETS ${TargetName})
endforeach()
