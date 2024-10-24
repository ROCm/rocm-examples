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

set(example_name hip_static_host_library)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

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
set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
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

set(library_name hip_static_host)
set(example_name_cxx ${example_name}_cxx)

set(include_dirs "../../Common" "../../External")
if(GPU_RUNTIME STREQUAL "CUDA")
    list(APPEND include_dirs "${ROCM_ROOT}/include")
endif()

# Create the HIP static host library.
add_library(${library_name} STATIC library/library.hip)
target_include_directories(${library_name} PRIVATE ${include_dirs})
target_include_directories(${library_name} PUBLIC library)
set_target_properties(${library_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_source_files_properties(
    library/library.hip
    PROPERTIES LANGUAGE ${GPU_RUNTIME}
)

# Create a driver executable using ROCm's bundled version of clang.
add_executable(${example_name} main.cpp)
# Link the static host library we have just created.
target_link_libraries(${example_name} PRIVATE ${library_name})
# We are creating a regular, non-HIP, executable, and so we don't need to pass
# a list of devices for this target.
set_target_properties(${example_name} PROPERTIES HIP_ARCHITECTURES FALSE)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${GPU_RUNTIME})

# Copy main.cpp so we don't use the same source file properties as above target
file(COPY_FILE main.cpp ${CMAKE_CURRENT_BINARY_DIR}/main_copy.cpp)
# Create a driver executable using the host c++ compiler.
add_executable(${example_name_cxx} ${CMAKE_CURRENT_BINARY_DIR}/main_copy.cpp)

# Link the static host library we have just created.
target_link_libraries(${example_name_cxx} PRIVATE ${library_name})
# Set the linker language to CXX so that CMake uses the host CXX compiler to compile
# and link this version of the executable.
set_target_properties(${example_name_cxx} PROPERTIES LINKER_LANGUAGE CXX)

# Make examples runnable using ctest
add_test(NAME ${example_name} COMMAND ${example_name})
add_test(NAME ${example_name_cxx} COMMAND ${example_name_cxx})

install(TARGETS ${example_name} ${example_name_cxx})
