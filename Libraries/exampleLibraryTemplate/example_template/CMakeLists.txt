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

# DELETE ME: This is a CMakeListst.txt-template for simple examples.
# DELETE ME: If applicable, adapt this file and change example_name and library_name to the appropriate name.
# DELETE ME: The example name should be snake_case and be prefixed with the library name (i.e. ${library_name}_example_name), to avoid naming conflicts between the examples.
set(example_name library_example_template)
# DELETE ME: replace ${library_name} with the appropriate name of the example-library
# DELETE ME: replace ${library_target_name} with the correct name to link to the library, usually <namespace>::${library_name} (e.g. hip::hipcub, roc::rocrand or roc::rocthrust)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

# DELETE ME: The logic for different GPU_RUNTIMEs can be removed, if the example only supports HIP.
# DELETE ME: The occurences of GPU_RUNTIME should then be replaced with HIP.
set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
set(GPU_RUNTIMES "HIP" "CUDA")
set_property(CACHE GPU_RUNTIME PROPERTY STRINGS ${GPU_RUNTIMES})

if(NOT "${GPU_RUNTIME}" IN_LIST GPU_RUNTIMES)
    set(ERROR_MESSAGE "GPU_RUNTIME is set to \"${GPU_RUNTIME}\".\nGPU_RUNTIME must be either HIP or CUDA.")
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

# DELETE ME: Remove if no required libraries (such as HIP examples without dependencies).
# List required packages
# find_package(${library_name} REQUIRED)

add_executable(${example_name} main.hip)
# Make example runnable using ctest
add_test(${example_name} ${example_name})

# DELETE ME: Update path depending on directory nesting.
set(include_dirs "../../../Common")
# DELETE ME: Adding the HIP header directory is only required for examples that directly
#            depend on HIP without linking to a HIP-based library
# For examples targeting NVIDIA, include the HIP header directory.
if(GPU_RUNTIME STREQUAL "CUDA")
    list(APPEND include_dirs "${ROCM_ROOT}/include")
endif()

# DELETE ME: remove if no libraries to be linked
# Link to example library
target_link_libraries(${example_name}
    PRIVATE
    ${library_target_name}
)
target_include_directories(${example_name} PRIVATE ${include_dirs})
set_source_files_properties(main.hip PROPERTIES LANGUAGE ${GPU_RUNTIME})
