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

set(example_name cuda_load_module_ex)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED) # We must explicitly link the CUDA driver library.

set(example_module_prefix "${example_name}_")
set(example_module_suffix "myKernel")
set(example_module "${example_module_prefix}${example_module_suffix}")
add_library(${example_module} OBJECT myKernel.cu)
target_compile_features(${example_module} PRIVATE cuda_std_17)
set_property(TARGET ${example_module} PROPERTY CUDA_PTX_COMPILATION ON)

add_executable(${example_name} main.cpp)
# Make example runnable using ctest
add_test(NAME ${example_name} COMMAND ${example_name})

add_dependencies(${example_name} ${example_module})
set_target_properties(${example_name} PROPERTIES CXX_EXTENSIONS OFF)
target_compile_features(${example_name} PRIVATE cxx_std_17)
target_link_libraries(${example_name} PRIVATE CUDA::cuda_driver)

set(example_module_ptx "$<TARGET_FILE_DIR:${example_name}>/${example_module_suffix}.ptx")
add_custom_command(TARGET ${example_name} POST_BUILD
                   COMMAND ${CMAKE_COMMAND}
                   ARGS -E copy $<TARGET_OBJECTS:${example_module}> ${example_module_ptx}
                   COMMAND_EXPAND_LISTS
                   COMMENT "Copying ${example_module_ptx}.")
install(FILES ${example_module_ptx} TYPE BIN)

install(TARGETS ${example_name})
