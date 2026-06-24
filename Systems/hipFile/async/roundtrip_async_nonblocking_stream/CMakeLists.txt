# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

set(example_name hipfile_roundtrip_async_nonblocking_stream)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/HipPlatform.cmake")
select_gpu_language()
enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/ROCmPath.cmake")

find_package(hipFile REQUIRED)

add_executable(${example_name}
    roundtrip-async-nonblocking-stream.cpp
    ../../common/examples_common.cpp
)

set_source_files_properties(
    roundtrip-async-nonblocking-stream.cpp ../../common/examples_common.cpp
    PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE}
)

target_compile_features(${example_name} PRIVATE cxx_std_17)

target_include_directories(${example_name}
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/../../common"
        $<$<COMPILE_LANGUAGE:CUDA>:${ROCM_PATH}/include>)

target_link_libraries(${example_name} PRIVATE hip::hipfile)

add_test(
    NAME ${example_name}
    COMMAND ${example_name}
        "${HIPFILE_TEST_DIR}/roundtrip_async_nb_read.bin"
        "${HIPFILE_TEST_DIR}/roundtrip_async_nb_write.bin"
)

install(TARGETS ${example_name})
