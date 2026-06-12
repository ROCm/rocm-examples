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

set(example_name hipfile_roundtrip_async_multi_stream_registered)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/HipPlatform.cmake")
select_gpu_language()
enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/ROCmPath.cmake")

add_executable(${example_name}
    roundtrip-async-multi-stream-registered.cpp
    ../../common/examples_common.cpp
)

set_source_files_properties(
    roundtrip-async-multi-stream-registered.cpp ../../common/examples_common.cpp
    PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE}
)

target_compile_features(${example_name} PRIVATE cxx_std_17)

target_include_directories(${example_name}
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/../../common"
        $<$<COMPILE_LANGUAGE:CUDA>:${ROCM_PATH}/include>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUFILE_INCLUDE_DIR}>
)

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    target_link_libraries(${example_name} PRIVATE hipfile ${CUFILE_LIBRARY})
else()
    target_link_libraries(${example_name} PRIVATE hipfile)
endif()

add_test(
    NAME ${example_name}
    COMMAND ${example_name}
        "${CMAKE_CURRENT_BINARY_DIR}/roundtrip_async_msr_read.bin"
        "${CMAKE_CURRENT_BINARY_DIR}/roundtrip_async_msr_write.bin"
)

install(TARGETS ${example_name})
