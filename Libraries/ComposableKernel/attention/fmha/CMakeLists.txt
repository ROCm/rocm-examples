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

set(example_name ComposableKernel_ck_tile_fmha)

set(example_fwd ${example_name}_fwd)
set(example_fwd_script test_ck_tile_fmha_fwd.sh)

set(example_bwd ${example_name}_bwd)
set(example_bwd_script test_ck_tile_fmha_bwd.sh)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Composable Kernel examples are supported only on Linux.")
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/HipPlatform.cmake")
select_gpu_language()
enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
select_hip_platform()

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    message(STATUS "Composable Kernel examples are not supported for CUDA.")
    return()
endif()

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

find_package(composable_kernel REQUIRED)
include("${CMAKE_CURRENT_LIST_DIR}/../../../../Common/FindCKTile.cmake")

find_package(hip REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

file(GLOB_RECURSE EXAMPLE_CODE_GEN_SCRIPTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/generate.py
                                                             ${CMAKE_CURRENT_LIST_DIR}/codegen/*.py
)

########################################################################################################################
# FWD
########################################################################################################################35
add_executable(${example_fwd})

execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
        --api fwd,fwd_splitkv,fwd_appendkv
        --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/fwd_src_list.txt
    RESULT_VARIABLE ret
)

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/fwd_src_list.txt EXAMPLE_FWD_GEN_SRCS)

add_custom_target(fmha_fwd_gen_srcs
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
        --api fwd,fwd_splitkv,fwd_appendkv
        --output_dir ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/fwd_src_list.txt" ${EXAMPLE_CODE_GEN_SCRIPTS}
    BYPRODUCTS ${EXAMPLE_FWD_GEN_SRCS}
    COMMENT "Generating FMHA forward pass source files"
)

add_dependencies(${example_fwd} fmha_fwd_gen_srcs)

set_source_files_properties(fmha_fwd.cpp ${EXAMPLE_FWD_GEN_SRCS} PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

target_compile_definitions(${example_fwd} PRIVATE CK_TILE_FMHA_FWD_FAST_EXP2=1
                                                  CK_TILE_FMHA_FWD_SPLITKV_API=1
                                                  CK_TILE_FMHA_FWD_APPENDKV_API=1
                                                  CK_TILE_FMHA_FWD_PAGEDKV_API=1
)
target_compile_features(${example_fwd} PRIVATE cxx_std_17 hip_std_17)
target_compile_options(${example_fwd} PRIVATE -Wno-float-equal
                                              -Wno-undefined-func-template
                                              -fgpu-flush-denormals-to-zero)
target_link_libraries(${example_fwd} PRIVATE composable_kernel::ck_tile hip::host)
target_sources(${example_fwd} PRIVATE fmha_fwd.cpp ${EXAMPLE_FWD_GEN_SRCS})

add_test(${example_fwd} COMMAND "${CMAKE_CURRENT_LIST_DIR}/${example_fwd_script} $<TARGET_FILE:${example_fwd}>")

########################################################################################################################
# BWD
########################################################################################################################
add_executable(${example_bwd})

execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
        --api bwd
        --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/bwd_src_list.txt
        --receipt 3
    RESULT_VARIABLE ret
)

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/bwd_src_list.txt EXAMPLE_BWD_GEN_SRCS)

add_custom_target(fmha_bwd_gen_srcs
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
        --api bwd
        --output_dir ${CMAKE_CURRENT_BINARY_DIR}
        --receipt 3
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/bwd_src_list.txt" ${EXAMPLE_CODE_GEN_SCRIPTS}
    BYPRODUCTS ${EXAMPLE_BWD_GEN_SRCS}
    COMMENT "Generating FMHA backward pass source files"
)

add_dependencies(${example_bwd} fmha_bwd_gen_srcs)

target_compile_options(${example_fwd} PRIVATE -Wno-float-equal
                                              -Wno-undefined-func-template
                                              -fgpu-flush-denormals-to-zero
)
target_compile_features(${example_bwd} PRIVATE cxx_std_17 hip_std_17)
target_link_libraries(${example_bwd} PRIVATE composable_kernel::ck_tile hip::host)
target_sources(${example_bwd} PRIVATE fmha_bwd.cpp ${EXAMPLE_BWD_GEN_SRCS})

set_source_files_properties(fmha_bwd.cpp ${EXAMPLE_BWD_GEN_SRCS}
                            PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

add_test(${example_bwd} COMMAND "${CMAKE_CURRENT_LIST_DIR}/${example_bwd_script} $<TARGET_FILE:${example_bwd}>")
