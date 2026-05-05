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

cmake_minimum_required(VERSION 3.25.2 FATAL_ERROR)
project(hipDNN_examples LANGUAGES CXX)

file(RELATIVE_PATH folder_bin ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${folder_bin})

if(ROCM_EXAMPLES_GPU_LANGUAGE STREQUAL "CUDA")
    message(STATUS "hipDNN examples do not support the CUDA runtime")
    return()
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include("${CMAKE_CURRENT_LIST_DIR}/../../Common/ROCmPath.cmake")

find_package(hipdnn_frontend QUIET CONFIG)
if(NOT hipdnn_frontend_FOUND)
    message(WARNING "hipDNN frontend could not be found, not building hipDNN examples.")
    return()
endif()

find_package(hipdnn_test_sdk QUIET CONFIG)
if(NOT hipdnn_test_sdk_FOUND)
    message(WARNING "hipDNN test SDK could not be found, not building hipDNN examples.")
    return()
endif()

# Batch normalization examples
add_subdirectory(bn_inference)
add_subdirectory(bn_inference_with_variance)
add_subdirectory(bn_training)
add_subdirectory(bn_backward)
add_subdirectory(fused_bn_training_activ)
add_subdirectory(fused_bn_inference_activ)
add_subdirectory(fused_bn_inference_variance_activ)
add_subdirectory(fused_bn_inf_drelu_bn_bwd)

# Convolution examples
add_subdirectory(conv_fprop)
add_subdirectory(conv_dgrad)
add_subdirectory(conv_wgrad)
add_subdirectory(conv_fprop_deterministic)

if(NOT WIN32)
    add_subdirectory(fused_conv_fprop_activ)
    add_subdirectory(fused_conv_fprop_bias_activ)
endif()

# Matrix multiplication examples
add_subdirectory(matmul)
