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

set(example_name mivisionx_mv_objdetect)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "MIVisionX examples are only available on Linux")
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../Common/ROCmPath.cmake")

# Model configuration
set(MODEL_URL "https://github.com/kiritigowda/YoloV2NCS/raw/master/models/caffemodels/yoloV2Tiny20.caffemodel")
set(MODEL_FILE "${CMAKE_CURRENT_BINARY_DIR}/yoloV2Tiny20.caffemodel")
set(BATCH_SIZE "1" CACHE STRING "Batch size for model compilation")

# Check for mv_compile
find_program(MV_COMPILE mv_compile PATHS ${ROCM_PATH}/bin REQUIRED)
if(NOT MV_COMPILE)
    message(FATAL_ERROR "mv_compile not found. Please install MIVisionX with model compiler support")
    return()
endif()

# Download model if not present
if(NOT EXISTS ${MODEL_FILE})
    message(STATUS "Downloading YoloV2 model...")
    file(DOWNLOAD ${MODEL_URL} ${MODEL_FILE}
         SHOW_PROGRESS
         STATUS download_status)
    list(GET download_status 0 status_code)
    if(NOT status_code EQUAL 0)
        message(FATAL_ERROR "Failed to download model from ${MODEL_URL}")
    endif()
    message(STATUS "Model downloaded successfully")
endif()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tmp)

# Generated files from mv_compile (in build directory)
set(GENERATED_FILES
    "${CMAKE_CURRENT_BINARY_DIR}/tmp/lib/libmv_deploy.so"
    "${CMAKE_CURRENT_BINARY_DIR}/tmp/mvdeploy_api.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/tmp/mvdeploy_api.h"
    "${CMAKE_CURRENT_BINARY_DIR}/tmp/mvdeploy.h"
    "${CMAKE_CURRENT_BINARY_DIR}/tmp/weights.bin"
    "${CMAKE_CURRENT_BINARY_DIR}/tmp/mv_extras/mv_extras_postproc.cpp"
)

# Custom command to run mv_compile
add_custom_command(
    OUTPUT ${GENERATED_FILES}
    COMMAND ${CMAKE_COMMAND} -E env
            "MIVISIONX_MODEL_COMPILER_PATH=${ROCM_PATH}/libexec/mivisionx/model_compiler"
            "ROCM_PATH=${ROCM_PATH}"
            ${MV_COMPILE} --model ${MODEL_FILE} --install_folder . --input_dims ${BATCH_SIZE},3,416,416
    DEPENDS ${MODEL_FILE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tmp
    COMMENT "Compiling YoloV2 model with mv_compile (batch size: ${BATCH_SIZE})..."
    VERBATIM
)

# Create a custom target for the generated files
add_custom_target(generate_model_files ALL DEPENDS ${GENERATED_FILES})

find_package(OpenCV REQUIRED)

add_executable(${example_name}
    main.cpp
    visualize.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/tmp/mvdeploy_api.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/tmp/mv_extras/mv_extras_postproc.cpp
)

# Make the executable depend on generated files
add_dependencies(${example_name} generate_model_files)

target_include_directories(
    ${example_name}
    PRIVATE
        "${ROCM_PATH}/include"
        "${ROCM_PATH}/include/mivisionx"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/tmp/mv_extras"
        "${CMAKE_CURRENT_BINARY_DIR}/tmp"
        "../../../Common"
        "../../../External"
)

target_link_directories(
    ${example_name}
    PRIVATE
        "${ROCM_PATH}/lib"
        "${CMAKE_CURRENT_BINARY_DIR}/tmp/lib"
)

target_link_libraries(
    ${example_name}
    PRIVATE
        openvx
        vx_nn
        vx_amd_media
        mv_deploy
        ${OpenCV_LIBRARIES}
        pthread
        ${CMAKE_DL_LIBS}
)

target_compile_definitions(
    ${example_name}
    PRIVATE
        ENABLE_MVDEPLOY=1
        ENABLE_OPENCV=1
        EXAMPLE_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
        MV_INSTALL_FOLDER="${CMAKE_CURRENT_BINARY_DIR}/tmp"
)

target_compile_options(
    ${example_name}
    PRIVATE
        -msse4.2
        -mf16c
)

install(TARGETS ${example_name})
