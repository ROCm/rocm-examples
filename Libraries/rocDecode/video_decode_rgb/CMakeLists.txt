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

set(example_name rocdecode_video_decode_rgb)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

include("../../../Common/HipPlatform.cmake")
select_gpu_language()

enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD 17)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_EXTENSIONS OFF)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD_REQUIRED ON)
select_hip_platform()
verify_hip_platform(PLATFORMS "amd")

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "rocDecode examples are only available on Linux")
    return()
else()
    set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

find_package(rocdecode REQUIRED)

# Find FFmpeg libraries
find_library(AVCODEC_LIBRARY avcodec REQUIRED)
find_library(AVFORMAT_LIBRARY avformat REQUIRED)
find_library(AVUTIL_LIBRARY avutil REQUIRED)
find_library(SWRESAMPLE_LIBRARY swresample REQUIRED)

find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h REQUIRED)
find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h REQUIRED)
find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h REQUIRED)

# Check FFmpeg version for compatibility using pkg-config (same as Makefile)
execute_process(
    COMMAND pkg-config --modversion libavcodec
    OUTPUT_VARIABLE AVCODEC_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# If pkg-config fails, try to get version from header as fallback
if(NOT AVCODEC_VERSION AND AVCODEC_INCLUDE_DIR)
    file(STRINGS "${AVCODEC_INCLUDE_DIR}/libavcodec/version.h" AVCODEC_VERSION_MAJOR_LINE
         REGEX "^#define LIBAVCODEC_VERSION_MAJOR")
    file(STRINGS "${AVCODEC_INCLUDE_DIR}/libavcodec/version.h" AVCODEC_VERSION_MINOR_LINE
         REGEX "^#define LIBAVCODEC_VERSION_MINOR")
    file(STRINGS "${AVCODEC_INCLUDE_DIR}/libavcodec/version.h" AVCODEC_VERSION_MICRO_LINE
         REGEX "^#define LIBAVCODEC_VERSION_MICRO")

    string(REGEX REPLACE "^#define LIBAVCODEC_VERSION_MAJOR[ \t]+([0-9]+).*$" "\\1" AVCODEC_MAJOR "${AVCODEC_VERSION_MAJOR_LINE}")
    string(REGEX REPLACE "^#define LIBAVCODEC_VERSION_MINOR[ \t]+([0-9]+).*$" "\\1" AVCODEC_MINOR "${AVCODEC_VERSION_MINOR_LINE}")
    string(REGEX REPLACE "^#define LIBAVCODEC_VERSION_MICRO[ \t]+([0-9]+).*$" "\\1" AVCODEC_MICRO "${AVCODEC_VERSION_MICRO_LINE}")

    set(AVCODEC_VERSION "${AVCODEC_MAJOR}.${AVCODEC_MINOR}.${AVCODEC_MICRO}")
endif()

add_executable(${example_name}
    main.cpp
    ${ROCM_ROOT}/share/rocdecode/utils/rocvideodecode/roc_video_dec.cpp
    ${ROCM_ROOT}/share/rocdecode/utils/colorspace_kernels.cpp
    ${ROCM_ROOT}/share/rocdecode/utils/resize_kernels.cpp
)

target_link_libraries(${example_name}
    PRIVATE
    rocdecode::rocdecode
    ${AVCODEC_LIBRARY}
    ${AVFORMAT_LIBRARY}
    ${AVUTIL_LIBRARY}
    ${SWRESAMPLE_LIBRARY}
)

target_include_directories(
    ${example_name}
    PRIVATE
    "../../../Common"
    "../../../External"
    "${ROCM_ROOT}/share/rocdecode/utils"
    "${ROCM_ROOT}/share/rocdecode/utils/rocvideodecode"
    ${AVCODEC_INCLUDE_DIR}
    ${AVFORMAT_INCLUDE_DIR}
    ${AVUTIL_INCLUDE_DIR}
)

# FFMPEG multi-version support
if(AVCODEC_VERSION VERSION_LESS_EQUAL 58.134.100)
    target_compile_definitions(${example_name} PRIVATE USE_AVCODEC_GREATER_THAN_58_134=0)
else()
    target_compile_definitions(${example_name} PRIVATE USE_AVCODEC_GREATER_THAN_58_134=1)
endif()

set_source_files_properties(
    main.cpp
    ${ROCM_ROOT}/share/rocdecode/utils/rocvideodecode/roc_video_dec.cpp
    ${ROCM_ROOT}/share/rocdecode/utils/colorspace_kernels.cpp
    ${ROCM_ROOT}/share/rocdecode/utils/resize_kernels.cpp
    PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE}
)

install(TARGETS ${example_name})
