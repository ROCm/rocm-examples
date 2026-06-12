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

set(example_name rocdecode_video_decode_pic_files)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../cmake")
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
endif()

include("../../../Common/ROCmPath.cmake")

find_package(HIP REQUIRED)
find_package(rocdecode REQUIRED)
find_package(rocdecode-host 1.0.0 QUIET)
find_package(FFmpeg REQUIRED)
find_package(rocprofiler-register REQUIRED)
find_package(Threads REQUIRED)

set(SOURCES
    main.cpp
    "../utils/rocvideodecode/roc_video_dec.cpp"
)

if(rocdecode-host_FOUND)
    list(APPEND SOURCES "../utils/ffmpegvideodecode/ffmpeg_video_dec.cpp")
endif()

add_executable(${example_name} ${SOURCES})

target_link_libraries(${example_name}
    PRIVATE
    hip::host
    rocdecode::rocdecode
    rocprofiler-register::rocprofiler-register
    ${FFMPEG_LIBRARIES}
    Threads::Threads
)

# Add rocdecode host utils if host library found
if(rocdecode-host_FOUND)
    target_include_directories(${example_name} PRIVATE "${rocdecode-host_INCLUDE_DIR}" "../utils/ffmpegvideodecode")
    target_link_libraries(${example_name} PRIVATE rocdecode::rocdecode-host)
    target_compile_definitions(${example_name} PRIVATE ENABLE_HOST_DECODE=1)
else()
    target_compile_definitions(${example_name} PRIVATE ENABLE_HOST_DECODE=0)
endif()

target_include_directories(
    ${example_name}
    PRIVATE
    "../../../Common"
    "../../../External"
    "../utils"
    "../utils/rocvideodecode"
    "${FFMPEG_INCLUDE_DIR}"
)

# FFMPEG multi-version support
if(_FFMPEG_AVCODEC_VERSION VERSION_LESS_EQUAL 58.134.100)
    target_compile_definitions(${example_name} PRIVATE USE_AVCODEC_GREATER_THAN_58_134=0)
else()
    target_compile_definitions(${example_name} PRIVATE USE_AVCODEC_GREATER_THAN_58_134=1)
endif()

set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

install(TARGETS ${example_name})
