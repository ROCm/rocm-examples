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

set(example_name rocdecode_video_decode_raw)

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
endif()

include("../../../Common/ROCmPath.cmake")

find_package(HIP REQUIRED)
find_package(rocdecode REQUIRED)
find_package(rocprofiler-register REQUIRED)
find_package(Threads REQUIRED)

add_executable(${example_name}
    main.cpp
    ../utils/rocvideodecode/roc_video_dec.cpp
)

target_link_libraries(${example_name}
    PRIVATE
    hip::host
    rocdecode::rocdecode
    rocprofiler-register::rocprofiler-register
    Threads::Threads
)

target_include_directories(
    ${example_name}
    PRIVATE
    "../../../Common"
    "../../../External"
    "../utils"
    "../utils/rocvideodecode"
)

set_source_files_properties(
    main.cpp
    ../utils/rocvideodecode/roc_video_dec.cpp
    PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE}
)

set(ROCDECODE_TEST_VIDEO_RAW_HEVC "${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H265.265")
if(EXISTS "${ROCDECODE_TEST_VIDEO_RAW_HEVC}")
    add_test(NAME ${example_name}-HEVC COMMAND ${example_name} -i "${ROCDECODE_TEST_VIDEO_RAW_HEVC}")
else()
    message(WARNING "rocDecode test video ${ROCDECODE_TEST_VIDEO_RAW_HEVC} not found, skipping ${example_name}-HEVC test")
endif()

set(ROCDECODE_TEST_VIDEO_RAW_AVC "${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H264.264")
if(EXISTS "${ROCDECODE_TEST_VIDEO_RAW_AVC}")
    add_test(NAME ${example_name}-AVC COMMAND ${example_name} -i "${ROCDECODE_TEST_VIDEO_RAW_AVC}")
else()
    message(WARNING "rocDecode test video ${ROCDECODE_TEST_VIDEO_RAW_AVC} not found, skipping ${example_name}-AVC test")
endif()

set(ROCDECODE_TEST_VIDEO_RAW_AV1 "${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-AV1.ivf")
if(EXISTS "${ROCDECODE_TEST_VIDEO_RAW_AV1}")
    add_test(NAME ${example_name}-AV1 COMMAND ${example_name} -i "${ROCDECODE_TEST_VIDEO_RAW_AV1}")
else()
    message(WARNING "rocDecode test video ${ROCDECODE_TEST_VIDEO_RAW_AV1} not found, skipping ${example_name}-AV1 test")
endif()

set(ROCDECODE_TEST_VIDEO_RAW_VP9 "${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-VP9.ivf")
if(EXISTS "${ROCDECODE_TEST_VIDEO_RAW_VP9}")
    add_test(NAME ${example_name}-VP9 COMMAND ${example_name} -i "${ROCDECODE_TEST_VIDEO_RAW_VP9}")
else()
    message(WARNING "rocDecode test video ${ROCDECODE_TEST_VIDEO_RAW_VP9} not found, skipping ${example_name}-VP9 test")
endif()

install(TARGETS ${example_name})
