# MIT License
#
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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

set(example_name rocdecode_rocdec_decode)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../cmake")
include("../../../Common/FindFilesystem.cmake")
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

find_package(HIP REQUIRED)
find_package(rocdecode REQUIRED)
find_package(rocdecode-host 1.0.0 QUIET)
find_package(FFmpeg QUIET)

add_executable(${example_name} main.cpp)

target_link_libraries(${example_name} PRIVATE hip::host rocdecode::rocdecode ${CXX_FS_LIBRARY})

# Link FFmpeg and host library if found
if(FFMPEG_FOUND)
    target_include_directories(${example_name} PRIVATE "${FFMPEG_INCLUDE_DIR}")
    target_link_libraries(${example_name} PRIVATE ${FFMPEG_LIBRARIES})
endif()

if(rocdecode-host_FOUND)
    # rocdecode-host
    target_include_directories(${example_name} PRIVATE "${rocdecode-host_INCLUDE_DIR}")
    target_link_libraries(${example_name} PRIVATE rocdecode::rocdecode-host)
endif()

target_include_directories(
    ${example_name}
    PRIVATE "../../../Common" "../../../External"
)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

set(ROCDECODE_TEST_FRAMES_DIR "${ROCM_ROOT}/share/rocdecode/frames")
if(EXISTS "${ROCDECODE_TEST_FRAMES_DIR}")
    add_test(NAME ${example_name} COMMAND ${example_name} -i "${ROCDECODE_TEST_FRAMES_DIR}")
else()
    message(WARNING "rocDecode test frames directory ${ROCDECODE_TEST_FRAMES_DIR} not found, skipping ${example_name} test")
endif()

install(TARGETS ${example_name})
