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

cmake_minimum_required(VERSION 3.21)

project(hip_graph_api_tutorial LANGUAGES CXX HIP)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)
include("${CMAKE_CURRENT_LIST_DIR}/../../../Common/FindFilesystem.cmake")

option(GRAPH_TUTORIAL_REAL_WORLD_DATA "Use a (large) real-world dataset instead of the Shepp-Logan phantom" OFF)

set(DATA_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${PROJECT_NAME}_data)
if(GRAPH_TUTORIAL_FETCH_REAL_WORLD_DATA)
    # Download seashell data
    set(DATA_BASE 20211124_seashell)
    set(DATA_URL https://zenodo.org/records/6983008/files/${DATA_BASE}.zip)
    set(DATA_ARCHIVE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${DATA_BASE}.zip)

    if(NOT EXISTS ${DATA_DIRECTORY}/${DATA_BASE}_0001.tif)
        message(STATUS "Downloading data archive")
        file(DOWNLOAD ${DATA_URL} ${DATA_ARCHIVE} SHOW_PROGRESS EXPECTED_HASH MD5=098895e54442370de4d6f0b526e262a1)

        file(ARCHIVE_EXTRACT INPUT ${DATA_ARCHIVE} DESTINATION ${DATA_DIRECTORY})
    endif()
endif()

find_package(TIFF COMPONENTS CXX)
if(NOT TIFF_FOUND)
    message(STATUS "Could not find system libtiff, fetching from upstream...")
    FetchContent_Declare(
        libtiff
        GIT_REPOSITORY https://gitlab.com/libtiff/libtiff.git
        GIT_TAG v4.7.0)
    set(BUILD_SHARED_LIBS "ON")
    FetchContent_MakeAvailable(libtiff)

    target_compile_definitions(tiff PRIVATE $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS>)
    target_compile_definitions(tiffxx PRIVATE $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS>)
    set(TIFF_TARGET TIFF::tiff) # CMake's FindTIFF module uses the uppercase notation, libTIFF uses lowercase
else()
    set(TIFF_TARGET TIFF::TIFF)
endif()

find_package(hip REQUIRED)
find_package(hipfft REQUIRED)

add_subdirectory(src)
