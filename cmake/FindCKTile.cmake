# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# There is no CMake component for ck_tile we link to, so we create our own
if(NOT TARGET ck_tile)
    find_package(hip REQUIRED)
    find_path(CK_TILE_INCLUDE_DIR NAMES ck_tile/core.hpp)

    add_library(ck_tile INTERFACE)
    target_include_directories(ck_tile INTERFACE ${CK_TILE_INCLUDE_DIR})
    target_link_libraries(ck_tile INTERFACE hip::host)

    add_library(composable_kernel::ck_tile ALIAS ck_tile)
endif()

# Without CK_TILE_USE_OCP_FP8, ck_tile uses the OCP fp8 format in device code for gfx950 and gfx12
# but the FNUZ format in host code, so the host reference results of the examples do not match.
# Define it when building for these targets, as Composable Kernel's own build does. This also
# applies to the other targets of the same build (e.g. gfx942), whose fp8 results are then wrong.
if(CMAKE_HIP_ARCHITECTURES MATCHES "gfx950|gfx12")
    add_compile_definitions(CK_TILE_USE_OCP_FP8)
endif()
