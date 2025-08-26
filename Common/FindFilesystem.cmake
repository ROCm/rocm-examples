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

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

include(CMakePushCheckState)
include(CheckCXXSourceCompiles)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_push_check_state()

set(test_code
[[
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main()
{
    std::cout << fs::current_path() << std::endl;
    return 0;
}
]]
)
# Check if std::filesystem works without additional libraries
check_cxx_source_compiles("${test_code}" CXX_FS_NO_LINK)

if (CXX_FS_NO_LINK)
    message(STATUS "No extra linking required to use std::filesystem")
else()
    # Check if we can link stdc++fs
	set(CMAKE_REQUIRED_LIBRARIES stdc++fs)
	check_cxx_source_compiles("${test_code}" CXX_FS_CAN_LINK)
    if (CXX_FS_CAN_LINK)
        set(CXX_FS_LIBRARY stdc++fs CACHE STRING "Additional library required to use std::filesystem" FORCE)
        message(STATUS "Need explicite linking to stdc++fs")
    endif()
endif()

unset(test_code)
cmake_pop_check_state()

if(NOT CXX_FS_NO_LINK AND NOT CXX_FS_CAN_LINK)
    message(FATAL_ERROR "Cannot run simple program using std::filesystem")
endif()
