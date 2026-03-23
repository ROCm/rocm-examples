# MIT License
#
# Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

include_guard(GLOBAL)

# Resolve the ROCm installation path with the following priority:
#   1. Existing cache value (-DROCM_PATH=... or previous configure)
#   2. $ENV{ROCM_PATH}  (TheRock / standard ROCm)
#   3. $ENV{HIP_PATH}   (Windows HIP SDK, also set on Linux by some CI)
#   4. /opt/rocm         (hardcoded fallback)

if(NOT DEFINED CACHE{ROCM_PATH})
    if(NOT "$ENV{ROCM_PATH}" STREQUAL "")
        set(_rocm_path_default "$ENV{ROCM_PATH}")
    elseif(NOT "$ENV{HIP_PATH}" STREQUAL "")
        set(_rocm_path_default "$ENV{HIP_PATH}")
    else()
        set(_rocm_path_default "/opt/rocm")
    endif()
else()
    set(_rocm_path_default "$CACHE{ROCM_PATH}")
endif()

set(ROCM_PATH "${_rocm_path_default}" CACHE PATH "Root directory of the ROCm installation")
unset(_rocm_path_default)

list(FIND CMAKE_PREFIX_PATH "${ROCM_PATH}" _rocm_path_idx)
if(_rocm_path_idx EQUAL -1)
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
endif()
unset(_rocm_path_idx)
