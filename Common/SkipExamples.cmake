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

# Build-skip hook for rocm-examples CI.
#
# Injected via `-DCMAKE_PROJECT_INCLUDE_BEFORE=<repo>/Common/SkipExamples.cmake`,
# so CMake runs it at the start of every project() call. It overrides
# add_subdirectory() to drop any example whose repo-root-relative path appears in
# `.github/build_tools/skip_build.txt` (generated from skip_manifest.py). A
# skipped leaf's project()/add_executable()/add_test() never run, so no dangling
# target and no ctest entry are created.
#
# The override is installed exactly once (guarded by a cache variable). Because
# CMake function overrides are inherited by subdirectories, installing it at the
# first project() call intercepts every add_subdirectory() at any depth.

if(NOT DEFINED ROCM_EXAMPLES_SKIP_BUILD_INITIALIZED)
    set(ROCM_EXAMPLES_SKIP_BUILD_INITIALIZED TRUE CACHE INTERNAL "")

    # This file lives in <repo>/Common, so the repo root is one directory up.
    # It is stable regardless of which folder root cmake -S points at.
    get_filename_component(_rocm_examples_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
    set(ROCM_EXAMPLES_ROOT "${_rocm_examples_root}" CACHE INTERNAL "")

    set(_skip_file "${ROCM_EXAMPLES_ROOT}/.github/build_tools/skip_build.txt")
    set(_skip_list "")
    if(EXISTS "${_skip_file}")
        file(STRINGS "${_skip_file}" _skip_list)
    endif()
    set(ROCM_EXAMPLES_SKIP_BUILD "${_skip_list}" CACHE INTERNAL "")

    if(ROCM_EXAMPLES_SKIP_BUILD)
        message(STATUS "SkipExamples: build-skip list = ${ROCM_EXAMPLES_SKIP_BUILD}")
    endif()

    function(add_subdirectory dir)
        get_filename_component(_abs "${dir}" ABSOLUTE)
        file(RELATIVE_PATH _rel "${ROCM_EXAMPLES_ROOT}" "${_abs}")
        if("${_rel}" IN_LIST ROCM_EXAMPLES_SKIP_BUILD)
            message(WARNING "SkipExamples: skipping ${_rel} (build-skip manifest)")
            return()
        endif()
        _add_subdirectory("${dir}" ${ARGN})
    endfunction()
endif()
