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

# Composable Kernel's own build defines the following macros when its GPU targets include certain
# architectures. ck_tile and the examples also use them in host code, so define them in the same way.
# They then apply to all GPU targets of the example: e.g. when an example is built for gfx942 and gfx950,
# gfx942 also uses the OCP fp8 format of gfx950, so its fp8 and bf8 results are wrong. (The GEMM example builds its
# tf32 GEMMs only for gfx950.)

# Kernel configurations for the WMMA instructions of gfx11 and gfx12 instead of MFMA
if(CMAKE_HIP_ARCHITECTURES MATCHES "gfx11|gfx12")
    add_compile_definitions(CK_TILE_USE_WMMA=1)
else()
    add_compile_definitions(CK_TILE_USE_WMMA=0)
endif()

# fp8 and bf8 kernel configurations for the WMMA instructions of gfx12. gfx11 has no fp8 WMMA instructions and
# these configurations do not compile for it, so unlike Composable Kernel's own build, a build that also includes
# gfx11 leaves them out.
if(CMAKE_HIP_ARCHITECTURES MATCHES "gfx12" AND NOT CMAKE_HIP_ARCHITECTURES MATCHES "gfx11")
    add_compile_definitions(CK_USE_WMMA_FP8)
endif()

# Without CK_TILE_USE_OCP_FP8, ck_tile uses the OCP fp8 format in device code for gfx950 and gfx12
# but the FNUZ format in host code, so the host reference results of the examples do not match.
if(CMAKE_HIP_ARCHITECTURES MATCHES "gfx950|gfx12")
    add_compile_definitions(CK_TILE_USE_OCP_FP8)
endif()

# Features specific to gfx950 (e.g. tf32 GEMMs) and gfx1250
if(CMAKE_HIP_ARCHITECTURES MATCHES "gfx950")
    add_compile_definitions(CK_GFX950_SUPPORT)
endif()
if(CMAKE_HIP_ARCHITECTURES MATCHES "gfx1250")
    add_compile_definitions(CK_USE_GFX1250)
endif()

# Overrides the macros above on <target>, which must already exist, as if it had been built with
# CMAKE_HIP_ARCHITECTURES set to <archs> instead of its actual value. Use this when <target>'s own device
# code is restricted (e.g. via its HIP_ARCHITECTURES property) to a GPU subset narrower than the rest of
# the build: the macros above apply once to every target in this directory scope, from the whole build's
# CMAKE_HIP_ARCHITECTURES, so they do not reflect a target whose own GPUs are a strict subset of that list
# (e.g. the GEMM example's weight preshuffle target, which excludes gfx11).
#
# Always undefines, then redefines if <archs> says so: this file's directory-wide macros above are plain
# -D compiler flags (add_compile_definitions), and so is a value override from target_compile_definitions;
# CMake does not reliably order two -D flags for the same macro by which one was added last (observed to
# depend on the flag text, not call or scope order), so a later -D is not guaranteed to beat an earlier
# one. A later -U reliably beats an earlier -D for the same macro, regardless of scope or call order
# (compile options, including a target's, are always emitted after every compile definition that applies
# to it); so this function uses target_compile_options exclusively, in a single call per target so the
# -U<macro> always precedes its own -D<macro> in the same options list.
function(ck_tile_target_architectures target archs)
    if(archs MATCHES "gfx11|gfx12")
        set(wmma 1)
    else()
        set(wmma 0)
    endif()
    set(opts -UCK_TILE_USE_WMMA -DCK_TILE_USE_WMMA=${wmma} -UCK_USE_WMMA_FP8 -UCK_TILE_USE_OCP_FP8
             -UCK_GFX950_SUPPORT -UCK_USE_GFX1250
    )
    if(archs MATCHES "gfx12" AND NOT archs MATCHES "gfx11")
        list(APPEND opts -DCK_USE_WMMA_FP8)
    endif()
    if(archs MATCHES "gfx950|gfx12")
        list(APPEND opts -DCK_TILE_USE_OCP_FP8)
    endif()
    if(archs MATCHES "gfx950")
        list(APPEND opts -DCK_GFX950_SUPPORT)
    endif()
    if(archs MATCHES "gfx1250")
        list(APPEND opts -DCK_USE_GFX1250)
    endif()
    target_compile_options(${target} PRIVATE ${opts})
endfunction()

# Undefines <macro> in the device code of <target> for every one of its own HIP offload architectures
# other than <keep_arch>, via clang's per-offload-architecture -Xarch_<arch> flag. Use this when a single
# target builds for several GPUs together (so ck_tile_target_architectures above does not apply: this
# target's own architectures are not a subset, every one of them is still built) and one of the macros
# above should only take effect for one of them (e.g. CK_GFX950_SUPPORT's tf32 GEMMs, which with some ROCm
# versions do not compile for the GEMM example's other gfx9 GPUs when built together with gfx950).
function(ck_tile_target_undefine_for_other_architectures target macro keep_arch)
    if(CMAKE_HIP_ARCHITECTURES MATCHES "${keep_arch}")
        foreach(arch IN LISTS CMAKE_HIP_ARCHITECTURES)
            if(NOT arch STREQUAL "${keep_arch}")
                target_compile_options(${target} PRIVATE "SHELL:-Xarch_${arch} -U${macro}")
            endif()
        endforeach()
    endif()
endfunction()
