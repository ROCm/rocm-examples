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

set(example_name hip_llvm_ir_to_executable)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../Common/HipPlatform.cmake")
select_gpu_language()

enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD 17)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_EXTENSIONS OFF)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD_REQUIRED ON)
select_hip_platform()
verify_hip_platform(PLATFORMS "amd")

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(ROCM_ROOT
        "$ENV{HIP_PATH}"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
else()
    set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(GPU_ARCHITECTURES "all" CACHE STRING "GPU architectures to compile for")
else()
    set(GPU_ARCHITECTURES
        "${CMAKE_HIP_ARCHITECTURES}"
        CACHE STRING
        "GPU architectures to compile for"
    )
endif()

if(GPU_ARCHITECTURES STREQUAL "all")
    set(GPU_ARCHITECTURES
        "gfx803;gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx1102"
        CACHE STRING
        "GPU architectures to compile for"
        FORCE
    )
endif()

# Remove duplicates
list(REMOVE_DUPLICATES GPU_ARCHITECTURES)
message(STATUS "GPU_ARCHITECTURES: ${GPU_ARCHITECTURES}")

set_source_files_properties(
    main.hip
    PROPERTIES COMPILE_OPTIONS "--cuda-host-only;-fuse-cuid=none"
)

find_program(
    LLVM_DIS_COMMAND
    llvm-dis
    PATH_SUFFIXES bin
    PATHS ${ROCM_ROOT}/llvm ${CMAKE_INSTALL_PREFIX}/llvm
    REQUIRED
    NO_DEFAULT_PATH
)

# Generate the device LLVM IR using the HIP compiler.
foreach(HIP_ARCHITECTURE ${GPU_ARCHITECTURES})
    add_custom_command(
        OUTPUT main_${HIP_ARCHITECTURE}.ll main_${HIP_ARCHITECTURE}.bc
        COMMAND
            ${CMAKE_HIP_COMPILER} --cuda-device-only -c -emit-llvm
            ${CMAKE_CURRENT_SOURCE_DIR}/main.hip
            --offload-arch=${HIP_ARCHITECTURE} -o main_${HIP_ARCHITECTURE}.bc -I
            ${CMAKE_CURRENT_LIST_DIR}/../../Common -std=c++17
        COMMAND
            ${LLVM_DIS_COMMAND} main_${HIP_ARCHITECTURE}.bc -o
            main_${HIP_ARCHITECTURE}.ll
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/main.hip
        VERBATIM
        COMMENT "Generating main_${HIP_ARCHITECTURE}.ll"
    )
endforeach()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(OBJ_TYPE obj)
    set(NULDEV NUL)
    set(HOST_TARGET x86_64-pc-windows-msvc)
    set(HIP_OBJ_GEN_FILE hip_obj_gen_win.mcin)
else()
    set(OBJ_TYPE o)
    set(NULDEV /dev/null)
    set(HOST_TARGET x86_64-unknown-linux-gnu)
    set(HIP_OBJ_GEN_FILE hip_obj_gen.mcin)
endif()

# Assemble the device assemblies to object files using the HIP compiler.
# The compiler needs -target amdgcn-amd-amdhsa -mcpu=gfx* in order to assemble the object file
# for the right GPU.
foreach(HIP_ARCHITECTURE ${GPU_ARCHITECTURES})
    add_custom_command(
        OUTPUT main_${HIP_ARCHITECTURE}.${OBJ_TYPE}
        COMMAND
            ${CMAKE_HIP_COMPILER} -fPIC -target amdgcn-amd-amdhsa
            -mcpu=${HIP_ARCHITECTURE} main_${HIP_ARCHITECTURE}.ll -o
            ${CMAKE_CURRENT_BINARY_DIR}/main_${HIP_ARCHITECTURE}.${OBJ_TYPE}
        DEPENDS main_${HIP_ARCHITECTURE}.ll
        VERBATIM
        COMMENT "Generating main_${HIP_ARCHITECTURE}.${OBJ_TYPE}"
    )
endforeach()

# Create an offload-bundle from the assembled object files. This needs the clang-offload-bundler tool.
find_program(
    OFFLOAD_BUNDLER_COMMAND
    clang-offload-bundler
    PATH_SUFFIXES bin
    PATHS ${ROCM_ROOT}/llvm ${CMAKE_INSTALL_PREFIX}/llvm
    REQUIRED
    NO_DEFAULT_PATH
)

# Generate object bundle.
# The invocation to generate is
# clang-offload-bundler -targets=<targets> -input=<input target #1> -inputs=<input target #2> ... -output=<output>
# Note that the host target must be the first target present here, and it should have an empty input associated to it.

# Generate BUNDLE_TARGETS as a string of: -targets=host-${HOST_TARGET},hip-amdgcn-amd-amdhsa-${HIP_ARCHITECTURE},...
set(BUNDLE_TARGETS "-targets=host-${HOST_TARGET}")
# Generate BUNDLE_INPUTS as a string of: -input=${CMAKE_CURRENT_BINARY_DIR}/main_${HIP_ARCHITECTURE}.${OBJ_TYPE} ...
set(BUNDLE_INPUTS "-input=${NULDEV}")
# Generate BUNDLE_OBJECTS as a string of: ${CMAKE_CURRENT_BINARY_DIR}/main_${HIP_ARCHITECTURE}.${OBJ_TYPE}
set(BUNDLE_OBJECTS "")
foreach(HIP_ARCHITECTURE ${GPU_ARCHITECTURES})
    set(BUNDLE_TARGETS
        "${BUNDLE_TARGETS},hipv4-amdgcn-amd-amdhsa--${HIP_ARCHITECTURE}"
    )
    list(
        APPEND
        BUNDLE_INPUTS
        "-input=${CMAKE_CURRENT_BINARY_DIR}/main_${HIP_ARCHITECTURE}.${OBJ_TYPE}"
    )
    list(
        APPEND
        BUNDLE_OBJECTS
        "${CMAKE_CURRENT_BINARY_DIR}/main_${HIP_ARCHITECTURE}.${OBJ_TYPE}"
    )
endforeach()

# Invoke clang-offload-bundler to generate an offload bundle.
set(BUNDLE "${CMAKE_CURRENT_BINARY_DIR}/offload_bundle.hipfb")
add_custom_command(
    OUTPUT "${BUNDLE}"
    COMMAND
        "${OFFLOAD_BUNDLER_COMMAND}" -type=o -bundle-align=4096
        "${BUNDLE_TARGETS}" ${BUNDLE_INPUTS} "-output=${BUNDLE}"
    DEPENDS ${BUNDLE_OBJECTS}
    VERBATIM
)

# Create the device binary by assembling the template that includes
# the offload bundle that was just generated using an .incbin directive.
# This needs an assembler.
find_program(
    LLVM_MC_COMMAND
    llvm-mc
    PATH_SUFFIXES bin
    PATHS ${ROCM_ROOT}/llvm ${CMAKE_INSTALL_PREFIX}/llvm
    REQUIRED
    NO_DEFAULT_PATH
)

# Invoke llvm-mc to generate an object file containing the offload bundle.
set(DEVICE_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/main_device.${OBJ_TYPE}")
add_custom_command(
    OUTPUT "${DEVICE_OBJECT}"
    COMMAND
        "${LLVM_MC_COMMAND}" -triple "${HOST_TARGET}"
        "${CMAKE_CURRENT_SOURCE_DIR}/${HIP_OBJ_GEN_FILE}" -o "${DEVICE_OBJECT}"
        --filetype=obj
    DEPENDS "${BUNDLE}"
    VERBATIM
)

# Finally, create the executable.
add_executable(${example_name} main.hip ${DEVICE_OBJECT})

# Make example runnable using ctest
add_test(NAME ${example_name} COMMAND ${example_name})

set(include_dirs "../../Common" "../../External")
target_include_directories(${example_name} PRIVATE ${include_dirs})
set_source_files_properties(main.hip PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})

install(TARGETS ${example_name})
