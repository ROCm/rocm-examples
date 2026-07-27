# MIT License
#
# Copyright (c) 2026 firedoil
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

set(example_prefix applications_pqc)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_prefix} LANGUAGES CXX)

include(CTest)

include("${CMAKE_CURRENT_LIST_DIR}/../../Common/HipPlatform.cmake")
select_gpu_language()

enable_language(${ROCM_EXAMPLES_GPU_LANGUAGE})
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD 17)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_EXTENSIONS OFF)
set(CMAKE_${ROCM_EXAMPLES_GPU_LANGUAGE}_STANDARD_REQUIRED ON)
select_hip_platform()
verify_hip_platform(PLATFORMS amd)

option(
    ROCM_EXAMPLES_PQC_BUILD_ALL_VARIANTS
    "Build every supported PQC parameter set"
    OFF
)

function(add_pqc_example target source algorithm parameter_mode)
    add_executable(${target} ${source})
    target_compile_definitions(
        ${target}
        PRIVATE ALGORITHM=${algorithm} PARAM_MODE=${parameter_mode}
    )
    set_source_files_properties(
        ${source}
        PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE}
    )
    if(BUILD_TESTING)
        add_test(NAME ${target} COMMAND ${target} --self-test)
        set_tests_properties(${target} PROPERTIES TIMEOUT 60)
    endif()
    install(TARGETS ${target})
endfunction()

function(add_pqc_signature_example target algorithm parameter_mode)
    add_pqc_example(${target} sig_api/main.hip ${algorithm} ${parameter_mode})
    target_compile_definitions(
        ${target}
        PRIVATE BATCH_KEYGEN_SAMPLE_SPLIT_FAST=1
    )
endfunction()

add_pqc_example(${example_prefix}_kyber768 kem_api/main.hip 1 3)
add_pqc_example(${example_prefix}_aigis_enc3 kem_api/main.hip 2 3)
add_pqc_signature_example(${example_prefix}_mldsa65 1 3)
add_pqc_signature_example(${example_prefix}_aigis_sig2 2 2)

if(ROCM_EXAMPLES_PQC_BUILD_ALL_VARIANTS)
    add_pqc_example(${example_prefix}_kyber512 kem_api/main.hip 1 2)
    add_pqc_example(${example_prefix}_kyber1024 kem_api/main.hip 1 4)
    add_pqc_example(${example_prefix}_aigis_enc1 kem_api/main.hip 2 1)
    add_pqc_example(${example_prefix}_aigis_enc2 kem_api/main.hip 2 2)
    add_pqc_example(${example_prefix}_aigis_enc4 kem_api/main.hip 2 4)
    add_pqc_signature_example(${example_prefix}_mldsa44 1 2)
    add_pqc_signature_example(${example_prefix}_mldsa87 1 5)
    add_pqc_signature_example(${example_prefix}_aigis_sig1 2 1)
    add_pqc_signature_example(${example_prefix}_aigis_sig3 2 3)
endif()
