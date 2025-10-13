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

set(example_objects ${PROJECT_NAME}_objects)
set(example_streams ${PROJECT_NAME}_streams)
set(example_graph_capture ${PROJECT_NAME}_graph_capture)
set(example_graph_creation ${PROJECT_NAME}_graph_creation)

include(FetchContent)

find_package(TIFF COMPONENTS CXX)
if(NOT TIFF_FOUND)
    message(STATUS "Could not find system libtiff, fetching from upstream...")
    FetchContent_Declare(
        libtiff
        GIT_REPOSITORY https://gitlab.com/libtiff/libtiff.git
        GIT_TAG v4.7.1)
    set(BUILD_SHARED_LIBS "ON")
    FetchContent_MakeAvailable(libtiff)

    target_compile_definitions(tiff PRIVATE $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS>)
    target_compile_definitions(tiffxx PRIVATE $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS>)

    set(TIFF_TARGET TIFF::tiff) # CMake's FindTIFF module uses the uppercase notation, libTIFF uses lowercase
    install(TARGETS tiff tiffxx)
else()
    set(TIFF_TARGET TIFF::TIFF)
endif()

find_package(hip REQUIRED)
find_package(hipfft REQUIRED)

add_library(${example_objects} OBJECT backprojection.hip
                                      filtering.hip
                                      log_transform.hip
                                      normalization.hip
                                      phantom.hip
                                      tiff.cpp
                                      volume.cpp
                                      weighting.hip)
set_source_files_properties(backprojection.hip
                            filtering.hip
                            log_transform.hip
                            normalization.hip
                            phantom.hip
                            weighting.hip
                            PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})
set_target_properties(${example_objects} PROPERTIES $<$<COMPILE_LANGUAGE:CUDA>:CUDA_EXTENSIONS OFF>
                                                    $<$<COMPILE_LANGUAGE:CUDA>:CUDA_SEPARABLE_COMPILATION ON>
                                                    CXX_EXTENSIONS OFF
                                                    CXX_SCAN_FOR_MODULES OFF
                                                    $<$<COMPILE_LANGUAGE:HIP>:HIP_EXTENSIONS OFF>)
target_compile_definitions(${example_objects} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:__HIP_PLATFORM_NVIDIA__>
                                                     $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS>)
target_compile_features(${example_objects} PRIVATE cuda_std_20 cxx_std_20 hip_std_20)
target_include_directories(${example_objects} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${ROCM_ROOT}/include>)
target_link_libraries(${example_objects} PUBLIC hip::host hip::hipfft ${TIFF_TARGET} TIFF::CXX)

add_executable(${example_streams} main_streams.hip)
add_test(NAME ${example_streams} COMMAND ${example_streams})
set_source_files_properties(main_streams.hip PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})
set_target_properties(${example_streams} PROPERTIES $<$<COMPILE_LANGUAGE:CUDA>:CUDA_EXTENSIONS OFF>
                                                    $<$<COMPILE_LANGUAGE:CUDA>:CUDA_SEPARABLE_COMPILATION ON>
                                                    CXX_EXTENSIONS OFF
                                                    CXX_SCAN_FOR_MODULES OFF
                                                    $<$<COMPILE_LANGUAGE:HIP>:HIP_EXTENSIONS OFF>)
target_compile_features(${example_streams} PRIVATE cuda_std_20 cxx_std_20 hip_std_20)
target_include_directories(${example_streams} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${ROCM_ROOT}/include>)
target_link_libraries(${example_streams} PRIVATE ${example_objects} hip::hipfft)
install(TARGETS ${example_streams})

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "The graph capture tutorial is available only on Linux.")
else()
    add_executable(${example_graph_capture} main_graph_capture.hip)
    add_test(NAME ${example_graph_capture} COMMAND ${example_graph_capture})
    set_source_files_properties(main_graph_capture.hip PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})
    set_target_properties(${example_graph_capture} PROPERTIES $<$<COMPILE_LANGUAGE:CUDA>:CUDA_EXTENSIONS OFF>
                                                              $<$<COMPILE_LANGUAGE:CUDA>:CUDA_SEPARABLE_COMPILATION ON>
                                                              CXX_EXTENSIONS OFF
                                                              CXX_SCAN_FOR_MODULES OFF
                                                              $<$<COMPILE_LANGUAGE:HIP>:HIP_EXTENSIONS OFF>)
    target_compile_features(${example_graph_capture} PRIVATE cuda_std_20 cxx_std_20 hip_std_20)
    target_include_directories(${example_graph_capture} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${ROCM_ROOT}/include>)
    target_link_libraries(${example_graph_capture} PRIVATE ${example_objects} hip::hipfft)
    install(TARGETS ${example_graph_capture})
endif()

add_executable(${example_graph_creation} main_graph_creation.hip)
add_test(NAME ${example_graph_creation} COMMAND ${example_graph_creation})
set_source_files_properties(main_graph_creation.hip PROPERTIES LANGUAGE ${ROCM_EXAMPLES_GPU_LANGUAGE})
set_target_properties(${example_graph_creation} PROPERTIES $<$<COMPILE_LANGUAGE:CUDA>:CUDA_EXTENSIONS OFF>
                                                           $<$<COMPILE_LANGUAGE:CUDA>:CUDA_SEPARABLE_COMPILATION ON>
                                                           CXX_EXTENSIONS OFF
                                                           CXX_SCAN_FOR_MODULES OFF
                                                           $<$<COMPILE_LANGUAGE:HIP>:HIP_EXTENSIONS OFF>)
target_compile_features(${example_graph_creation} PRIVATE cuda_std_20 cxx_std_20 hip_std_20)
target_include_directories(${example_graph_creation} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${ROCM_ROOT}/include>)
target_link_libraries(${example_graph_creation} PRIVATE ${example_objects} hip::hipfft)
install(TARGETS ${example_graph_creation})

# Make sure the TIFF library can be picked up by the tutorial application
if(NOT TIFF_FOUND)
    set_target_properties(tiff PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:${example_streams}>
                                          RUNTIME_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:${example_streams}>)
    set_target_properties(tiffxx PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:${example_streams}
                                            RUNTIME_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:${example_streams}>)
endif()
