# MIT License
#
# Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.21.3 FATAL_ERROR)
project(ROCm-SDK-Examples LANGUAGES CXX VERSION 6.2.0)
enable_testing()

add_subdirectory(Applications)
add_subdirectory(HIP-Basic)
add_subdirectory(Libraries)

# Copy source files in root to package
install(
    FILES LICENSE.md CMakeLists.txt README.md
    DESTINATION /opt/rocm/examples
)

# Copy source directories to package
install(
    DIRECTORY
        AI
        Applications
        Common
        Dockerfiles
        External
        HIP-Basic
        Libraries
        LLVM_ASAN
    DESTINATION /opt/rocm/examples
    PATTERN *
    PATTERN "**/build" EXCLUDE
    PATTERN ".gitignore" EXCLUDE
    PATTERN "*.vcxproj**" EXCLUDE
    PATTERN "*.sln" EXCLUDE
    PATTERN "bin" EXCLUDE      # Exclude binary directories
    PATTERN "*.o" EXCLUDE      # Exclude object files
    PATTERN "*.exe" EXCLUDE    # Exclude executable files
)

###
# Packaging Directives
###

set(CPACK_PACKAGE_NAME "ROCm-SDK-Examples")
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_CONTACT "ROCm Developer Support <rocm-dev.support@amd.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "A collection of examples for the ROCm software stack"
)

set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION
    "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}"
)

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm/examples")

set(CPACK_GENERATOR "DEB;RPM")

# Generate DEB package
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/ROCm/ROCm-examples")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS YES)

# Generate RPM package
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
set(CPACK_RPM_AUTOREQPROV YES)

## Build system packaging overrides
if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
    set(CPACK_PACKAGE_VERSION
        "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}"
    )
endif()

if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
endif()

if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
    set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
endif()

# Include CPack
include(CPack)