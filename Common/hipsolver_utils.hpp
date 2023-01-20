// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef COMMON_HIPSOLVER_UTILS_HPP
#define COMMON_HIPSOLVER_UTILS_HPP

#include "example_utils.hpp"

#include <hipsolver/hipsolver.h>

#include <iostream>

/// \brief Converts a \p hipsolverStatus_t variable to its correspondent string.
const char* hipsolverStatusToString(hipsolverStatus_t status)
{
    switch(status)
    {
        case HIPSOLVER_STATUS_SUCCESS: return "HIPSOLVER_STATUS_SUCCESS";
        case HIPSOLVER_STATUS_NOT_INITIALIZED: return "HIPSOLVER_STATUS_NOT_INITIALIZED";
        case HIPSOLVER_STATUS_ALLOC_FAILED: return "HIPSOLVER_STATUS_ALLOC_FAILED";
        case HIPSOLVER_STATUS_INVALID_VALUE: return "HIPSOLVER_STATUS_INVALID_VALUE";
        case HIPSOLVER_STATUS_MAPPING_ERROR: return "HIPSOLVER_STATUS_MAPPING_ERROR";
        case HIPSOLVER_STATUS_EXECUTION_FAILED: return "HIPSOLVER_STATUS_EXECUTION_FAILED";
        case HIPSOLVER_STATUS_INTERNAL_ERROR: return "HIPSOLVER_STATUS_INTERNAL_ERROR";
        case HIPSOLVER_STATUS_NOT_SUPPORTED: return "HIPSOLVER_STATUS_NOT_SUPPORTED";
        case HIPSOLVER_STATUS_ARCH_MISMATCH: return "HIPSOLVER_STATUS_ARCH_MISMATCH";
        case HIPSOLVER_STATUS_HANDLE_IS_NULLPTR: return "HIPSOLVER_STATUS_HANDLE_IS_NULLPTR";
        case HIPSOLVER_STATUS_INVALID_ENUM: return "HIPSOLVER_STATUS_INVALID_ENUM";
        case HIPSOLVER_STATUS_UNKNOWN: return "HIPSOLVER_STATUS_UNKNOWN";
    }
    // We don't use default so that the compiler warns if any valid enums are missing from the
    // switch. If the value is not a valid hipsolverStatus_t, we return the following.
    return "<undefined hipsolverStatus_t value>";
}

/// \brief Checks if the provided status code is \p HIPSOLVER_STATUS_SUCCESS and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIPSOLVER_CHECK(condition)                                                            \
    {                                                                                         \
        const hipsolverStatus_t status = condition;                                           \
        if(status != HIPSOLVER_STATUS_SUCCESS)                                                \
        {                                                                                     \
            std::cerr << "hipSOLVER error encountered: \"" << hipsolverStatusToString(status) \
                      << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;                \
            std::exit(error_exit_code);                                                       \
        }                                                                                     \
    }

#endif // COMMON_HIPSOLVER_UTILS_HPP
