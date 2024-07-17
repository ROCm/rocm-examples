// MIT License
//
// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
inline const char* hipsolverStatusToString(hipsolverStatus_t status)
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
#if(hipsolverVersionMajor == 1 && hipsolverVersionMinor >= 8) || hipsolverVersionMajor >= 2
        case HIPSOLVER_STATUS_ZERO_PIVOT: return "HIPSOLVER_STATUS_ZERO_PIVOT";
#endif
#if(hipsolverVersionMajor >= 2 && hipsolverVersionMinor >= 1)
        case HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
#endif
        // We do use default because we are not in control of these enumeration values.
        // Ideally this function is something hipsolver would provide
        default: return "<unknown hipsolverStatus_t value>";
    }
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

/// \brief Interprets the output parameter \p info used in hipSOLVER API calls.
inline void hipsolver_print_info(int info)
{
    if(info == 0)
    {
        std::cout << "hipSOLVER call finished successfully." << std::endl;
    }
    else
    {
        std::cout << "hipSOLVER call finished unsuccessfully (" << info << ")." << std::endl;
    }
}

#endif // COMMON_HIPSOLVER_UTILS_HPP
