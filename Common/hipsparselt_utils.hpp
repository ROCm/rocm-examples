// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef COMMON_HIPSPARSELT_UTILS_HPP
#define COMMON_HIPSPARSELT_UTILS_HPP

#include <hipsparselt/hipsparselt.h>

#include <cstdlib>
#include <iostream>

/// \brief Converts a \p hipsparseStatus_t to its correspondent string.
inline const char* hipsparseStatusToString(hipsparseStatus_t status)
{
    switch(status)
    {
    case HIPSPARSE_STATUS_SUCCESS:
        return "Success";

    case HIPSPARSE_STATUS_NOT_INITIALIZED:
        return "hipSPARSELt was not initialized";

    case HIPSPARSE_STATUS_ALLOC_FAILED:
        return "Resource allocation failed";

    case HIPSPARSE_STATUS_INVALID_VALUE:
        return "Invalid value";

    case HIPSPARSE_STATUS_ARCH_MISMATCH:
        return "Device architecture not supported";

    case HIPSPARSE_STATUS_MAPPING_ERROR:
        return "Access to GPU memory space failed";

    case HIPSPARSE_STATUS_EXECUTION_FAILED:
        return "GPU program failed to execute";

    case HIPSPARSE_STATUS_INTERNAL_ERROR:
        return "An internal hipSPARSELt operation failed";

    case HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "Matrix type not supported";

    case HIPSPARSE_STATUS_ZERO_PIVOT:
        return "Zero pivot was computed";

    case HIPSPARSE_STATUS_NOT_SUPPORTED:
        return "Operation is not supported";

#if !defined(CUDART_VERSION) || (defined(CUDART_VERSION) && CUDART_VERSION >= 11003)
    case HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES:
        return "Resources are insufficient";    
#endif

        // We do use default because we are not in control of these enumeration values.
        // Ideally this function is something hipSPARSELt would provide
    default:
        return "<unknown hipsparseStatus_t value>";
    }
}

/// \brief Checks if the provided status code is \p HIPSPARSE_STATUS_SUCCESS and if not, prints an error message to the
/// standard error output and terminates the program with an error code.
#define HIPSPARSELT_CHECK(condition)                            \
{                                                               \
    const hipsparseStatus_t status = condition;                 \
    if(status != HIPSPARSE_STATUS_SUCCESS)                      \
    {                                                           \
        std::cerr << "hipSPARSELt error encountered: \""        \
                  << hipsparseStatusToString(status)            \
                  << "\" at " << __FILE__ << ':' << __LINE__    \
                  << std::endl;                                 \
        std::exit(EXIT_FAILURE);                                \
    }                                                           \
}

#endif
