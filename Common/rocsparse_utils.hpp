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

#ifndef COMMON_ROCSPARSE_UTILS_HPP
#define COMMON_ROCSPARSE_UTILS_HPP

#include "example_utils.hpp"

#include <rocsparse/rocsparse.h>

#include <iostream>

/// \brief Converts a \p rocsparse_status variable to its correspondent string.
inline const char* rocsparse_status_to_string(rocsparse_status status)
{
    switch(status)
    {
        case rocsparse_status_success: return "rocsparse_status_success";
        case rocsparse_status_invalid_handle: return "rocsparse_status_invalid_handle";
        case rocsparse_status_not_implemented: return "rocsparse_status_not_implemented";
        case rocsparse_status_invalid_pointer: return "rocsparse_status_invalid_pointer";
        case rocsparse_status_invalid_size: return "rocsparse_status_invalid_size";
        case rocsparse_status_memory_error: return "rocsparse_status_memory_error";
        case rocsparse_status_internal_error: return "rocsparse_status_internal_error";
        case rocsparse_status_invalid_value: return "rocsparse_status_invalid_value";
        case rocsparse_status_arch_mismatch: return "rocsparse_status_arch_mismatch";
        case rocsparse_status_zero_pivot: return "rocsparse_status_zero_pivot";
        case rocsparse_status_not_initialized: return "rocsparse_status_not_initialized";
        case rocsparse_status_type_mismatch: return "rocsparse_status_type_mismatch";
        case rocsparse_status_thrown_exception:
            return "rocsparse_status_thrown_exception";
// rocSPARSE 3.0 adds new status
#if ROCSPARSE_VERSION_MAJOR >= 3
        case rocsparse_status_continue: return "rocsparse_status_continue";
#endif
        case rocsparse_status_requires_sorted_storage:
            return "rocsparse_status_requires_sorted_storage";
        // We do use default because we are not in control of these enumeration values.
        // Ideally this function is something rocsparse would provide
        default: return "<unknown rocsparse_status value>";
    }
}

/// \brief Checks if the provided status code is \p rocsparse_status_success and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define ROCSPARSE_CHECK(condition)                                                               \
    {                                                                                            \
        const rocsparse_status status = (condition);                                             \
        if(status != rocsparse_status_success)                                                   \
        {                                                                                        \
            std::cerr << "rocSPARSE error encountered: \"" << rocsparse_status_to_string(status) \
                      << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;                   \
            std::exit(error_exit_code);                                                          \
        }                                                                                        \
    }

#endif // COMMON_ROCSPARSE_UTILS_HPP
