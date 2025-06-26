// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef COMMON_HIPFFT_UTILS_HPP
#define COMMON_HIPFFT_UTILS_HPP

#include <hipfft/hipfft.h>

#include <iostream>

/// \brief Converts a \p hipfftResult_t variable to its correspondent string.
inline const char* hipfftResultToString(hipfftResult_t status)
{
    switch(status)
    {
        case HIPFFT_SUCCESS: return "HIPFFT_SUCCESS";
        case HIPFFT_INVALID_PLAN: return "HIPFFT_INVALID_PLAN";
        case HIPFFT_ALLOC_FAILED: return "HIPFFT_ALLOC_FAILED";
        case HIPFFT_INVALID_TYPE: return "HIPFFT_INVALID_TYPE";
        case HIPFFT_INVALID_VALUE: return "HIPFFT_INVALID_VALUE";
        case HIPFFT_INTERNAL_ERROR: return "HIPFFT_INTERNAL_ERROR";
        case HIPFFT_EXEC_FAILED: return "HIPFFT_EXEC_FAILED";
        case HIPFFT_SETUP_FAILED: return "HIPFFT_SETUP_FAILED";
        case HIPFFT_INVALID_SIZE: return "HIPFFT_INVALID_SIZE";
        case HIPFFT_UNALIGNED_DATA: return "HIPFFT_UNALIGNED_DATA";
        case HIPFFT_INCOMPLETE_PARAMETER_LIST: return "HIPFFT_INCOMPLETE_PARAMETER_LIST";
        case HIPFFT_INVALID_DEVICE: return "HIPFFT_INVALID_DEVICE";
        case HIPFFT_PARSE_ERROR: return "HIPFFT_PARSE_ERROR";
        case HIPFFT_NO_WORKSPACE: return "HIPFFT_NO_WORKSPACE";
        case HIPFFT_NOT_IMPLEMENTED: return "HIPFFT_NOT_IMPLEMENTED";
        case HIPFFT_NOT_SUPPORTED: return "HIPFFT_NOT_SUPPORTED";

        // We do use default because we are not in control of these enumeration values.
        // Ideally this function is something hipFFT would provide
        default: return "<unknown hipfftResult_t value>";
    }
}

/// \brief Checks if the provided status code is \p HIPFFT_SUCCESS and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIPFFT_CHECK(condition)                                                         \
    {                                                                                   \
        const hipfftResult status = condition;                                          \
        if(status != HIPFFT_SUCCESS)                                                    \
        {                                                                               \
            std::cerr << "hipFFT error encountered: \"" << hipfftResultToString(status) \
                      << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;          \
            std::exit(error_exit_code);                                                 \
        }                                                                               \
    }

#endif // COMMON_HIPFFT_UTILS_HPP
