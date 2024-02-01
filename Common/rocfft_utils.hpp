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

#ifndef COMMON_ROCFFT_UTILS_HPP
#define COMMON_ROCFFT_UTILS_HPP

#include "example_utils.hpp"

#include <hip/hip_complex.h>
#include <rocfft/rocfft.h>

#include <iostream>
#include <numeric>

/// \brief Converts a \p rocfft_status variable to its correspondent string.
inline const char* rocfftStatusToString(rocfft_status status)
{
    switch(status)
    {
        case rocfft_status_success: return "rocfft_status_success";
        case rocfft_status_failure: return "rocfft_status_failure";
        case rocfft_status_invalid_arg_value: return "rocfft_status_invalid_arg_value";
        case rocfft_status_invalid_dimensions: return "rocfft_status_invalid_dimensions";
        case rocfft_status_invalid_array_type: return "rocfft_status_invalid_array_type";
        case rocfft_status_invalid_strides: return "rocfft_status_invalid_strides";
        case rocfft_status_invalid_distance: return "rocfft_status_invalid_distance";
        case rocfft_status_invalid_offset: return "rocfft_status_invalid_offset";
        case rocfft_status_invalid_work_buffer: return "rocfft_status_invalid_work_buffer";

        // We do use default because we are not in control of these enumeration values.
        // Ideally this function is something rocFFT would provide
        default: return "<unknown rocfft_status value>";
    }
}

/// \brief Checks if the provided status code is \p rocfft_status_success and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define ROCFFT_CHECK(condition)                                                         \
    {                                                                                   \
        const rocfft_status status = condition;                                         \
        if(status != rocfft_status_success)                                             \
        {                                                                               \
            std::cerr << "rocFFT error encountered: \"" << rocfftStatusToString(status) \
                      << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;          \
            std::exit(error_exit_code);                                                 \
        }                                                                               \
    }

std::ostream& operator<<(std::ostream& stream, hipDoubleComplex c)
{
    stream << "(" << c.x << "," << c.y << ")";
    return stream;
}

/// \brief Increment the index (column-major) for looping over arbitrary dimensional loops with
/// dimensions \p length. Returns a bool if end of increment has been reached.
template<class T1, class T2>
bool increment_cm(std::vector<T1>& index, const std::vector<T2>& length)
{
    for(unsigned int idim = 0; idim < length.size(); ++idim)
    {
        if(index[idim] < length[idim])
        {
            if(++index[idim] == length[idim])
            {
                index[idim] = 0;
                continue;
            }
            break;
        }
    }
    // End the loop when we get back to the start:
    return !std::all_of(index.begin(), index.end(), [](int i) { return i == 0; });
}

#endif // COMMON_ROCFFT_UTILS_HPP
