// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef COMMON_ROCBLAS_UTILS_HPP
#define COMMON_ROCBLAS_UTILS_HPP

#include <rocblas/rocblas.h>

#include "example_utils.hpp"

#include <iostream>

/// \brief Checks if the provided status code is \p rocblas_status_success and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define ROCBLAS_CHECK(condition)                                                             \
    {                                                                                        \
        const rocblas_status status = condition;                                             \
        if(status != rocblas_status_success)                                                 \
        {                                                                                    \
            std::cerr << "rocBLAS error encountered: \"" << rocblas_status_to_string(status) \
                      << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;               \
            std::exit(error_exit_code);                                                      \
        }                                                                                    \
    }

#endif // COMMON_ROCBLAS_UTILS_HPP
