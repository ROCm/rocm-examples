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

/// \brief Generate an identity matrix.
/// The identity matrix is a $m \times n$ matrix with ones in the main diagonal and zeros elsewhere.
template<typename T>
void generate_identity_matrix(T* A, int m, int n, size_t lda)
{
    for(rocblas_int i = 0; i < m; ++i)
    {
        for(rocblas_int j = 0; j < n; ++j)
        {
            A[i + j * lda] = T(i == j);
        }
    }
}

/// \brief Multiply an $A$ matrix ($m \times k$) with a $B$ matrix ($k \times n$) as:
/// $C := \alpha \cdot A \cdot B + \beta \cdot C$
template<typename T>
void multiply_matrices(T   alpha,
                       T   beta,
                       int m,
                       int n,
                       int k,
                       T*  A,
                       int stride1_a,
                       int stride2_a,
                       T*  B,
                       int stride1_b,
                       int stride2_b,
                       T*  C,
                       int stride_c)
{
    for(rocblas_int i1 = 0; i1 < m; ++i1)
    {
        for(rocblas_int i2 = 0; i2 < n; ++i2)
        {
            T t = T(0.0);
            for(rocblas_int i3 = 0; i3 < k; ++i3)
            {
                t += A[i1 * stride1_a + i3 * stride2_a] * B[i3 * stride1_b + i2 * stride2_b];
            }
            C[i1 + i2 * stride_c] = beta * C[i1 + i2 * stride_c] + alpha * t;
        }
    }
}

#endif // COMMON_ROCBLAS_UTILS_HPP
