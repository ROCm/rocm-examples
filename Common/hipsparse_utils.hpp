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

#ifndef COMMON_HIPSPARSE_UTILS_HPP
#define COMMON_HIPSPARSE_UTILS_HPP

#include "example_utils.hpp"

#include <chrono>
#include <hipsparse/hipsparse.h>
#include <vector>

/// \brief Checks if the provided status code is \p HIPSPARSE_STATUS_SUCCESS and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIPSPARSE_CHECK(condition)                                                           \
    {                                                                                        \
        const hipsparseStatus_t status = condition;                                          \
        if(status != HIPSPARSE_STATUS_SUCCESS)                                               \
        {                                                                                    \
            std::cerr << "hipSPARSE error encountered: \"" << status << "\" at " << __FILE__ \
                      << ':' << __LINE__ << std::endl;                                       \
            std::exit(error_exit_code);                                                      \
        }                                                                                    \
    }

/// \brief CPU Timer(in microsecond): synchronize with the default device and return wall time
inline double get_time_us(void)
{
    std::ignore = hipDeviceSynchronize();
    auto now    = std::chrono::steady_clock::now();
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
}

/// \brief  Generate 2D laplacian on unit square in CSR format
template<typename I, typename J, typename T>
J gen_2d_laplacian(int                  ndim,
                   std::vector<I>&      rowptr,
                   std::vector<J>&      col,
                   std::vector<T>&      val,
                   hipsparseIndexBase_t idx_base)
{
    if(ndim == 0)
    {
        return 0;
    }

    J n       = ndim * ndim;
    I nnz_mat = n * 5 - ndim * 4;

    rowptr.resize(n + 1);
    col.resize(nnz_mat);
    val.resize(nnz_mat);

    I nnz = 0;

    // Fill local arrays
    for(int i = 0; i < ndim; ++i)
    {
        for(int j = 0; j < ndim; ++j)
        {
            J idx       = i * ndim + j;
            rowptr[idx] = nnz + idx_base;
            // if no upper boundary element, connect with upper neighbor
            if(i != 0)
            {
                col[nnz] = idx - ndim + idx_base;
                val[nnz] = T(-1.0);
                ++nnz;
            }
            // if no left boundary element, connect with left neighbor
            if(j != 0)
            {
                col[nnz] = idx - 1 + idx_base;
                val[nnz] = T(-1.0);
                ++nnz;
            }
            // element itself
            col[nnz] = idx + idx_base;
            val[nnz] = T(4.0);
            ++nnz;
            // if no right boundary element, connect with right neighbor
            if(j != ndim - 1)
            {
                col[nnz] = idx + 1 + idx_base;
                val[nnz] = T(-1.0);
                ++nnz;
            }
            // if no lower boundary element, connect with lower neighbor
            if(i != ndim - 1)
            {
                col[nnz] = idx + ndim + idx_base;
                val[nnz] = T(-1.0);
                ++nnz;
            }
        }
    }
    rowptr[n] = nnz + idx_base;

    return n;
}

/// \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX);
// for complex number, the real/imag part would be initialized with the same value
template<typename T>
void hipsparseInit(std::vector<T>& A, int M, int N)
{
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            A[i + j] = T(rand() % 10 + 1);
        }
    }
}

#endif // COMMON_HIPSPARSE_UTILS_HPP
