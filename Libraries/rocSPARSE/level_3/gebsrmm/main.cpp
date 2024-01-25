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

#include "rocsparse_utils.hpp"

#include <rocsparse/rocsparse.h>

#include <hip/hip_runtime.h>

#include <array>
#include <cstdio>
#include <iostream>

int main()
{
    // 1. Set up input data
    // Number of rows and columns of the input matrices.
    constexpr rocsparse_int m = 4;
    constexpr rocsparse_int k = 6;
    constexpr rocsparse_int n = 10;

    // Sparse matrix A (m x k)
    //     ( 1 2 0 | 3 0 0 )
    //     ( 0 4 5 | 0 0 0 )   ( A_{00} | A_{01} )
    // A = (-------+-------) = (--------+--------)
    //     ( 0 0 0 | 7 8 0 )   (    O   | A_{11} )
    //     ( 0 0 0 | 2 4 1 )

    // GEBSR block row and column dimensions.
    constexpr rocsparse_int bsr_row_dim = 2;
    constexpr rocsparse_int bsr_col_dim = 3;

    // Number of rows and columns of the block matrix.
    constexpr rocsparse_int mb = (m + bsr_row_dim - 1) / bsr_row_dim;
    constexpr rocsparse_int kb = (k + bsr_col_dim - 1) / bsr_col_dim;

    // Padded dimensions of input matrix.
    constexpr size_t m_padded = mb * bsr_row_dim;
    constexpr size_t k_padded = kb * bsr_col_dim;

    // Number of non-zero block entries
    constexpr rocsparse_int nnzb = 3;

    // GEBSR row pointers
    constexpr std::array<rocsparse_int, mb + 1> h_bsr_row_ptr = {0, 2, 3};

    // GEBSR column indices
    constexpr std::array<rocsparse_int, nnzb> h_bsr_col_ind = {0, 1, 1};

    // GEBSR values
    constexpr rocsparse_int elements_of_val = nnzb * bsr_row_dim * bsr_col_dim;
    // clang-format off
    constexpr std::array<double, elements_of_val>
        h_bsr_val{1.0, 2.0, 0.0, 0.0, 4.0, 5.0,  // A_{00}
                  3.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // A_{01}
                  7.0, 8.0, 0.0, 2.0, 4.0, 1.0}; // A_{10};
    // clang-format on

    // Storage scheme of the GEBSR blocks.
    constexpr rocsparse_direction dir = rocsparse_direction_row;

    // Transposition of the matrix
    constexpr rocsparse_operation trans_A = rocsparse_operation_none;

    // Dense matrix B (k x n)
    //     (  9  11  13  15  17  10  12  14  16  18 )
    //     (  8  10   1  10   6  11   7   3  12  17 )
    // B = ( 11  11   0   4   6  12   2   9  13   2 )
    //     ( 15   3   2   3   8   1   2   4   6   6 )
    //     (  2   5   7   0   1  15   9   4  10   1 )
    //     (  7  12  12   1  12   5   1  11   1  14 )

    // Matrix B elements in column-major
    constexpr rocsparse_int elements_of_B = k_padded * n;
    // clang-format off
    constexpr std::array<double, elements_of_B>
        h_B{ 9,  8, 11, 15,  2,  7, 
            11, 10, 11,  3,  5, 12,
            13,  1,  0,  2,  7, 12,
            15, 10,  4,  3,  0,  1,
            17,  6,  6,  8,  1, 12,
            10, 11, 12,  1, 15,  5,
            12,  7,  2,  2,  9,  1,
            14,  3,  9,  4,  4, 11,
            16, 12, 13,  6, 10,  1,
            18, 17,  2,  6,  1, 14};
    // clang-format on

    // Transposition of the matrix
    constexpr rocsparse_operation trans_B = rocsparse_operation_none;

    // Initialize a dense matrix C (m x n)
    constexpr rocsparse_int           elements_of_C = m_padded * n;
    std::array<double, elements_of_C> h_C{};

    // Scalar alpha and beta
    constexpr double alpha = 1.0;
    constexpr double beta  = 0.0;

    // 2. Prepare device for calculation
    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Matrix descriptor
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // 3. Offload data to device
    rocsparse_int* d_bsr_row_ptr;
    rocsparse_int* d_bsr_col_ind;
    double*        d_bsr_val;
    double*        d_B;
    double*        d_C;

    constexpr size_t size_B       = sizeof(*d_B) * elements_of_B;
    constexpr size_t size_C       = sizeof(*d_C) * elements_of_C;
    constexpr size_t size_val     = sizeof(*d_bsr_val) * elements_of_val;
    constexpr size_t size_row_ptr = sizeof(*d_bsr_row_ptr) * (mb + 1);
    constexpr size_t size_col_ind = sizeof(*d_bsr_col_ind) * nnzb;

    HIP_CHECK(hipMalloc(&d_bsr_row_ptr, size_row_ptr));
    HIP_CHECK(hipMalloc(&d_bsr_col_ind, size_col_ind));
    HIP_CHECK(hipMalloc(&d_bsr_val, size_val));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));

    HIP_CHECK(hipMemcpy(d_bsr_row_ptr, h_bsr_row_ptr.data(), size_row_ptr, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bsr_col_ind, h_bsr_col_ind.data(), size_col_ind, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bsr_val, h_bsr_val.data(), size_val, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C.data(), size_C, hipMemcpyHostToDevice));

    // 4. Call gebsrmm to perform C = alpha * A' * B' + beta * C
    ROCSPARSE_CHECK(rocsparse_dgebsrmm(handle,
                                       dir,
                                       trans_A,
                                       trans_B,
                                       mb,
                                       n,
                                       kb,
                                       nnzb,
                                       &alpha,
                                       descr,
                                       d_bsr_val,
                                       d_bsr_row_ptr,
                                       d_bsr_col_ind,
                                       bsr_row_dim,
                                       bsr_col_dim,
                                       d_B,
                                       k_padded,
                                       &beta,
                                       d_C,
                                       m_padded));

    // 5. Copy C to host from device
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, size_C, hipMemcpyDeviceToHost));

    // 6. Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));

    // 7. Clear device memory
    HIP_CHECK(hipFree(d_bsr_row_ptr));
    HIP_CHECK(hipFree(d_bsr_col_ind));
    HIP_CHECK(hipFree(d_bsr_val));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    // 8. Print result.
    //     (  70  40  21  44  53  35  32  32  58  70 )
    // C = (  87  95   4  60  54 104  38  57 113  78 )
    //     ( 121  61  70  21  64 127  86  60 122  50 )
    //     (  45  38  44   7  32  67  41  35  53  30 )
    std::cout << "C =" << std::endl;

    for(rocsparse_int i = 0; i < m; ++i)
    {
        std::cout << "    (";
        for(rocsparse_int j = 0; j < n; ++j)
        {
            std::printf("%5.0lf", h_C[i + j * m_padded]);
        }

        std::cout << " )" << std::endl;
    }

    return 0;
}
