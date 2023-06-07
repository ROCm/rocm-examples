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

#include <iostream>

int main()
{
    // 1. Setup input data
    //
    // alpha *         A         *    x    + beta *    y    =      y
    //
    // alpha * ( 1.0  0.0  2.0 ) * ( 1.0 ) + beta * ( 4.0 ) = (  31.1 )
    //         ( 3.0  0.0  4.0 ) * ( 2.0 )          ( 5.0 ) = (  62.0 )
    //         ( 5.0  6.0  0.0 ) * ( 3.0 )          ( 6.0 ) = (  70.7 )
    //         ( 7.0  0.0  8.0 ) *                  ( 7.0 ) = ( 123.8 )

    // Set up BSR matrix
    // BSR block dimension
    constexpr rocsparse_int bsr_dim = 2;

    // Number of block rows and columns
    constexpr rocsparse_int mb = 2;
    constexpr rocsparse_int nb = 2;

    // BSR values
    constexpr double h_bsr_val[16]
        = {1.0, 3.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 5.0, 7.0, 6.0, 0.0, 0.0, 8.0, 0.0, 0.0};

    // BSR row pointers
    constexpr rocsparse_int h_bsr_row_ptr[3] = {0, 2, 4};

    // BSR column indices
    constexpr rocsparse_int h_bsr_col_ind[4] = {0, 1, 0, 1};

    // Number of non-zero blocks
    constexpr rocsparse_int nnzb = 4;

    // Block storage in column major
    constexpr rocsparse_direction dir = rocsparse_direction_column;

    // Transposition of the matrix
    constexpr rocsparse_operation trans = rocsparse_operation_none;

    // Set up scalars
    constexpr double alpha = 3.7;
    constexpr double beta  = 1.3;

    // Set up x and y vectors
    constexpr double h_x[4] = {1.0, 2.0, 3.0, 0.0};
    double           h_y[4] = {4.0, 5.0, 6.0, 7.0};

    // 2. Prepare device for calculation

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Matrix descriptor
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // Matrix info
    rocsparse_mat_info info;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));

    // 3. Offload data to device
    rocsparse_int* d_bsr_row_ptr;
    rocsparse_int* d_bsr_col_ind;
    double*        d_bsr_val;
    double*        d_x;
    double*        d_y;

    constexpr size_t x_size       = sizeof(*d_x) * nb * bsr_dim;
    constexpr size_t y_size       = sizeof(*d_y) * mb * bsr_dim;
    constexpr size_t val_size     = sizeof(*d_bsr_val) * nnzb * bsr_dim * bsr_dim;
    constexpr size_t row_ptr_size = sizeof(*d_bsr_row_ptr) * (mb + 1);
    constexpr size_t col_ind_size = sizeof(*d_bsr_col_ind) * nnzb;

    HIP_CHECK(hipMalloc((void**)&d_bsr_row_ptr, row_ptr_size));
    HIP_CHECK(hipMalloc((void**)&d_bsr_col_ind, col_ind_size));
    HIP_CHECK(hipMalloc((void**)&d_bsr_val, val_size));
    HIP_CHECK(hipMalloc((void**)&d_x, x_size));
    HIP_CHECK(hipMalloc((void**)&d_y, y_size));

    HIP_CHECK(hipMemcpy(d_bsr_row_ptr, h_bsr_row_ptr, row_ptr_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bsr_col_ind, h_bsr_col_ind, col_ind_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bsr_val, h_bsr_val, val_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x, x_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y, y_size, hipMemcpyHostToDevice));

    // 4. Call bsrmv to perform y = alpha * A x + beta * y
    ROCSPARSE_CHECK(rocsparse_dbsrmv_ex(handle,
                                        dir,
                                        trans,
                                        mb,
                                        nb,
                                        nnzb,
                                        &alpha,
                                        descr,
                                        d_bsr_val,
                                        d_bsr_row_ptr,
                                        d_bsr_col_ind,
                                        bsr_dim,
                                        info,
                                        d_x,
                                        &beta,
                                        d_y));

    // 5. Copy y to host from device
    HIP_CHECK(hipMemcpy(h_y, d_y, y_size, hipMemcpyDeviceToHost));

    // 6. Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));

    // 7. Clear device memory
    HIP_CHECK(hipFree(d_bsr_row_ptr));
    HIP_CHECK(hipFree(d_bsr_col_ind));
    HIP_CHECK(hipFree(d_bsr_val));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));

    // 8. Print result
    std::cout << "y = (";
    for(int i = 0; i < mb * bsr_dim; ++i)
    {
        std::cout << " " << h_y[i];
    }
    std::cout << ")" << std::endl;
    return 0;
}
