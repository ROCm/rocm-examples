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

#include "rocsparse_utils.hpp"

#include <rocsparse/rocsparse.h>

#include <hip/hip_runtime.h>

#include <array>
#include <iostream>

int main()
{
    // 1. Set up input data.
    //
    // unmasked:
    //
    // alpha *      op(A)        *    x    + beta *    y    =      y
    //
    //  3.7  * ( 1.0  0.0  2.0 ) * ( 1.0 ) +  1.3 * ( 4.0 ) = (  31.1 )
    //         ( 3.0  0.0  4.0 ) * ( 2.0 )          ( 5.0 ) = (  62.0 )
    //         ( 5.0  6.0  0.0 ) * ( 3.0 )          ( 6.0 ) = (  70.7 )
    //         ( 7.0  0.0  8.0 ) *                  ( 7.0 ) = ( 123.8 )
    //
    // masked:
    //
    //  3.7  * ( 0.0  0.0  0.0 ) * ( 1.0 ) +  1.3 * ( 0.0 ) + ( 4.0 ) = (   4.0 )
    //         ( 0.0  0.0  0.0 ) * ( 2.0 )          ( 0.0 ) + ( 5.0 ) = (   5.0 )
    //         ( 5.0  6.0  0.0 ) * ( 3.0 )          ( 6.0 ) + ( 0.0 ) = (  70.7 )
    //         ( 7.0  0.0  0.0 ) *                  ( 7.0 ) + ( 0.0 ) = (  35.0 )

    // Set up BSR matrix.
    // BSR block dimension
    constexpr rocsparse_int bsr_dim = 2;

    // Number of rows and columns of the input matrix
    constexpr rocsparse_int m = 4;
    constexpr rocsparse_int n = 3;

    // Number of rows and columns of the block matrix
    constexpr rocsparse_int mb = (m + bsr_dim - 1) / bsr_dim;
    constexpr rocsparse_int nb = (n + bsr_dim - 1) / bsr_dim;

    // Padded dimensions of input matrix
    constexpr size_t n_padded = nb * bsr_dim;
    constexpr size_t m_padded = mb * bsr_dim;

    // Number of non-zero blocks
    constexpr rocsparse_int nnzb = 4;

    // Number of non-zero block values
    constexpr rocsparse_int number_of_values = nnzb * bsr_dim * bsr_dim;

    // BSR values
    constexpr std::array<double, number_of_values> h_bsr_val
        = {1.0, 3.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 5.0, 7.0, 6.0, 0.0, 0.0, 8.0, 0.0, 0.0};

    // BSRX row and end pointers
    constexpr std::array<rocsparse_int, mb> h_bsrx_row_ptr = {0, 2};
    constexpr std::array<rocsparse_int, mb> h_bsrx_end_ptr = {2, 3};

    // BSR column indices
    constexpr std::array<rocsparse_int, nnzb> h_bsr_col_ind = {0, 1, 0, 1};

    // Block storage in column major
    constexpr rocsparse_direction dir = rocsparse_direction_column;

    // Transposition of the matrix
    constexpr rocsparse_operation trans = rocsparse_operation_none;

    // Set up scalars.
    constexpr double alpha = 3.7;
    constexpr double beta  = 1.3;

    // Set up x and y vectors.
    constexpr std::array<double, n_padded> h_x = {1.0, 2.0, 3.0, 0.0};
    std::array<double, m_padded>           h_y = {4.0, 5.0, 6.0, 7.0};

    // Set up mask pointers.
    constexpr rocsparse_int                          mask_length    = 1;
    constexpr std::array<rocsparse_int, mask_length> h_mask_row_ptr = {1};

    // 2. Prepare device for calculation.

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // Matrix descriptor
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // 3. Offload data to device.
    rocsparse_int* d_bsrx_row_ptr;
    rocsparse_int* d_bsrx_end_ptr;
    rocsparse_int* d_bsr_col_ind;
    rocsparse_int* d_mask_row_ptr;
    double*        d_bsr_val;
    double*        d_x;
    double*        d_y;

    constexpr size_t x_size       = sizeof(*d_x) * n_padded;
    constexpr size_t y_size       = sizeof(*d_y) * m_padded;
    constexpr size_t val_size     = sizeof(*d_bsr_val) * number_of_values;
    constexpr size_t ptr_size     = sizeof(*d_bsrx_row_ptr) * mb;
    constexpr size_t col_ind_size = sizeof(*d_bsr_col_ind) * nnzb;
    constexpr size_t mask_size    = sizeof(*d_mask_row_ptr) * mask_length;

    HIP_CHECK(hipMalloc(&d_bsrx_row_ptr, ptr_size));
    HIP_CHECK(hipMalloc(&d_bsrx_end_ptr, ptr_size));
    HIP_CHECK(hipMalloc(&d_bsr_col_ind, col_ind_size));
    HIP_CHECK(hipMalloc(&d_mask_row_ptr, mask_size));
    HIP_CHECK(hipMalloc(&d_bsr_val, val_size));
    HIP_CHECK(hipMalloc(&d_x, x_size));
    HIP_CHECK(hipMalloc(&d_y, y_size));

    HIP_CHECK(hipMemcpy(d_bsrx_row_ptr, h_bsrx_row_ptr.data(), ptr_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bsrx_end_ptr, h_bsrx_end_ptr.data(), ptr_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bsr_col_ind, h_bsr_col_ind.data(), col_ind_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_mask_row_ptr, h_mask_row_ptr.data(), mask_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bsr_val, h_bsr_val.data(), val_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y.data(), y_size, hipMemcpyHostToDevice));

    // 4. Call masked matrix-vector multiplication.
    // This function is non blocking and executed asynchronously with respect to the host.
    ROCSPARSE_CHECK(rocsparse_dbsrxmv(handle,
                                      dir,
                                      trans,
                                      mask_length,
                                      mb,
                                      nb,
                                      nnzb,
                                      &alpha,
                                      descr,
                                      d_bsr_val,
                                      d_mask_row_ptr,
                                      d_bsrx_row_ptr,
                                      d_bsrx_end_ptr,
                                      d_bsr_col_ind,
                                      bsr_dim,
                                      d_x,
                                      &beta,
                                      d_y));

    // 5. Copy y to host from device. This call synchronizes with the host.
    HIP_CHECK(hipMemcpy(h_y.data(), d_y, y_size, hipMemcpyDeviceToHost));

    // 6. Free rocSPARSE resources and device memory.
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));

    HIP_CHECK(hipFree(d_bsrx_row_ptr));
    HIP_CHECK(hipFree(d_bsrx_end_ptr));
    HIP_CHECK(hipFree(d_mask_row_ptr));
    HIP_CHECK(hipFree(d_bsr_col_ind));
    HIP_CHECK(hipFree(d_bsr_val));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));

    // 7. Print result: (4 5 70.7 35)
    std::cout << "y = " << format_range(std::begin(h_y), std::end(h_y)) << std::endl;

    return 0;
}
