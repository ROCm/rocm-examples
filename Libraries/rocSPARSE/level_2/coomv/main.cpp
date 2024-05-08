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

#include "example_utils.hpp"
#include "rocsparse_utils.hpp"

#include <rocsparse/rocsparse.h>

#include <hip/hip_runtime.h>

#include <array>
#include <iostream>

int main()
{
    // 1. Set up input data
    //
    // alpha *      op(A)        *    x    + beta *    y    =      y
    //
    //   3.7 * ( 1.0  0.0  2.0 ) * ( 1.0 ) +  1.3 * ( 4.0 ) = (  31.1 )
    //         ( 3.0  0.0  4.0 ) * ( 2.0 )          ( 5.0 ) = (  62.0 )
    //         ( 5.0  6.0  0.0 ) * ( 3.0 )          ( 6.0 ) = (  70.7 )
    //         ( 7.0  0.0  8.0 ) *                  ( 7.0 ) = ( 123.8 )

    // Set up COO matrix

    // Number of rows and columns
    constexpr rocsparse_int m = 4;
    constexpr rocsparse_int n = 3;

    // Number of non-zero elements
    constexpr rocsparse_int nnz = 8;

    // COO values
    constexpr std::array<double, nnz> h_coo_val  = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    constexpr std::array<double, m>   expected_y = {31.1, 62.0, 70.7, 123.8};

    // COO row indices
    constexpr std::array<rocsparse_int, nnz> h_coo_row_ind = {0, 0, 1, 1, 2, 2, 3, 3};

    // COO column indices
    constexpr std::array<rocsparse_int, nnz> h_coo_col_ind = {0, 2, 0, 2, 0, 1, 0, 2};

    // Transposition of the matrix
    constexpr rocsparse_operation trans = rocsparse_operation_none;

    // Set up scalars
    constexpr double alpha = 3.7;
    constexpr double beta  = 1.3;

    // Set up x and y vectors
    constexpr std::array<double, n> h_x = {1.0, 2.0, 3.0};
    std::array<double, m>           h_y = {4.0, 5.0, 6.0, 7.0};

    // 2. Prepare device for calculation

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // Matrix descriptor
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // 3. Offload data to device
    rocsparse_int* d_coo_row_ind;
    rocsparse_int* d_coo_col_ind;
    double*        d_coo_val{};
    double*        d_x{};
    double*        d_y{};

    constexpr size_t x_size   = sizeof(*d_x) * n;
    constexpr size_t y_size   = sizeof(*d_y) * m;
    constexpr size_t val_size = sizeof(*d_coo_val) * nnz;
    constexpr size_t ind_size = sizeof(*d_coo_col_ind) * nnz;

    HIP_CHECK(hipMalloc(&d_coo_row_ind, ind_size));
    HIP_CHECK(hipMalloc(&d_coo_col_ind, ind_size));
    HIP_CHECK(hipMalloc(&d_coo_val, val_size));
    HIP_CHECK(hipMalloc(&d_x, x_size));
    HIP_CHECK(hipMalloc(&d_y, y_size));

    HIP_CHECK(hipMemcpy(d_coo_row_ind, h_coo_row_ind.data(), ind_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_coo_col_ind, h_coo_col_ind.data(), ind_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_coo_val, h_coo_val.data(), val_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y.data(), y_size, hipMemcpyHostToDevice));

    // 4. Call coomv to perform y = alpha * op(A) * x + beta * y
    // This function is non blocking and executed asynchronously with respect to the host.
    ROCSPARSE_CHECK(rocsparse_dcoomv(handle,
                                     trans,
                                     m,
                                     n,
                                     nnz,
                                     &alpha,
                                     descr,
                                     d_coo_val,
                                     d_coo_row_ind,
                                     d_coo_col_ind,
                                     d_x,
                                     &beta,
                                     d_y));

    // 5. Copy y to host from device. This call synchronizes with the host.
    HIP_CHECK(hipMemcpy(h_y.data(), d_y, y_size, hipMemcpyDeviceToHost));

    // 6. Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));

    // 7. Clear device memory
    HIP_CHECK(hipFree(d_coo_row_ind));
    HIP_CHECK(hipFree(d_coo_col_ind));
    HIP_CHECK(hipFree(d_coo_val));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));

    // 8. Print results to standard output.
    std::cout << "Solution successfully computed: ";

    std::cout << "y = " << format_range(std::begin(h_y), std::end(h_y)) << std::endl;

    // Compare solution with the expected result.
    int          errors{};
    const double eps = 1.0e5 * std::numeric_limits<double>::epsilon();
    for(size_t i = 0; i < h_y.size(); ++i)
    {
        errors += std::fabs(h_y[i] - expected_y[i]) > eps;
    }

    // Print validation result.
    return report_validation_result(errors);
}
