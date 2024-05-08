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
#include <cmath>
#include <iostream>
#include <limits>

int main()
{
    // 1. Setup input data.

    //  alpha * op(A) * x
    // + beta * y =
    //            = 1.0 * ( 4.0  0.0  0.0  0.0 ) * ( 1.0 ) - 1.0 * ( 4.0 )
    //                    ( 4.0  3.0  0.0  0.0 )   ( 2.0 )         ( 5.0 )
    //                    ( 4.0  3.0  2.0  0.0 )   ( 3.0 )         ( 6.0 )
    //                    ( 4.0  3.0  2.0  1.0 )   ( 4.0 )         ( 7.0 )
    //
    //            = 1.0 * ( 4.0  0.0  0.0 | 0.0  0.0  0.0 ) * ( 1.0 ) - 1.0 * ( 4.0 )
    //                    ( 4.0  3.0  0.0 | 0.0  0.0  0.0 )   ( 2.0 )         ( 5.0 )
    //                    ( ----------------------------- )   ( 3.0 )         ( 6.0 )
    //                    ( 4.0  3.0  2.0 | 0.0  0.0  0.0 )   ( 4.0 )         ( 7.0 )
    //                    ( 4.0  3.0  2.0 | 1.0  0.0  0.0 )   ( 0.0 )
    //                                                        ( 0.0 )
    //
    //            = 1.0 * ( A_{00} |   O    ) * ( 1.0 ) - 1.0 * ( 4.0 )
    //                    (-----------------)   ( 2.0 )         ( 5.0 )
    //                    ( A_{10} | A_{11} )   ( 3.0 )         ( 6.0 )
    //                                          ( 4.0 )         ( 7.0 )
    //                                          ( 0.0 )
    //                                          ( 0.0 )

    // GEBSR block row and column dimensions.
    constexpr rocsparse_int bsr_row_dim = 2;
    constexpr rocsparse_int bsr_col_dim = 3;

    // Number of rows and columns of the input matrix.
    constexpr rocsparse_int m = 4;
    constexpr rocsparse_int n = 4;

    // Number of rows and columns of the block matrix.
    constexpr rocsparse_int mb = (m + bsr_row_dim - 1) / bsr_row_dim;
    constexpr rocsparse_int nb = (n + bsr_col_dim - 1) / bsr_col_dim;

    // Padded dimensions of input matrix.
    constexpr size_t m_padded = mb * bsr_row_dim;
    constexpr size_t n_padded = nb * bsr_col_dim;

    // Number of non-zero blocks.
    constexpr rocsparse_int nnzb = 3;

    // GEBSR values (in row-major) vector.
    // clang-format off
    constexpr std::array<double, nnzb * bsr_row_dim * bsr_col_dim>
        h_bsr_val{4.0, 0.0, 0.0,
                  4.0, 3.0, 0.0,  // op(A)_{00}

                  4.0, 3.0, 2.0,
                  4.0, 3.0, 2.0,  // op(A)_{10}

                  0.0, 0.0, 0.0,
                  1.0, 0.0, 0.0}; // op(A)_{11}
    // clang-format on

    // GEBSR row pointers.
    constexpr std::array<rocsparse_int, mb + 1> h_bsr_row_ptr = {0, 1, 3};

    // GEBSR column indices.
    constexpr std::array<rocsparse_int, nnzb> h_bsr_col_ind = {0, 0, 1};

    // Storage scheme of the BSR blocks.
    constexpr rocsparse_direction dir = rocsparse_direction_row;

    // Operation applied to the matrix.
    constexpr rocsparse_operation trans = rocsparse_operation_none;

    // Set up scalars.
    constexpr double alpha = 1.0;
    constexpr double beta  = -1.0;

    // Set up x and y vectors and expected solution.
    constexpr std::array<double, n_padded> h_x{1.0, 2.0, 3.0, 4.0, 0.0, 0.0};
    std::array<double, m_padded>           h_y{4.0, 5.0, 6.0, 7.0};
    constexpr std::array<double, m_padded> expected_y{0.0, 5.0, 10.0, 13.0};

    // 2. Allocate device memory and offload input data to device.
    rocsparse_int* d_bsr_row_ptr;
    rocsparse_int* d_bsr_col_ind;
    double*        d_bsr_val{};
    double*        d_x{};
    double*        d_y{};

    constexpr size_t x_size       = sizeof(*d_x) * n_padded;
    constexpr size_t y_size       = sizeof(*d_y) * m_padded;
    constexpr size_t val_size     = sizeof(*d_bsr_val) * nnzb * bsr_row_dim * bsr_col_dim;
    constexpr size_t row_ptr_size = sizeof(*d_bsr_row_ptr) * (mb + 1);
    constexpr size_t col_ind_size = sizeof(*d_bsr_col_ind) * nnzb;

    HIP_CHECK(hipMalloc((void**)&d_bsr_row_ptr, row_ptr_size));
    HIP_CHECK(hipMalloc((void**)&d_bsr_col_ind, col_ind_size));
    HIP_CHECK(hipMalloc((void**)&d_bsr_val, val_size));
    HIP_CHECK(hipMalloc((void**)&d_x, x_size));
    HIP_CHECK(hipMalloc((void**)&d_y, y_size));

    HIP_CHECK(hipMemcpy(d_bsr_row_ptr, h_bsr_row_ptr.data(), row_ptr_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bsr_col_ind, h_bsr_col_ind.data(), col_ind_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bsr_val, h_bsr_val.data(), val_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y.data(), y_size, hipMemcpyHostToDevice));

    // 3. Initialize rocSPARSE by creating a handle.
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // 4. Prepare utility variables for rocSPARSE gebsrmv invocation.
    // Matrix descriptor.
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // 5. Call gebsrmv to perform y = alpha * op(A) * x + beta * y.
    // This function is non blocking and executed asynchronously with respect to the host.
    ROCSPARSE_CHECK(rocsparse_dgebsrmv(handle,
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
                                       bsr_row_dim,
                                       bsr_col_dim,
                                       d_x,
                                       &beta,
                                       d_y));

    // 6. Copy solution to host from device. This call synchronizes with the host.
    HIP_CHECK(hipMemcpy(h_y.data(), d_y, y_size, hipMemcpyDeviceToHost));

    // 7. Clear rocSPARSE allocations on device.
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));

    // Clear device arrays.
    HIP_CHECK(hipFree(d_bsr_row_ptr));
    HIP_CHECK(hipFree(d_bsr_col_ind));
    HIP_CHECK(hipFree(d_bsr_val));
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
