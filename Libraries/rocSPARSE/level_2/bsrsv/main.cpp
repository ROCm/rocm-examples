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

#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

int main()
{
    // 1. Setup input data.

    // A = ( 1.0  0.0  0.0  0.0 )
    //     ( 2.0  3.0  0.0  0.0 )
    //     ( 4.0  5.0  6.0  0.0 )
    //     ( 7.0  0.0  8.0  9.0 )
    //
    //   = ( 1.0  0.0 | 0.0  0.0 )
    //     ( 2.0  3.0 | 0.0  0.0 )
    //     (---------------------)
    //     ( 4.0  5.0 | 6.0  0.0 )
    //     ( 7.0  0.0 | 8.0  9.0 )
    //
    //   = ( A_{00} |   O    )
    //     (-----------------)
    //     ( A_{10} | A_{11} )

    // BSR block dimension.
    constexpr rocsparse_int bsr_dim = 2;

    // Number of rows and columns of the input matrix.
    constexpr rocsparse_int m = 4;
    constexpr rocsparse_int n = 4;

    // Number of rows and columns of the block matrix.
    constexpr rocsparse_int mb = (m + bsr_dim - 1) / bsr_dim;

    // Number of non-zero blocks.
    constexpr rocsparse_int nnzb = 3;

    // BSR row pointers vector.
    constexpr std::array<rocsparse_int, mb + 1> h_bsr_row_ptr = {0, 1, 3};

    // BSR column indices vector.
    constexpr std::array<rocsparse_int, nnzb> h_bsr_col_ind = {0, 0, 1};

    // BSR values vector.
    constexpr std::array<double, (nnzb * bsr_dim * bsr_dim)> h_bsr_val = {1.0,
                                                                          2.0,
                                                                          0.0,
                                                                          3.0, /*A_{00}*/
                                                                          4.0,
                                                                          7.0,
                                                                          5.0,
                                                                          0.0, /*A_{10}*/
                                                                          6.0,
                                                                          8.0,
                                                                          0.0,
                                                                          9.0}; /*A_{11}*/

    // Storage scheme of the BSR blocks.
    constexpr rocsparse_direction dir = rocsparse_direction_column;

    // Operation applied to the matrix.
    constexpr rocsparse_operation trans = rocsparse_operation_none;

    // Analysis and solve policies.
    constexpr rocsparse_analysis_policy analysis_policy = rocsparse_analysis_policy_reuse;
    constexpr rocsparse_solve_policy    solve_policy    = rocsparse_solve_policy_auto;

    // Scalar alpha.
    constexpr double alpha = 1.0;

    // Host vectors for the right hand side and solution of the linear system.
    std::array<double, m> h_x;
    std::iota(h_x.begin(), h_x.end(), 1.0);
    std::array<double, n> h_y;

    // Expected solution.
    constexpr std::array<double, n> expected_y = {1.0, 0.0, -1.0 / 6.0, -5.0 / 27.0};

    // 2. Allocate device memory and offload input data to device.
    rocsparse_int* d_bsr_row_ptr{};
    rocsparse_int* d_bsr_col_ind{};
    double*        d_bsr_val{};
    double*        d_x{};
    double*        d_y{};

    HIP_CHECK(hipMalloc(&d_bsr_row_ptr, sizeof(*d_bsr_row_ptr) * (mb + 1)));
    HIP_CHECK(hipMalloc(&d_bsr_col_ind, sizeof(*d_bsr_col_ind) * nnzb));
    HIP_CHECK(hipMalloc(&d_bsr_val, sizeof(*d_bsr_val) * nnzb * bsr_dim * bsr_dim));
    HIP_CHECK(hipMalloc(&d_x, sizeof(*d_x) * m));
    HIP_CHECK(hipMalloc(&d_y, sizeof(*d_y) * n));

    HIP_CHECK(hipMemcpy(d_bsr_row_ptr,
                        h_bsr_row_ptr.data(),
                        sizeof(*d_bsr_row_ptr) * (mb + 1),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bsr_col_ind,
                        h_bsr_col_ind.data(),
                        sizeof(*d_bsr_col_ind) * nnzb,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bsr_val,
                        h_bsr_val.data(),
                        sizeof(*d_bsr_val) * nnzb * bsr_dim * bsr_dim,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), sizeof(*d_x) * m, hipMemcpyHostToDevice));

    // 3. Initialize rocSPARSE by creating a handle.
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // 4. Prepare utility variables for rocSPARSE bsrmv invocation.
    // Matrix descriptor.
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // Matrix fill mode.
    ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower));

    // Matrix diagonal type.
    ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_non_unit));

    // Matrix info structure.
    rocsparse_mat_info info;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));

    // Obtain required buffer size in bytes for analysis and solve stages.
    // This function is non blocking and executed asynchronously with respect to the host.
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_dbsrsv_buffer_size(handle,
                                                 dir,
                                                 trans,
                                                 mb,
                                                 nnzb,
                                                 descr,
                                                 d_bsr_val,
                                                 d_bsr_row_ptr,
                                                 d_bsr_col_ind,
                                                 bsr_dim,
                                                 info,
                                                 &buffer_size));
    // No synchronization with the device is needed because for scalar results, when using host
    // pointer mode (the default pointer mode) this function blocks the CPU till the GPU has copied
    // the results back to the host. See rocsparse_set_pointer_mode.

    // Allocate temporary buffer.
    void* temp_buffer{};
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // 5. Perform analysis step.
    ROCSPARSE_CHECK(rocsparse_dbsrsv_analysis(handle,
                                              dir,
                                              trans,
                                              mb,
                                              nnzb,
                                              descr,
                                              d_bsr_val,
                                              d_bsr_row_ptr,
                                              d_bsr_col_ind,
                                              bsr_dim,
                                              info,
                                              analysis_policy,
                                              solve_policy,
                                              temp_buffer));

    // 6. Perform triangular solve op(A) * y = alpha * x.
    ROCSPARSE_CHECK(rocsparse_dbsrsv_solve(handle,
                                           dir,
                                           trans,
                                           mb,
                                           nnzb,
                                           &alpha,
                                           descr,
                                           d_bsr_val,
                                           d_bsr_row_ptr,
                                           d_bsr_col_ind,
                                           bsr_dim,
                                           info,
                                           d_x,
                                           d_y,
                                           solve_policy,
                                           temp_buffer));

    // 7. Check results obtained.
    rocsparse_int    position;
    rocsparse_status bsrsv_status = rocsparse_bsrsv_zero_pivot(handle, info, &position);

    int errors{};

    if(bsrsv_status == rocsparse_status_zero_pivot)
    {
        std::cout << "Found zero pivot in matrix row " << position << std::endl;
        errors++;
    }
    else
    {
        ROCSPARSE_CHECK(bsrsv_status);
    }

    std::cout << "Solution successfully computed: ";

    HIP_CHECK(hipMemcpy(h_y.data(), d_y, sizeof(*d_y) * n, hipMemcpyDeviceToHost));

    std::cout << "y = " << format_range(h_y.begin(), h_y.end()) << std::endl;

    // Compare solution with the expected result.
    const double eps = 1.0e5 * std::numeric_limits<double>::epsilon();
    for(size_t i = 0; i < h_y.size(); i++)
    {
        errors += std::fabs(h_y[i] - expected_y[i]) > eps;
    }

    // 8. Free rocSPARSE resources and device memory.
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));

    HIP_CHECK(hipFree(d_bsr_row_ptr));
    HIP_CHECK(hipFree(d_bsr_col_ind));
    HIP_CHECK(hipFree(d_bsr_val));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(temp_buffer));

    // 9. Print validation result.
    return report_validation_result(errors);
}
