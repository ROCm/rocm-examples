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
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

int main()
{
    // 1. Set up input data.

    // A = L * L^H
    //
    //   = (  1  0  0  0  0  0 ) * (  1  2  3  4  5  6 )
    //     (  2  1  0  0  0  0 )   (  0  1  2  3  4  5 )
    //     (  3  2  1  0  0  0 )   (  0  0  1  2  3  4 )
    //     (  4  3  2  1  0  0 )   (  0  0  0  1  2  3 )
    //     (  5  4  3  2  1  0 )   (  0  0  0  0  1  2 )
    //     (  6  5  4  3  2  1 )   (  0  0  0  0  0  1 )
    //
    //   = (  1  2   3   4   5   6  )
    //     (  2  5   8   11  14  17 )
    //     (  3  8   14  20  26  32 )
    //     (  4  11  20  30  40  50 )
    //     (  5  14  26  40  55  70 )
    //     (  6  17  32  50  70  91 )
    //
    //   = (  1  2  | 3   4  | 5   6  )
    //     (  2  5  | 8   11 | 14  17 )
    //     (--------------------------)
    //     (  3  8  | 14  20 | 26  32 )
    //     (  4  11 | 20  30 | 40  50 )
    //     (--------------------------)
    //     (  5  14 | 26  40 | 55  70 )
    //     (  6  17 | 32  50 | 70  91 )
    //
    //   = ( A_{00} | A_{O1} | A_{O2} )
    //     (--------------------------)
    //     ( A_{10} | A_{11} | A_{12} )
    //     (--------------------------)
    //     ( A_{20} | A_{21} | A_{22} )

    // BSR block dimension.
    constexpr rocsparse_int bsr_dim = 2;

    // Number of rows and columns of the input matrix.
    constexpr rocsparse_int m = 6;
    constexpr rocsparse_int n = 6;

    // Number of rows and columns of the block matrix.
    constexpr rocsparse_int mb = (m + bsr_dim - 1) / bsr_dim;
    constexpr rocsparse_int nb = (n + bsr_dim - 1) / bsr_dim;

    // Number of non-zero blocks.
    constexpr rocsparse_int nnzb = 9;

    // BSR row pointers vector.
    constexpr std::array<rocsparse_int, mb + 1> h_bsr_row_ptr = {0, 3, 6, 9};

    // BSR column indices vector.
    constexpr std::array<rocsparse_int, nnzb> h_bsr_col_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};

    // BSR values vector.
    constexpr std::array<double, (nnzb * bsr_dim * bsr_dim)> h_bsr_val
        = {1, 2,  2, 5 /*A_{00}*/,  3,  4,  8,  11 /*A_{01}*/, 5,  6,  14, 17 /*A_{02}*/,
           3, 8,  4, 11 /*A_{10}*/, 14, 20, 20, 30 /*A_{11}*/, 26, 32, 40, 50 /*A_{12}*/,
           5, 14, 6, 17 /*A_{20}*/, 26, 40, 32, 50 /*A_{21}*/, 55, 70, 70, 91 /*A_{22}*/};

    // Storage scheme of the BSR blocks.
    constexpr rocsparse_direction dir = rocsparse_direction_row;

    // Analysis and solve policies.
    constexpr rocsparse_analysis_policy analysis_policy = rocsparse_analysis_policy_reuse;
    constexpr rocsparse_solve_policy    solve_policy    = rocsparse_solve_policy_auto;

    // 2. Allocate device memory and offload input data to the device.
    rocsparse_int* d_bsr_row_ptr{};
    rocsparse_int* d_bsr_col_ind{};
    double*        d_bsr_val{};

    HIP_CHECK(hipMalloc(&d_bsr_row_ptr, sizeof(*d_bsr_row_ptr) * (mb + 1)));
    HIP_CHECK(hipMalloc(&d_bsr_col_ind, sizeof(*d_bsr_col_ind) * nnzb));
    HIP_CHECK(hipMalloc(&d_bsr_val, sizeof(*d_bsr_val) * nnzb * bsr_dim * bsr_dim));

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

    // 3. Initialize rocSPARSE by creating a handle.
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // 4. Prepare utility variables for rocSPARSE bsric0 invocation.
    // Matrix descriptor.
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // Matrix fill mode.
    ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower));

    // Matrix info structure.
    rocsparse_mat_info info;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));

    // Obtain the required buffer size in bytes for analysis and solve stages.
    // This function is non blocking and executed asynchronously with respect to the host.
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_dbsric0_buffer_size(handle,
                                                  dir,
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

    // 5. Perform the analysis step.
    ROCSPARSE_CHECK(rocsparse_dbsric0_analysis(handle,
                                               dir,
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

    // 6. Call dbsric0 to perform incomplete Cholesky factorization.
    ROCSPARSE_CHECK(rocsparse_dbsric0(handle,
                                      dir,
                                      mb,
                                      nnzb,
                                      descr,
                                      d_bsr_val,
                                      d_bsr_row_ptr,
                                      d_bsr_col_ind,
                                      bsr_dim,
                                      info,
                                      solve_policy,
                                      temp_buffer));

    // 7. Check zero-pivots.
    rocsparse_int    pivot_position;
    rocsparse_status bsric0_status = rocsparse_bsric0_zero_pivot(handle, info, &pivot_position);

    int errors{};

    if(bsric0_status == rocsparse_status_zero_pivot)
    {
        std::cout << "Found zero pivot in matrix row " << pivot_position << std::endl;
        errors++;
    }
    else
    {
        ROCSPARSE_CHECK(bsric0_status);
    }

    // 8. Convert the resulting BSR sparse matrix to a dense matrix. Check and print the resulting matrix.
    // Host and device allocations of the result matrix for conversion routines.
    constexpr size_t           size_A = n * n;
    std::array<double, size_A> A;

    double*          d_A{};
    constexpr size_t size_bytes_A = sizeof(*d_A) * size_A;
    HIP_CHECK(hipMalloc(&d_A, size_bytes_A));

    // 8a. Convert BSR sparse matrix to CSR format.
    rocsparse_mat_descr csr_descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&csr_descr));
    ROCSPARSE_CHECK(rocsparse_set_mat_type(csr_descr, rocsparse_matrix_type_general));

    constexpr rocsparse_int nnze = size_A; /*non-zero elements*/

    rocsparse_int* d_csr_row_ptr{};
    rocsparse_int* d_csr_col_ind{};
    double*        d_csr_val{};

    HIP_CHECK(hipMalloc(&d_csr_row_ptr, sizeof(*d_csr_row_ptr) * (n + 1)));
    HIP_CHECK(hipMalloc(&d_csr_col_ind, sizeof(*d_csr_col_ind) * nnze));
    HIP_CHECK(hipMalloc(&d_csr_val, sizeof(*d_csr_val) * nnze));

    ROCSPARSE_CHECK(rocsparse_dbsr2csr(handle,
                                       dir,
                                       mb,
                                       nb,
                                       descr,
                                       d_bsr_val,
                                       d_bsr_row_ptr,
                                       d_bsr_col_ind,
                                       bsr_dim,
                                       csr_descr,
                                       d_csr_val,
                                       d_csr_row_ptr,
                                       d_csr_col_ind));

    // 8b. Convert CSR sparse matrix to dense.
    ROCSPARSE_CHECK(rocsparse_dcsr2dense(handle,
                                         n,
                                         n,
                                         csr_descr,
                                         d_csr_val,
                                         d_csr_row_ptr,
                                         d_csr_col_ind,
                                         d_A,
                                         n));

    HIP_CHECK(hipMemcpy(A.data(), d_A, size_bytes_A, hipMemcpyDeviceToHost));

    // 8c. Print the resulting L matrix and compare it with the expected result.
    // Expected L matrix in dense format.
    constexpr std::array<double, size_A> expected
        = {1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 0, 0, 1, 2, 3, 4,
           0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1};

    std::cout << "Incomplete Cholesky factorization A = L * L^H successfully computed with L "
                 "matrix: \n";

    const double eps = 1.0e5 * std::numeric_limits<double>::epsilon();
    for(rocsparse_int i = 0; i < n; ++i)
    {
        for(rocsparse_int j = 0; j < n; ++j)
        {
            const double val = (j <= i) ? A[j * n + i] : 0;
            std::cout << std::setw(3) << val;

            errors += std::fabs(val - expected[j * n + i]) > eps;
        }
        std::cout << std::endl;
    }

    // 9. Free rocSPARSE resources and device memory.
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(csr_descr));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));

    HIP_CHECK(hipFree(d_bsr_row_ptr));
    HIP_CHECK(hipFree(d_bsr_col_ind));
    HIP_CHECK(hipFree(d_bsr_val));
    HIP_CHECK(hipFree(d_csr_row_ptr));
    HIP_CHECK(hipFree(d_csr_col_ind));
    HIP_CHECK(hipFree(d_csr_val));
    HIP_CHECK(hipFree(temp_buffer));
    HIP_CHECK(hipFree(d_A));

    // 10. Print validation result.
    return report_validation_result(errors);
}
