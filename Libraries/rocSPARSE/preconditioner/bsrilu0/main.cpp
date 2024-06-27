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

int main()
{
    // 1. Set up input data.
    //
    // A = (  1  0  0  0  0  0 ) * (  2  3  4  5  6  7 )
    //     (  2  1  0  0  0  0 )   (  0  2  3  4  5  6 )
    //     (  3  2  1  0  0  0 )   (  0  0  2  3  4  5 )
    //     (  4  3  2  1  0  0 )   (  0  0  0  2  3  4 )
    //     (  5  4  3  2  1  0 )   (  0  0  0  0  2  3 )
    //     (  6  5  4  3  2  1 )   (  0  0  0  0  0  2 )
    //
    //   = (  2  3   4   5   6   7   )
    //     (  4  8   11  14  17  20  )
    //     (  6  13  20  26  32  38  )
    //     (  8  18  29  40  50  60  )
    //     (  10 23  38  54  70  85  )
    //     (  12 28  47  68  90  112 )
    //
    //   = (  2  3  | 4   5  | 6   7   )
    //     (  4  8  | 11  14 | 17  20  )
    //     (---------------------------)
    //     (  6  13 | 20  26 | 32  38  )
    //     (  8  18 | 29  40 | 50  60  )
    //     (---------------------------)
    //     (  10 23 | 38  54 | 70  85  )
    //     (  12 28 | 47  68 | 90  112 )
    //
    //   = ( A_{00} | A_{O1} | A_{O2} )
    //     (--------------------------)
    //     ( A_{10} | A_{11} | A_{12} )
    //     (--------------------------)
    //     ( A_{20} | A_{21} | A_{22} )

    // BSR block dimension.
    constexpr rocsparse_int bsr_dim = 2;

    // Number of rows and columns of the input matrix.
    constexpr rocsparse_int n = 6;

    // Number of rows and columns of the block matrix.
    constexpr rocsparse_int nb = (n + bsr_dim - 1) / bsr_dim;

    // Number of non-zero blocks.
    constexpr rocsparse_int nnzb = 9;

    // BSR row pointers vector.
    constexpr std::array<rocsparse_int, nb + 1> h_bsr_row_ptr = {0, 3, 6, 9};

    // BSR column indices vector.
    constexpr std::array<rocsparse_int, nnzb> h_bsr_col_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};

    // BSR values vector.
    constexpr std::array<double, (nnzb * bsr_dim * bsr_dim)> h_bsr_val
        = {2,  4,  3,  8 /*A_{00}*/,  4,  11, 5,  14 /*A_{01}*/, 6,  17, 7,  20 /*A_{02}*/,
           6,  8,  13, 18 /*A_{10}*/, 20, 29, 26, 40 /*A_{11}*/, 32, 50, 38, 60 /*A_{12}*/,
           10, 12, 23, 28 /*A_{20}*/, 38, 47, 54, 68 /*A_{21}*/, 70, 90, 85, 112 /*A_{22}*/};

    // Storage scheme of the BSR blocks.
    constexpr rocsparse_direction dir = rocsparse_direction_column;

    // Analysis and solve policies.
    constexpr rocsparse_analysis_policy analysis_policy = rocsparse_analysis_policy_reuse;
    constexpr rocsparse_solve_policy    solve_policy    = rocsparse_solve_policy_auto;

    // 2. Allocate device memory and offload input data to the device.
    rocsparse_int* d_bsr_row_ptr{};
    rocsparse_int* d_bsr_col_ind{};
    double*        d_bsr_val{};

    constexpr size_t size_bsr_row_ptr = sizeof(*d_bsr_row_ptr) * (nb + 1);
    constexpr size_t size_bsr_col_ind = sizeof(*d_bsr_col_ind) * nnzb;
    constexpr size_t size_bsr_val     = sizeof(*d_bsr_val) * nnzb * bsr_dim * bsr_dim;

    HIP_CHECK(hipMalloc(&d_bsr_row_ptr, size_bsr_row_ptr));
    HIP_CHECK(hipMalloc(&d_bsr_col_ind, size_bsr_col_ind));
    HIP_CHECK(hipMalloc(&d_bsr_val, size_bsr_val));

    HIP_CHECK(
        hipMemcpy(d_bsr_row_ptr, h_bsr_row_ptr.data(), size_bsr_row_ptr, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_bsr_col_ind, h_bsr_col_ind.data(), size_bsr_col_ind, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bsr_val, h_bsr_val.data(), size_bsr_val, hipMemcpyHostToDevice));

    // 3. Initialize rocSPARSE by creating a handle.
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // 4. Prepare utility variables for rocSPARSE bsrilu0 invocation.
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
    ROCSPARSE_CHECK(rocsparse_dbsrilu0_buffer_size(handle,
                                                   dir,
                                                   nb,
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
    ROCSPARSE_CHECK(rocsparse_dbsrilu0_analysis(handle,
                                                dir,
                                                nb,
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

    // 6. Call dbsrilu0 to perform incomplete LU factorization.
    ROCSPARSE_CHECK(rocsparse_dbsrilu0(handle,
                                       dir,
                                       nb,
                                       nnzb,
                                       descr,
                                       d_bsr_val,
                                       d_bsr_row_ptr,
                                       d_bsr_col_ind,
                                       bsr_dim,
                                       info,
                                       solve_policy,
                                       temp_buffer))

    // 7. Check zero-pivots.
    // No synchronization with the device is needed because for scalar results, when using host
    // pointer mode (the default pointer mode) this function blocks the CPU till the GPU has copied
    // the results back to the host. See rocsparse_set_pointer_mode.
    rocsparse_int    pivot_position;
    rocsparse_status bsrilu0_status = rocsparse_bsrilu0_zero_pivot(handle, info, &pivot_position);

    int errors{};

    if(bsrilu0_status == rocsparse_status_zero_pivot)
    {
        std::cout << "Found zero pivot in matrix row " << pivot_position << std::endl;
        errors++;
    }
    else
    {
        ROCSPARSE_CHECK(bsrilu0_status);
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

    constexpr rocsparse_int nnze = nnzb * bsr_dim * bsr_dim; /*non-zero elements*/

    rocsparse_int* d_csr_row_ptr{};
    rocsparse_int* d_csr_col_ind{};
    double*        d_csr_val{};

    constexpr size_t size_csr_row_ptr = sizeof(*d_csr_row_ptr) * (n + 1);
    constexpr size_t size_csr_col_ind = sizeof(*d_csr_col_ind) * nnze;
    constexpr size_t size_csr_val     = sizeof(*d_csr_val) * nnze;

    HIP_CHECK(hipMalloc(&d_csr_row_ptr, size_csr_row_ptr));
    HIP_CHECK(hipMalloc(&d_csr_col_ind, size_csr_col_ind));
    HIP_CHECK(hipMalloc(&d_csr_val, size_csr_val));

    ROCSPARSE_CHECK(rocsparse_dbsr2csr(handle,
                                       dir,
                                       nb,
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

    // 8c. Print the resulting L and U matrices and compare it with the expected results.
    // Expected L and U matrices in dense format.
    constexpr std::array<double, size_A> L_expected
        = {1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 0, 0, 1, 2, 3, 4,
           0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1};
    constexpr std::array<double, size_A> U_expected
        = {2, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 4, 3, 2, 0, 0, 0,
           5, 4, 3, 2, 0, 0, 6, 5, 4, 3, 2, 0, 7, 6, 5, 4, 3, 2};

    std::cout << "Incomplete LU factorization A = L * U successfully computed with L "
                 "matrix: \n";

    // L matrix is stored in the lower part of A. The diagonal is not stored as it is known
    // that all the diagonal elements are the multiplicative identity (1 in this case).
    const double eps = 1.0e5 * std::numeric_limits<double>::epsilon();
    for(rocsparse_int i = 0; i < n; ++i)
    {
        for(rocsparse_int j = 0; j < n; ++j)
        {
            const double val = (j < i) ? A[j * n + i] : (j == i);
            std::cout << std::setw(3) << val;

            errors += std::fabs(val - L_expected[j * n + i]) > eps;
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    std::cout << "and U matrix: \n";

    for(rocsparse_int i = 0; i < n; ++i)
    {
        for(rocsparse_int j = 0; j < n; ++j)
        {
            const double val = (j >= i) ? A[j * n + i] : 0;
            std::cout << std::setw(3) << val;

            errors += std::fabs(val - U_expected[j * n + i]) > eps;
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

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
