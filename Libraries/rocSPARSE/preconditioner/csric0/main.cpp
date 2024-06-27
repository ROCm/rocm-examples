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

    // Number of rows and columns of the input matrix.
    constexpr rocsparse_int m = 6;
    constexpr rocsparse_int n = 6;

    // Number of non-zero elements.
    constexpr rocsparse_int nnz = 36;

    // CSR values vector.
    // clang-format off
    constexpr std::array<double, nnz> h_csr_val
        = {1,  2,   3,   4,   5,   6,
           2,  5,   8,   11,  14,  17,
           3,  8,   14,  20,  26,  32,
           4,  11,  20,  30,  40,  50,
           5,  14,  26,  40,  55,  70,
           6,  17,  32,  50,  70,  91};
    // clang-format on

    // CSR row pointers vector.
    constexpr std::array<rocsparse_int, m + 1> h_csr_row_ptr = {0, 6, 12, 18, 24, 30, 36};

    // CSR column indices vector.
    constexpr std::array<rocsparse_int, nnz> h_csr_col_ind
        = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
           0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};

    // 2. Allocate device memory and offload input data to the device.
    double*        d_csr_val{};
    rocsparse_int* d_csr_row_ptr{};
    rocsparse_int* d_csr_col_ind{};

    constexpr size_t val_size     = sizeof(*d_csr_val) * nnz;
    constexpr size_t row_ptr_size = sizeof(*d_csr_row_ptr) * (m + 1);
    constexpr size_t col_ind_size = sizeof(*d_csr_col_ind) * nnz;

    HIP_CHECK(hipMalloc(&d_csr_val, val_size));
    HIP_CHECK(hipMalloc(&d_csr_row_ptr, row_ptr_size));
    HIP_CHECK(hipMalloc(&d_csr_col_ind, col_ind_size));

    HIP_CHECK(hipMemcpy(d_csr_val, h_csr_val.data(), val_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_row_ptr, h_csr_row_ptr.data(), row_ptr_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_col_ind, h_csr_col_ind.data(), col_ind_size, hipMemcpyHostToDevice));

    // 3. Initialize rocSPARSE by creating a handle.
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // 4. Prepare utility variables for rocSPARSE csric0 invocation.
    // Analysis and solve policies.
    constexpr rocsparse_analysis_policy analysis_policy = rocsparse_analysis_policy_reuse;
    constexpr rocsparse_solve_policy    solve_policy    = rocsparse_solve_policy_auto;

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
    ROCSPARSE_CHECK(rocsparse_dcsric0_buffer_size(handle,
                                                  m,
                                                  nnz,
                                                  descr,
                                                  d_csr_val,
                                                  d_csr_row_ptr,
                                                  d_csr_col_ind,
                                                  info,
                                                  &buffer_size));
    // No synchronization with the device is needed because for scalar results, when using host
    // pointer mode (the default pointer mode) this function blocks the CPU till the GPU has copied
    // the results back to the host. See rocsparse_set_pointer_mode.

    // Allocate temporary buffer.
    void* temp_buffer{};
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // 5. Perform the analysis step.
    ROCSPARSE_CHECK(rocsparse_dcsric0_analysis(handle,
                                               m,
                                               nnz,
                                               descr,
                                               d_csr_val,
                                               d_csr_row_ptr,
                                               d_csr_col_ind,
                                               info,
                                               analysis_policy,
                                               solve_policy,
                                               temp_buffer));

    // 6. Call dcsric0 to perform incomplete Cholesky factorization.
    ROCSPARSE_CHECK(rocsparse_dcsric0(handle,
                                      m,
                                      nnz,
                                      descr,
                                      d_csr_val,
                                      d_csr_row_ptr,
                                      d_csr_col_ind,
                                      info,
                                      solve_policy,
                                      temp_buffer))

    // 7. Check zero-pivots.
    rocsparse_int    pivot_position;
    rocsparse_status csric0_status = rocsparse_csric0_zero_pivot(handle, info, &pivot_position);

    int errors{};

    if(csric0_status == rocsparse_status_zero_pivot)
    {
        std::cout << "Found zero pivot in matrix row " << pivot_position << std::endl;
        errors++;
    }
    else
    {
        ROCSPARSE_CHECK(csric0_status);
    }

    // 8. Convert the resulting CSR sparse matrix to a dense matrix. Check and print the resulting matrix.
    // Host and device allocations of the result matrix for conversion routines.
    constexpr size_t           size_A = n * m;
    std::array<double, size_A> A;

    double*          d_A{};
    constexpr size_t size_bytes_A = sizeof(*d_A) * size_A;
    HIP_CHECK(hipMalloc(&d_A, size_bytes_A));

    ROCSPARSE_CHECK(
        rocsparse_dcsr2dense(handle, m, n, descr, d_csr_val, d_csr_row_ptr, d_csr_col_ind, d_A, n));

    HIP_CHECK(hipMemcpy(A.data(), d_A, size_bytes_A, hipMemcpyDeviceToHost));

    // Print the resulting L matrix and compare it with the expected result.
    // Expected L matrix in dense format.
    constexpr std::array<double, size_A> expected
        = {1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 0, 0, 1, 2, 3, 4,
           0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1};

    std::cout << "Incomplete Cholesky factorization A = L * L^H successfully computed with L "
                 "matrix: \n";

    const double eps = 1.0e5 * std::numeric_limits<double>::epsilon();
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
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
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));

    HIP_CHECK(hipFree(d_csr_val));
    HIP_CHECK(hipFree(d_csr_row_ptr));
    HIP_CHECK(hipFree(d_csr_col_ind));
    HIP_CHECK(hipFree(temp_buffer));
    HIP_CHECK(hipFree(d_A));

    // 10. Print validation result.
    return report_validation_result(errors);
}
