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
    //   = (  2   3   4   5   6   7   )
    //     (  4   8   11  14  17  20  )
    //     (  6   13  20  26  32  38  )
    //     (  8   18  29  40  50  60  )
    //     (  10  23  38  54  70  85  )
    //     (  12  28  47  68  90  112 )

    // Number of rows and columns of the input matrix.
    constexpr rocsparse_int n = 6;

    // Number of non-zero elements.
    constexpr rocsparse_int nnz = 36;

    // CSR values vector.
    // clang-format off
    constexpr std::array<double, nnz> h_csr_val
        = {2,   3,   4,   5,   6,   7,
           4,   8,   11,  14,  17,  20,
           6,   13,  20,  26,  32,  38,
           8,   18,  29,  40,  50,  60,
           10,  23,  38,  54,  70,  85,
           12,  28,  47,  68,  90,  112};
    // clang-format on

    // CSR row pointers vector.
    constexpr std::array<rocsparse_int, n + 1> h_csr_row_ptr = {0, 6, 12, 18, 24, 30, 36};

    // CSR column indices vector.
    constexpr std::array<rocsparse_int, nnz> h_csr_col_ind
        = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
           0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};

    // 2. Allocate device memory and offload input data to the device.
    rocsparse_int* d_csr_row_ptr{};
    rocsparse_int* d_csr_col_ind{};
    double*        d_csr_val{};

    constexpr size_t size_csr_row_ptr = sizeof(*d_csr_row_ptr) * (n + 1);
    constexpr size_t size_csr_col_ind = sizeof(*d_csr_col_ind) * nnz;
    constexpr size_t size_csr_val     = sizeof(*d_csr_val) * nnz;

    HIP_CHECK(hipMalloc(&d_csr_row_ptr, size_csr_row_ptr));
    HIP_CHECK(hipMalloc(&d_csr_col_ind, size_csr_col_ind));
    HIP_CHECK(hipMalloc(&d_csr_val, size_csr_val));

    HIP_CHECK(
        hipMemcpy(d_csr_row_ptr, h_csr_row_ptr.data(), size_csr_row_ptr, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_csr_col_ind, h_csr_col_ind.data(), size_csr_col_ind, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_val, h_csr_val.data(), size_csr_val, hipMemcpyHostToDevice));

    // 3. Initialize rocSPARSE by creating a handle.
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // 4. Prepare utility variables for rocSPARSE csrilu0 invocation.
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
    ROCSPARSE_CHECK(rocsparse_dcsrilu0_buffer_size(handle,
                                                   n,
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
    ROCSPARSE_CHECK(rocsparse_dcsrilu0_analysis(handle,
                                                n,
                                                nnz,
                                                descr,
                                                d_csr_val,
                                                d_csr_row_ptr,
                                                d_csr_col_ind,
                                                info,
                                                analysis_policy,
                                                solve_policy,
                                                temp_buffer));

    // 6. Call dcsrilu0 to perform incomplete LU factorization.
    ROCSPARSE_CHECK(rocsparse_dcsrilu0(handle,
                                       n,
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
    rocsparse_status csrilu0_status = rocsparse_csrilu0_zero_pivot(handle, info, &pivot_position);

    int errors{};

    if(csrilu0_status == rocsparse_status_zero_pivot)
    {
        std::cout << "Found zero pivot in matrix row " << pivot_position << std::endl;
        errors++;
    }
    else
    {
        ROCSPARSE_CHECK(csrilu0_status);
    }

    // 8. Convert the resulting CSR sparse matrix to a dense matrix. Check and print the resulting matrix.
    // Host and device allocations of the result matrix for conversion routines.
    constexpr size_t           size_A = n * n;
    std::array<double, size_A> A;

    double*          d_A{};
    constexpr size_t size_bytes_A = sizeof(*d_A) * size_A;
    HIP_CHECK(hipMalloc(&d_A, size_bytes_A));

    // 8b. Convert CSR sparse matrix to dense.
    ROCSPARSE_CHECK(
        rocsparse_dcsr2dense(handle, n, n, descr, d_csr_val, d_csr_row_ptr, d_csr_col_ind, d_A, n));

    HIP_CHECK(hipMemcpy(A.data(), d_A, size_bytes_A, hipMemcpyDeviceToHost));

    // 8c. Print the resulting L and U matrices and compare it with the expected results.
    // Expected L and U matrices in dense format.
    constexpr std::array<double, size_A> L_expected
        = {1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 0, 0, 1, 2, 3, 4,
           0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1};
    constexpr std::array<double, size_A> U_expected
        = {2, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 4, 3, 2, 0, 0, 0,
           5, 4, 3, 2, 0, 0, 6, 5, 4, 3, 2, 0, 7, 6, 5, 4, 3, 2};

    std::cout << "Incomplete LU factorization A ~= L * U successfully computed with L "
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
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));

    HIP_CHECK(hipFree(d_csr_row_ptr));
    HIP_CHECK(hipFree(d_csr_col_ind));
    HIP_CHECK(hipFree(d_csr_val));
    HIP_CHECK(hipFree(temp_buffer));
    HIP_CHECK(hipFree(d_A));

    // 10. Print validation result.
    return report_validation_result(errors);
}
