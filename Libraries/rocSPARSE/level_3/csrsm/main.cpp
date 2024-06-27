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

#include <hip/hip_runtime_api.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

int main()
{
    // 1. Set up input data.
    // Solve op_a(A) * op_b(X) = alpha * op_b(B), with triangular sparse matrix A, and a dense
    // matrix B containing several right-hand sides (b_1, ..., b_nrhs) as columns.
    //
    //         A       *            X                           = alpha *           B
    //  ( 1  0  0  0 ) * ( X_{00} X_{01} X_{02} X_{03} X_{04} ) =   1   * ( 1   2   4   0    7 )
    //  ( 2  3  0  0 )   ( X_{10} X_{11} X_{12} X_{13} X_{14} )           ( 2  13   8  18   14 )
    //  ( 4  5  6  0 )   ( X_{20} X_{21} X_{22} X_{23} X_{24} )           ( 4  23  46  30   76 )
    //  ( 7  0  8  9 )   ( X_{30} X_{31} X_{32} X_{33} X_{34} )           ( 7  14  68   0  194 )
    //
    // The solution is given as a dense matrix X. Expected result:
    //
    //  X = ( 1  2  4  0  7 )
    //      ( 0  3  0  6  0 )
    //      ( 0  0  5  0  8 )
    //      ( 0  0  0  0  9 )

    // Number of rows and columns of the input matrix A.
    constexpr rocsparse_int n = 4;

    // Number of right-hand sides (columns of B and X).
    constexpr rocsparse_int nrhs = 5;

    // Number of non-zero elements.
    constexpr rocsparse_int nnz = 9;

    // CSR row pointers.
    constexpr std::array<rocsparse_int, n + 1> h_csr_row_ptr{0, 1, 3, 6, 9};

    // CSR column indices.
    constexpr std::array<rocsparse_int, nnz> h_csr_col_ind{0, 0, 1, 0, 1, 2, 0, 2, 3};

    // CSR values.
    constexpr std::array<double, nnz> h_csr_val{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    // Scalar alpha.
    constexpr double alpha = 1.0;

    // Right-hand side (and solution matrix) leading dimension.
    // The leading dimension can be used to adjust the alignment of the rows of the matrix.
    // It must be greater than or equal to the amount of rows.
    constexpr rocsparse_int ldb = n;

    // Right-hand side values, stored in column-major order.
    // clang-format off
    constexpr std::array<double, ldb * nrhs> h_B{1.0,
                                                 2.0,
                                                 4.0,
                                                 7.0, // B_0
                                                 2.0,
                                                 13.0,
                                                 23.0,
                                                 14.0, // B_1
                                                 4.0,
                                                 8.0,
                                                 46.0,
                                                 68.0, // B_2
                                                 0.0,
                                                 18.0,
                                                 30.0,
                                                 0.0, // B_3
                                                 7.0,
                                                 14.0,
                                                 76.0,
                                                 194.0}; // B_4
    // clang-format on

    // Operations applied to the input matrices.
    constexpr rocsparse_operation trans_A = rocsparse_operation_none;
    constexpr rocsparse_operation trans_B = rocsparse_operation_none;

    // 2. Allocate device memory and copy input data to device.
    rocsparse_int* d_csr_row_ptr{};
    rocsparse_int* d_csr_col_ind{};
    double*        d_csr_val{};
    double*        d_B{};

    constexpr size_t size_csr_row_ptr = sizeof(*d_csr_row_ptr) * (n + 1);
    constexpr size_t size_csr_col_ind = sizeof(*d_csr_col_ind) * nnz;
    constexpr size_t size_csr_val     = sizeof(*d_csr_val) * nnz;
    constexpr size_t size_B           = sizeof(*d_B) * ldb * nrhs;

    HIP_CHECK(hipMalloc(&d_csr_row_ptr, size_csr_row_ptr));
    HIP_CHECK(hipMalloc(&d_csr_col_ind, size_csr_col_ind));
    HIP_CHECK(hipMalloc(&d_csr_val, size_csr_val));
    HIP_CHECK(hipMalloc(&d_B, size_B));

    HIP_CHECK(
        hipMemcpy(d_csr_row_ptr, h_csr_row_ptr.data(), size_csr_row_ptr, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_csr_col_ind, h_csr_col_ind.data(), size_csr_col_ind, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_val, h_csr_val.data(), size_csr_val, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));

    // 3. Initialize rocSPARSE by creating a handle.
    // Create rocSPARSE handle.
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // 4. Prepare utility variables for rocSPARSE csrsm invocation.
    // Create matrix descriptor.
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // Set matrix fill mode.
    ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower));

    // Set matrix diagonal type.
    ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_non_unit));

    // Create matrix info structure.
    rocsparse_mat_info info;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));

    // Indexing: zero based.
    constexpr rocsparse_index_base idx_base = rocsparse_index_base_zero;

    // Analysis policy.
    constexpr rocsparse_analysis_policy analysis_policy = rocsparse_analysis_policy_reuse;

    // Solve policy.
    constexpr rocsparse_solve_policy solve_policy = rocsparse_solve_policy_auto;

    // Obtain required buffer size.
    // This function is non blocking and executed asynchronously with respect to the host.
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_dcsrsm_buffer_size(handle,
                                                 trans_A,
                                                 trans_B,
                                                 n,
                                                 nrhs,
                                                 nnz,
                                                 &alpha,
                                                 descr,
                                                 d_csr_val,
                                                 d_csr_row_ptr,
                                                 d_csr_col_ind,
                                                 d_B,
                                                 ldb,
                                                 info,
                                                 solve_policy,
                                                 &buffer_size));
    // No synchronization with the device is needed because for scalar results, when using host
    // pointer mode (the default pointer mode) this function blocks the CPU till the GPU has copied
    // the results back to the host. See rocsparse_set_pointer_mode.

    // Allocate temporary buffer.
    void* temp_buffer{};
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // 5. Perform analysis step.
    ROCSPARSE_CHECK(rocsparse_dcsrsm_analysis(handle,
                                              trans_A,
                                              trans_B,
                                              n,
                                              nrhs,
                                              nnz,
                                              &alpha,
                                              descr,
                                              d_csr_val,
                                              d_csr_row_ptr,
                                              d_csr_col_ind,
                                              d_B,
                                              ldb,
                                              info,
                                              analysis_policy,
                                              solve_policy,
                                              temp_buffer));

    // 6. Sort CSR matrix before calling the csrsm_solve function.
    rocsparse_int* perm;
    HIP_CHECK(hipMalloc(&perm, size_csr_col_ind));
    ROCSPARSE_CHECK(rocsparse_create_identity_permutation(handle, nnz, perm));

    // Query the required buffer size in bytes and allocate a temporary buffer for sorting.
    // This function is non blocking and executed asynchronously with respect to the host.
    size_t sort_buffer_size;
    void*  sort_temp_buffer{};
    ROCSPARSE_CHECK(rocsparse_csrsort_buffer_size(handle,
                                                  n,
                                                  n,
                                                  nnz,
                                                  d_csr_row_ptr,
                                                  d_csr_col_ind,
                                                  &sort_buffer_size));
    // No synchronization with the device is needed because for scalar results, when using host
    // pointer mode (the default pointer mode) this function blocks the CPU till the GPU has copied
    // the results back to the host. See rocsparse_set_pointer_mode.

    HIP_CHECK(hipMalloc(&sort_temp_buffer, sort_buffer_size));

    // Sort the CSR matrix applying the identity permutation.
    ROCSPARSE_CHECK(rocsparse_csrsort(handle,
                                      n,
                                      n,
                                      nnz,
                                      descr,
                                      d_csr_row_ptr,
                                      d_csr_col_ind,
                                      perm,
                                      sort_temp_buffer));

    // Gather sorted values array.
    ROCSPARSE_CHECK(rocsparse_dgthr(handle, nnz, d_csr_val, d_csr_val, perm, idx_base));

    // 7. Call to rocSPARSE csrsm to solve the linear system.
    // This function is non blocking and executed asynchronously with respect to the host.
    ROCSPARSE_CHECK(rocsparse_dcsrsm_solve(handle,
                                           trans_A,
                                           trans_B,
                                           n,
                                           nrhs,
                                           nnz,
                                           &alpha,
                                           descr,
                                           d_csr_val,
                                           d_csr_row_ptr,
                                           d_csr_col_ind,
                                           d_B,
                                           ldb,
                                           info,
                                           solve_policy,
                                           temp_buffer));

    // 8. Check results.
    // Check for zero pivots.
    // No synchronization with the device is needed because for scalar results, when using host
    // pointer mode (the default pointer mode) this function blocks the CPU till the GPU has copied
    // the results back to the host. See rocsparse_set_pointer_mode.
    rocsparse_int    pivot_position;
    rocsparse_status csrsm_status = rocsparse_csrsm_zero_pivot(handle, info, &pivot_position);

    int errors{};

    if(csrsm_status == rocsparse_status_zero_pivot)
    {
        std::cout << "Found zero pivot in matrix row " << pivot_position << std::endl;
        ++errors;
    }
    else
    {
        ROCSPARSE_CHECK(csrsm_status);
    }

    // Define expected result, stored in column-major ordering.
    constexpr std::array<double, nrhs * n> expected_X{1.0, 0.0, 0.0,
                                                      0.0, // X_0
                                                      2.0, 3.0, 0.0,
                                                      0.0, // X_1
                                                      4.0, 0.0, 5.0,
                                                      0.0, // X_2
                                                      0.0, 6.0, 0.0,
                                                      0.0, // X_3
                                                      7.0, 0.0, 8.0, 9.0}; // X_4

    // Allocate array for solution on host and copy result from device.
    std::array<double, ldb * nrhs> h_X{};
    HIP_CHECK(hipMemcpy(h_X.data(), d_B, size_B, hipMemcpyDeviceToHost));

    // Print solution and compare with expected results.
    std::cout << "Solution successfully computed: " << std::endl;
    std::cout << "X = " << std::endl;
    constexpr double eps = std::numeric_limits<double>::epsilon();
    for(rocsparse_int i = 0; i < n; ++i)
    {
        std::cout << "(";
        for(rocsparse_int j = 0; j < nrhs; ++j)
        {
            std::printf("%9.2lf", h_X[i * ldb + j]);
            errors += std::fabs(h_X[i * ldb + j] - expected_X[i * n + j]) > eps;
        }
        std::cout << "  )" << std::endl;
    }

    // 9. Free rocSPARSE resources and device memory.
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));

    HIP_CHECK(hipFree(perm));
    HIP_CHECK(hipFree(sort_temp_buffer));
    HIP_CHECK(hipFree(temp_buffer));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_csr_val));
    HIP_CHECK(hipFree(d_csr_col_ind));
    HIP_CHECK(hipFree(d_csr_row_ptr));

    // 10. Print validation result.
    return report_validation_result(errors);
}
