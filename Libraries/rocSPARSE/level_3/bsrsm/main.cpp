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
    // Solve op_a(A) * op_b(X) = alpha * op_b(B) for X, with triangular sparse matrix A,
    // and a dense matrix B containing several right-hand sides {b_1, ..., b_nrhs}.
    //
    //         A       *            X                    = alpha *       B
    //  ( 1  0  0  0 ) * ( X_{00} X_{01} X_{02} X_{03} ) =   27  * ( 1 5 9  13 )
    //  ( 2  3  0  0 )   ( X_{10} X_{11} X_{12} X_{13} )           ( 2 6 10 14 )
    //  ( 4  5  6  0 )   ( X_{20} X_{21} X_{22} X_{23} )           ( 3 7 11 15 )
    //  ( 7  0  8  9 )   ( X_{30} X_{31} X_{32} X_{33} )           ( 4 8 12 16 )
    //
    //  A in BSR format:
    //  ( 1 0 | 0 0 )
    //  ( 2 3 | 0 0 )
    //  ( ----------)
    //  ( 4 5 | 6 0 )
    //  ( 7 0 | 8 9 )
    //
    //  ( A_{00} |    0   )
    //  ( ----------------)
    //  ( A_{10} | A_{11} )
    //
    // Expected result for X:
    // ( 27.0       135.0      243.0       351.0 )
    // (  0.0       -36.0      -72.0      -108.0 )
    // ( -4.5       -28.5      -52.5       -76.5 )
    // ( -5.0   -167.0/3.0   -319.0/3.0   -157.0 )

    // BSR block dimension.
    constexpr rocsparse_int bsr_dim = 2;

    // Number of rows and columns of the input matrix.
    constexpr rocsparse_int n = 4;

    // Number of rows and columns of the block matrix.
    constexpr rocsparse_int nb = (n + bsr_dim - 1) / bsr_dim;

    // Number of non-zero blocks.
    constexpr rocsparse_int nnzb = 3;

    // BSR row pointers.
    constexpr std::array<rocsparse_int, nb + 1> h_bsr_row_ptr{0, 1, 3};

    // BSR column indices.
    constexpr std::array<rocsparse_int, nnzb> h_bsr_col_ind{0, 0, 1};

    // BSR values.
    // The blocks are stored in column-major order.
    // clang-format off
    constexpr std::array<double, nnzb * bsr_dim * bsr_dim>
        h_bsr_val{1.0, 2.0, 0.0, 3.0,  // A_{00}
                  4.0, 7.0, 5.0, 0.0,  // A_{10}
                  6.0, 8.0, 0.0, 9.0}; // A_{11}
    // clang-format on

    // Storage scheme of the BSR blocks.
    constexpr rocsparse_direction dir = rocsparse_direction_column;

    // Scalar alpha.
    constexpr double alpha = 27.0;

    // Right-hand side and solution matrix leading dimensions.
    // The leading dimension can be used to adjust the alignment of the rows of the matrix.
    // It must be greater than or equal to the amount of rows.
    constexpr rocsparse_int ldb = nb * bsr_dim;
    constexpr rocsparse_int ldx = nb * bsr_dim;

    // Number of right-hand sides (columns of B and X).
    constexpr rocsparse_int nrhs = 4;

    // Right-hand side values, stored in column-major order.
    // clang-format off
    constexpr std::array<double, ldb * nrhs>
        h_B{ 1.0,  2.0,  3.0,  4.0,  // B_0
             5.0,  6.0,  7.0,  8.0,  // B_1
             9.0, 10.0, 11.0, 12.0,  // B_2
            13.0, 14.0, 15.0, 16.0}; // B_3
    // clang-format on

    // Operations applied to the matrices.
    // Operation on X and B must be the same.
    constexpr rocsparse_operation trans_A = rocsparse_operation_none;
    constexpr rocsparse_operation trans_X = rocsparse_operation_none;

    // 2. Allocate device memory and copy input data to device.
    rocsparse_int* d_bsr_row_ptr{};
    rocsparse_int* d_bsr_col_ind{};
    double*        d_bsr_val{};
    double*        d_B{};
    double*        d_X{};

    constexpr size_t bsr_row_ptr_size = sizeof(*d_bsr_row_ptr) * (nb + 1);
    constexpr size_t bsr_col_ind_size = sizeof(*d_bsr_col_ind) * nnzb;
    constexpr size_t bsr_val_size     = sizeof(*d_bsr_val) * nnzb * bsr_dim * bsr_dim;
    constexpr size_t B_size           = sizeof(*d_B) * ldb * nrhs;
    constexpr size_t X_size           = sizeof(*d_X) * ldx * nrhs;

    HIP_CHECK(hipMalloc(&d_bsr_row_ptr, bsr_row_ptr_size));
    HIP_CHECK(hipMalloc(&d_bsr_col_ind, bsr_col_ind_size));
    HIP_CHECK(hipMalloc(&d_bsr_val, bsr_val_size));
    HIP_CHECK(hipMalloc(&d_B, B_size));
    HIP_CHECK(hipMalloc(&d_X, X_size));

    HIP_CHECK(
        hipMemcpy(d_bsr_row_ptr, h_bsr_row_ptr.data(), bsr_row_ptr_size, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_bsr_col_ind, h_bsr_col_ind.data(), bsr_col_ind_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bsr_val, h_bsr_val.data(), bsr_val_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), B_size, hipMemcpyHostToDevice));

    // 3. Initialize rocSPARSE by creating a handle.
    // Create rocSPARSE handle.
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // 4. Prepare utility variables for rocSPARSE bsrsm invocation.
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

    // Analysis policy.
    constexpr rocsparse_analysis_policy analysis_policy = rocsparse_analysis_policy_reuse;

    // Solve policy.
    constexpr rocsparse_solve_policy solve_policy = rocsparse_solve_policy_auto;

    // Obtain required buffer size.
    // This function is non blocking and executed asynchronously with respect to the host.
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_dbsrsm_buffer_size(handle,
                                                 dir,
                                                 trans_A,
                                                 trans_X,
                                                 nb,
                                                 nrhs,
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
    ROCSPARSE_CHECK(rocsparse_dbsrsm_analysis(handle,
                                              dir,
                                              trans_A,
                                              trans_X,
                                              nb,
                                              nrhs,
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

    // 6. Call dbsrsm to solve op_a(A) * op_b(X) = alpha * op_b(B).
    ROCSPARSE_CHECK(rocsparse_dbsrsm_solve(handle,
                                           dir,
                                           trans_A,
                                           trans_X,
                                           nb,
                                           nrhs,
                                           nnzb,
                                           &alpha,
                                           descr,
                                           d_bsr_val,
                                           d_bsr_row_ptr,
                                           d_bsr_col_ind,
                                           bsr_dim,
                                           info,
                                           d_B,
                                           ldb,
                                           d_X,
                                           ldx,
                                           solve_policy,
                                           temp_buffer));

    // 7. Check results.
    // Check for zero pivots.
    rocsparse_int    pivot_position;
    rocsparse_status bsrsm_status = rocsparse_bsrsm_zero_pivot(handle, info, &pivot_position);

    int errors{};

    if(bsrsm_status == rocsparse_status_zero_pivot)
    {
        std::cout << "Found zero pivot in matrix row " << pivot_position << std::endl;
        ++errors;
    }
    else
    {
        ROCSPARSE_CHECK(bsrsm_status);
    }

    // Define expected result, stored in column-major ordering.
    // clang-format off
    constexpr std::array<double, nrhs * n>
        expected_X { 27.0,    0.0,  -4.5,         -5.0,  // X_0
                    135.0,  -36.0, -28.5, -167.0 / 3.0,  // X_1
                    243.0,  -72.0, -52.5, -319.0 / 3.0,  // X_2
                    351.0, -108.0, -76.5,       -157.0}; // X_3
    // clang-format on

    // Allocate array for solution on host and copy result from device.
    std::array<double, ldx * nrhs> h_X;
    HIP_CHECK(
        hipMemcpy(h_X.data(), d_X, sizeof(double) * nb * bsr_dim * nrhs, hipMemcpyDeviceToHost));

    // Print solution and compare with expected results.
    std::cout << "Solution successfully computed: " << std::endl;
    std::cout << "X = " << std::endl;
    constexpr double eps = std::numeric_limits<double>::epsilon();
    for(rocsparse_int i = 0; i < n; ++i)
    {
        std::cout << "(";
        for(rocsparse_int j = 0; j < n; ++j)
        {
            std::printf("%9.2lf", h_X[i * ldx + j]);
            errors += std::fabs(h_X[i * ldx + j] - expected_X[i * n + j]) > eps;
        }
        std::cout << "  )" << std::endl;
    }

    // 8. Free rocSPARSE resources and device memory.
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));

    HIP_CHECK(hipFree(temp_buffer));
    HIP_CHECK(hipFree(d_X));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_bsr_val));
    HIP_CHECK(hipFree(d_bsr_col_ind));
    HIP_CHECK(hipFree(d_bsr_row_ptr));

    // 9. Print validation result.
    return report_validation_result(errors);
}
