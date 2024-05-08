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

int main()
{
    // 1. Setup input data.

    // alpha  *           op(A)          *      y      =     x
    //   1.0  *  ( 1.0  0.0  0.0  0.0 )  *  (   1   )  =  ( 1.0 )
    //           ( 2.0  3.0  0.0  0.0 )  *  (   0   )     ( 2.0 )
    //           ( 4.0  5.0  6.0  0.0 )  *  ( -1/6  )     ( 3.0 )
    //           ( 7.0  0.0  8.0  9.0 )  *  ( -5/27 )     ( 4.0 )

    // Number of rows and columns of the input matrix.
    constexpr rocsparse_int m = 4;
    constexpr rocsparse_int n = 4;

    // Number of non-zero elements.
    constexpr rocsparse_int nnz = 9;

    // CSR values.
    constexpr std::array<double, nnz> h_csr_val = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    // CSR row indices.
    constexpr std::array<rocsparse_int, m + 1> h_csr_row_ptr = {0, 1, 3, 6, 9};

    // CSR column indices.
    constexpr std::array<rocsparse_int, nnz> h_csr_col_ind = {0, 0, 1, 0, 1, 2, 0, 2, 3};

    // Operation applied to the matrix.
    constexpr rocsparse_operation trans = rocsparse_operation_none;

    // Analysis and solve policies.
    constexpr rocsparse_analysis_policy analysis_policy = rocsparse_analysis_policy_reuse;
    constexpr rocsparse_solve_policy    solve_policy    = rocsparse_solve_policy_auto;

    // Scalar alpha.
    constexpr double alpha = 1.0;

    // Host vector for the right hand side of the linear system.
    std::array<double, m> h_x = {1.0, 2.0, 3.0, 4.0};

    // 2. Allocate device memory and offload input data to device.
    rocsparse_int* d_csr_row_ptr;
    rocsparse_int* d_csr_col_ind;
    double*        d_csr_val;
    double*        d_x;
    double*        d_y;

    constexpr size_t x_size       = sizeof(*d_x) * m;
    constexpr size_t y_size       = sizeof(*d_y) * n;
    constexpr size_t val_size     = sizeof(*d_csr_val) * nnz;
    constexpr size_t row_ptr_size = sizeof(*d_csr_row_ptr) * (m + 1);
    constexpr size_t col_ind_size = sizeof(*d_csr_col_ind) * nnz;

    HIP_CHECK(hipMalloc(&d_csr_row_ptr, row_ptr_size));
    HIP_CHECK(hipMalloc(&d_csr_col_ind, col_ind_size));
    HIP_CHECK(hipMalloc(&d_csr_val, val_size));
    HIP_CHECK(hipMalloc(&d_x, x_size));
    HIP_CHECK(hipMalloc(&d_y, y_size));

    HIP_CHECK(hipMemcpy(d_csr_row_ptr, h_csr_row_ptr.data(), row_ptr_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_col_ind, h_csr_col_ind.data(), col_ind_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_val, h_csr_val.data(), val_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size, hipMemcpyHostToDevice));

    // 3. Initialize rocSPARSE by creating a handle.
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // 4. Prepare utility variables for rocSPARSE csrmv invocation.
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
    ROCSPARSE_CHECK(rocsparse_dcsrsv_buffer_size(handle,
                                                 trans,
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

    // 5. Perform analysis step.
    ROCSPARSE_CHECK(rocsparse_dcsrsv_analysis(handle,
                                              trans,
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

    // 6. Perform triangular solve op(A) * y = alpha * x.
    ROCSPARSE_CHECK(rocsparse_dcsrsv_solve(handle,
                                           trans,
                                           m,
                                           nnz,
                                           &alpha,
                                           descr,
                                           d_csr_val,
                                           d_csr_row_ptr,
                                           d_csr_col_ind,
                                           info,
                                           d_x,
                                           d_y,
                                           solve_policy,
                                           temp_buffer));

    // 7. Check results obtained.
    rocsparse_int    position;
    rocsparse_status pivot_status = rocsparse_csrsv_zero_pivot(handle, descr, info, &position);

    int errors{};

    if(pivot_status == rocsparse_status_zero_pivot)
    {
        std::cout << "Found zero pivot in matrix row " << position << std::endl;
        ++errors;
    }
    else
    {
        ROCSPARSE_CHECK(pivot_status);
    }

    // Host vector for the solution of the linear system.
    std::array<double, n> h_y;
    HIP_CHECK(hipMemcpy(h_y.data(), d_y, y_size, hipMemcpyDeviceToHost));

    std::cout << "Solution successfully computed: ";
    std::cout << "y = " << format_range(h_y.begin(), h_y.end()) << std::endl;

    // Define expected result.
    constexpr std::array<double, n> expected_y = {1.0, 0.0, -1.0 / 6.0, -5.0 / 27.0};

    // Compare solution with the expected result.
    const double eps = 1.0e5 * std::numeric_limits<double>::epsilon();
    for(size_t i = 0; i < h_y.size(); ++i)
    {
        errors += std::fabs(h_y[i] - expected_y[i]) > eps;
    }

    // 8. Free rocSPARSE resources and device memory.
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));

    HIP_CHECK(hipFree(d_csr_row_ptr));
    HIP_CHECK(hipFree(d_csr_col_ind));
    HIP_CHECK(hipFree(d_csr_val));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(temp_buffer));

    // 9. Print validation result.
    return report_validation_result(errors);
}
