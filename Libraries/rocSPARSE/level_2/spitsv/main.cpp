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
#include <iostream>

int main()
{
    // 1. Set up input data.
    //           op(A)         *    y    = alpha *     x
    //
    //  ( 1.0  0.0  0.0  0.0 )   ( 1.0 )           (  0.4 )
    //  ( 2.0  3.0  0.0  0.0 ) * ( 2.0 ) =   2.5 * (  3.2 )
    //  ( 4.0  5.0  6.0  0.0 )   ( 3.0 )           ( 12.8 )
    //  ( 7.0  0.0  8.0  9.0 )   ( 4.0 )           ( 26.8 )

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

    // Scalar alpha.
    constexpr double alpha = 2.5;

    // Host vectors for the right hand side and solution of the linear system.
    constexpr std::array<double, n> h_x = {0.4, 3.2, 12.8, 26.8};
    std::array<double, n>           h_y;

    // Index and data type.
    constexpr rocsparse_indextype index_type = rocsparse_indextype_i32;
    constexpr rocsparse_datatype  data_type  = rocsparse_datatype_f64_r;

    // Index base.
    constexpr rocsparse_index_base index_base = rocsparse_index_base_zero;

    // Use default algorithm.
    constexpr rocsparse_spitsv_alg alg = rocsparse_spitsv_alg_default;

    // 2. Allocate device memory and offload input data to device.
    double*        d_x{};
    double*        d_y{};
    double*        d_csr_val{};
    rocsparse_int* d_csr_row_ptr{};
    rocsparse_int* d_csr_col_ind{};

    constexpr size_t val_size     = sizeof(*d_csr_val) * nnz;
    constexpr size_t row_ptr_size = sizeof(*d_csr_row_ptr) * (m + 1);
    constexpr size_t col_ind_size = sizeof(*d_csr_col_ind) * nnz;
    constexpr size_t size_vector  = sizeof(*d_x) * n;

    HIP_CHECK(hipMalloc(&d_x, size_vector));
    HIP_CHECK(hipMalloc(&d_y, size_vector));
    HIP_CHECK(hipMalloc(&d_csr_row_ptr, row_ptr_size));
    HIP_CHECK(hipMalloc(&d_csr_col_ind, col_ind_size));
    HIP_CHECK(hipMalloc(&d_csr_val, val_size));

    HIP_CHECK(hipMemcpy(d_x, h_x.data(), size_vector, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_row_ptr, h_csr_row_ptr.data(), row_ptr_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_col_ind, h_csr_col_ind.data(), col_ind_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_val, h_csr_val.data(), val_size, hipMemcpyHostToDevice));

    // 3. Initialize rocSPARSE by creating a handle.
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // Create sparse matrix and dense vector descriptors.
    rocsparse_spmat_descr descr_A{};
    rocsparse_dnvec_descr descr_x{};
    rocsparse_dnvec_descr descr_y{};

    rocsparse_create_csr_descr(&descr_A,
                               m,
                               n,
                               nnz,
                               d_csr_row_ptr,
                               d_csr_col_ind,
                               d_csr_val,
                               index_type,
                               index_type,
                               index_base,
                               data_type);

    rocsparse_create_dnvec_descr(&descr_x, n, d_x, data_type);
    rocsparse_create_dnvec_descr(&descr_y, n, d_y, data_type);

    // Set maximum iteration number.
    constexpr rocsparse_int max_iter     = 200;
    rocsparse_int           iter_counter = max_iter;

    // Set tolerance.
    constexpr double tolerance = 1e-4;

    // Set up convergence history.
    std::array<double, max_iter> residual_history{};

    // 4. Prepare device for rocSPARSE iterative SpSV invocation.
    // Obtain required buffer size in bytes for analysis and solve stages.
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_spitsv(handle,
                                     &iter_counter,
                                     &tolerance,
                                     residual_history.data(),
                                     trans,
                                     &alpha,
                                     descr_A,
                                     descr_x,
                                     descr_y,
                                     data_type,
                                     alg,
                                     rocsparse_spitsv_stage_buffer_size,
                                     &buffer_size,
                                     nullptr));
    // Synchronize threads.
    HIP_CHECK(hipDeviceSynchronize());

    // 5. Analysis.
    // Allocate temporary buffer.
    void* temp_buffer{};
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Perform analysis.
    ROCSPARSE_CHECK(rocsparse_spitsv(handle,
                                     &iter_counter,
                                     &tolerance,
                                     residual_history.data(),
                                     trans,
                                     &alpha,
                                     descr_A,
                                     descr_x,
                                     descr_y,
                                     data_type,
                                     alg,
                                     rocsparse_spitsv_stage_preprocess,
                                     nullptr,
                                     temp_buffer));

    // 6. Perform triangular solve op(A) * y = alpha * x.
    // This function is non blocking and executed asynchronously with respect to the host.
    ROCSPARSE_CHECK(rocsparse_spitsv(handle,
                                     &iter_counter,
                                     &tolerance,
                                     residual_history.data(),
                                     trans,
                                     &alpha,
                                     descr_A,
                                     descr_x,
                                     descr_y,
                                     data_type,
                                     alg,
                                     rocsparse_spitsv_stage_compute,
                                     &buffer_size,
                                     temp_buffer));

    // 7. Copy result from device to host. This call synchronizes with the host.
    HIP_CHECK(hipMemcpy(h_y.data(), d_y, sizeof(*d_y) * n, hipMemcpyDeviceToHost));

    // 8. Free rocSPARSE resources and device memory.
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(descr_A));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(descr_x));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(descr_y));

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(d_csr_val));
    HIP_CHECK(hipFree(d_csr_row_ptr));
    HIP_CHECK(hipFree(d_csr_col_ind));
    HIP_CHECK(hipFree(temp_buffer));

    // 9. Fetch the convergence data.
    std::cout << "Iterations performed: " << iter_counter << std::endl;

    // Check last residual computed to confirm that convergence was successful.
    const double last_residual = residual_history[iter_counter];
    const bool   is_converged  = last_residual < tolerance;

    int errors{};

    if(!is_converged)
    {
        std::cout << "Residual in last iteration " << last_residual
                  << " is greater or equal than tolerance " << tolerance
                  << ". Iterative incomplete LU factorization was unsuccessful." << std::endl;
        ++errors;
    }
    else
    {

        // 10. Print result: (1, 2, 3, 4).
        std::cout << "y = " << format_range(h_y.begin(), h_y.end()) << std::endl;
    }

    // 11. Print validation result.
    return report_validation_result(errors);
}
