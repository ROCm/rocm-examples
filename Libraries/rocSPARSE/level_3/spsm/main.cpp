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

#include "rocsparse_utils.hpp"

#include <rocsparse/rocsparse.h>

#include <hip/hip_runtime.h>

#include <array>
#include <iomanip>
#include <iostream>

int main()
{
    // 1. Set up input data
    // Number of rows and columns of the input matrices.
    constexpr rocsparse_int m = 6;
    constexpr rocsparse_int n = 8;

    // Sparse matrix A (m x m)
    //     ( 1 2 0 3  0  0 )
    // A = ( 0 4 5 0  0  0 )
    //     ( 0 0 6 7  8  0 )
    //     ( 0 0 0 9 10 11 )
    //     ( 0 0 0 0 12  0 )
    //     ( 0 0 0 0  0 13 )

    // Number of non-zero entries
    constexpr rocsparse_int nnz = 13;

    // CSR values
    std::array<double, nnz> h_csr_val = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.};

    // CSR column indices
    std::array<rocsparse_int, nnz> h_csr_col_ind = {0, 1, 3, 1, 2, 2, 3, 4, 3, 4, 5, 4, 5};

    // CSR row indices
    std::array<rocsparse_int, (m + 1)> h_csr_row_ptr = {0, 3, 5, 8, 11, 12, nnz};

    // Transposition of the matrix
    constexpr rocsparse_order     order   = rocsparse_order_column;
    constexpr rocsparse_operation trans_A = rocsparse_operation_none;

    // Dense matrix B (m x n)
    //     (  9 11   13   15 17 10 12 14 )
    //     (  8 10    1   10  6 11  7  3 )
    // B = ( 12  6    0   18 12  9 12 15 )
    //     (  9 27   18   36 27 18 27  9 )
    //     ( 24 15   21    0  3 15  9 18 )
    //     ( 13  6.5  3.25 0 26 13 39 13 )

    // Matrix B elements in column-major
    std::array<double, (m * n)> h_B
        = {9,  8,    12, 9,  24, 13, 11, 10, 6,  27, 15, 6.5, 13, 1,  0,  18,
           21, 3.25, 15, 10, 18, 36, 0,  0,  17, 6,  12, 27,  3,  26, 10, 11,
           9,  18,   15, 13, 12, 7,  12, 27, 9,  39, 14, 3,   15, 9,  18, 13};

    // Transposition of the matrix
    constexpr rocsparse_operation trans_B = rocsparse_operation_none;

    // Initialize a dense matrix X (m x n)
    std::array<double, (m * n)> h_X{};

    // Scalar alpha
    constexpr double alpha = 1.0;

    // Index and data type.
    constexpr rocsparse_indextype index_type = rocsparse_indextype_i32;
    constexpr rocsparse_datatype  data_type  = rocsparse_datatype_f64_r;

    // Index base.
    constexpr rocsparse_index_base index_base = rocsparse_index_base_zero;

    // 2. Prepare device for calculation
    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // 3. Offload data to device
    double*        d_csr_val;
    rocsparse_int* d_csr_col_ind;
    rocsparse_int* d_csr_row_ptr;
    double*        d_B;
    double*        d_X;

    constexpr size_t size_val     = sizeof(*d_csr_val) * nnz;
    constexpr size_t size_col_ind = sizeof(*d_csr_col_ind) * nnz;
    constexpr size_t size_row_ind = sizeof(*d_csr_row_ptr) * (m + 1);
    constexpr size_t size_B       = sizeof(*d_B) * m * n;
    constexpr size_t size_X       = sizeof(*d_X) * m * n;

    HIP_CHECK(hipMalloc(&d_csr_val, size_val));
    HIP_CHECK(hipMalloc(&d_csr_col_ind, size_col_ind));
    HIP_CHECK(hipMalloc(&d_csr_row_ptr, size_row_ind));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_X, size_X));

    HIP_CHECK(hipMemcpy(d_csr_val, h_csr_val.data(), size_val, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_col_ind, h_csr_col_ind.data(), size_col_ind, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_row_ptr, h_csr_row_ptr.data(), size_row_ind, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_X, 0, size_X));

    // 4. Create matrix descriptors
    // Matrix descriptor
    rocsparse_spmat_descr mat_A_desc{};
    rocsparse_dnmat_descr mat_B_desc{};
    rocsparse_dnmat_descr mat_X_desc{};

    // Create sparse csr descriptor
    rocsparse_create_csr_descr(&mat_A_desc,
                               m,
                               m,
                               nnz,
                               d_csr_row_ptr,
                               d_csr_col_ind,
                               d_csr_val,
                               index_type,
                               index_type,
                               index_base,
                               data_type);

    // Create dense descriptors
    rocsparse_create_dnmat_descr(&mat_B_desc, m, n, m, d_B, data_type, order);
    rocsparse_create_dnmat_descr(&mat_X_desc, m, n, m, d_X, data_type, order);

    // 5. Call spsm to solve op_a(A) * C = alpha * op_b(B)
    // Query the necessary temp buffer size
    size_t buffer_size;
    void*  d_temp_buffer{};
    ROCSPARSE_CHECK(rocsparse_spsm(handle,
                                   trans_A,
                                   trans_B,
                                   &alpha,
                                   mat_A_desc,
                                   mat_B_desc,
                                   mat_X_desc,
                                   data_type,
                                   rocsparse_spsm_alg::rocsparse_spsm_alg_default,
                                   rocsparse_spsm_stage::rocsparse_spsm_stage_buffer_size,
                                   &buffer_size,
                                   d_temp_buffer));
    // No synchronization with the device is needed because for scalar results, when using host
    // pointer mode (the default pointer mode) this function blocks the CPU till the GPU has copied
    // the results back to the host. See rocsparse_set_pointer_mode.

    // Allocate the temp buffer
    HIP_CHECK(hipMalloc(&d_temp_buffer, buffer_size));

    // Do the necessary pre calculations. The result is written in the temp buffer.
    ROCSPARSE_CHECK(rocsparse_spsm(handle,
                                   trans_A,
                                   trans_B,
                                   &alpha,
                                   mat_A_desc,
                                   mat_B_desc,
                                   mat_X_desc,
                                   data_type,
                                   rocsparse_spsm_alg::rocsparse_spsm_alg_default,
                                   rocsparse_spsm_stage::rocsparse_spsm_stage_preprocess,
                                   &buffer_size,
                                   d_temp_buffer));

    // Compute the solution.
    // This function is non blocking and executed asynchronously with respect to the host.
    ROCSPARSE_CHECK(rocsparse_spsm(handle,
                                   trans_A,
                                   trans_B,
                                   &alpha,
                                   mat_A_desc,
                                   mat_B_desc,
                                   mat_X_desc,
                                   data_type,
                                   rocsparse_spsm_alg::rocsparse_spsm_alg_default,
                                   rocsparse_spsm_stage::rocsparse_spsm_stage_compute,
                                   &buffer_size,
                                   d_temp_buffer));

    // 6. Copy C to host from device. This call synchronizes with the host.
    HIP_CHECK(hipMemcpy(h_X.data(), d_X, size_X, hipMemcpyDeviceToHost));

    // 7. Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(mat_A_desc));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(mat_B_desc));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(mat_X_desc));

    // 8. Clear device memory
    HIP_CHECK(hipFree(d_temp_buffer));
    HIP_CHECK(hipFree(d_csr_val));
    HIP_CHECK(hipFree(d_csr_col_ind));
    HIP_CHECK(hipFree(d_csr_row_ptr));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_X));

    // 9. Print result
    std::cout << "X =" << std::endl;

    for(rocsparse_int i = 0; i < m; ++i)
    {
        std::cout << "    (";
        for(rocsparse_int j = 0; j < n; ++j)
        {
            std::cout << std::setw(5) << h_X[i + j * m];
        }

        std::cout << " )" << std::endl;
    }

    return 0;
}
