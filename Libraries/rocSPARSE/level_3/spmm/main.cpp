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
#include <cstdio>
#include <iostream>

int main()
{
    // 1. Set up input data.
    // Number of rows and columns of the input matrices.
    constexpr rocsparse_int m = 4;
    constexpr rocsparse_int k = 6;
    constexpr rocsparse_int n = 10;

    // Sparse matrix A (m x k)
    //     ( 1 2 0 3 0 0 )
    // A = ( 0 4 5 0 0 0 )
    //     ( 0 0 0 7 8 0 )
    //     ( 0 0 1 2 4 1 )

    // Number of non-zero elements
    constexpr rocsparse_int nnz = 11;

    // COO values vector.
    constexpr std::array<double, nnz> h_coo_val
        = {1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 1.0, 2.0, 4.0, 1.0};

    // COO indices vector.
    constexpr std::array<rocsparse_int, nnz> h_coo_row_ind = {0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3};
    constexpr std::array<rocsparse_int, nnz> h_coo_col_ind = {0, 1, 3, 1, 2, 3, 4, 2, 3, 4, 5};

    // Transposition of the matrix
    constexpr rocsparse_operation trans_A = rocsparse_operation_none;

    // Dense matrix B (k x n)
    //     (  9 11 13 15 17 10 12 14 16 18 )
    //     (  8 10  1 10  6 11  7  3 12 17 )
    // B = ( 11 11  0  4  6 12  2  9 13  2 )
    //     ( 15  3  2  3  8  1  2  4  6  6 )
    //     (  2  5  7  0  1 15  9  4 10  1 )
    //     (  7 12 12  1 12  5  1 11  1 14 )

    // clang-format off
    constexpr std::array<double, (k * n)> 
        h_B{ 9,  8, 11, 15,  2,  7,
            11, 10, 11,  3,  5, 12,
            13,  1,  0,  2,  7, 12,
            15, 10,  4,  3,  0,  1,
            17,  6,  6,  8,  1, 12,
            10, 11, 12,  1, 15,  5,
            12,  7,  2,  2,  9,  1,
            14,  3,  9,  4,  4, 11,
            16, 12, 13,  6, 10,  1,
            18, 17,  2,  6,  1, 14};
    // clang-format on

    // Transposition of the matrix
    constexpr rocsparse_operation trans_B = rocsparse_operation_none;

    // Initialize a dense matrix C (m x n)
    // Matrix C elements in column-major
    constexpr rocsparse_int     ldc = m;
    std::array<double, (m * n)> h_C{};

    // Scalar alpha and beta
    constexpr double alpha = 1.0;
    constexpr double beta  = 0.0;

    // Index and data type.
    constexpr rocsparse_indextype index_type = rocsparse_indextype_i32;
    constexpr rocsparse_datatype  data_type  = rocsparse_datatype_f64_r;

    // Index base.
    constexpr rocsparse_index_base index_base = rocsparse_index_base_zero;

    // Use default algorithm.
    constexpr rocsparse_spmm_alg alg = rocsparse_spmm_alg_default;

    // 2. Offload data to device.
    double*        d_B{};
    double*        d_C{};
    double*        d_coo_val{};
    rocsparse_int* d_coo_row_ind{};
    rocsparse_int* d_coo_col_ind{};

    constexpr size_t size_val = sizeof(*d_coo_val) * nnz;
    constexpr size_t size_ind = sizeof(*d_coo_row_ind) * nnz;
    constexpr size_t size_B   = sizeof(*d_B) * k * n;
    constexpr size_t size_C   = sizeof(*d_C) * m * n;

    HIP_CHECK(hipMalloc(&d_coo_val, size_val));
    HIP_CHECK(hipMalloc(&d_coo_row_ind, size_ind));
    HIP_CHECK(hipMalloc(&d_coo_col_ind, size_ind));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));

    HIP_CHECK(hipMemcpy(d_coo_val, h_coo_val.data(), size_val, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_coo_row_ind, h_coo_row_ind.data(), size_ind, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_coo_col_ind, h_coo_col_ind.data(), size_ind, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C.data(), size_C, hipMemcpyHostToDevice));

    // 3. Initialize rocSPARSE by creating a handle.
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // 4. Create matrix descriptors.
    rocsparse_spmat_descr descr_A{};
    rocsparse_dnmat_descr descr_B{};
    rocsparse_dnmat_descr descr_C{};

    rocsparse_create_coo_descr(&descr_A,
                               m,
                               k,
                               nnz,
                               d_coo_row_ind,
                               d_coo_col_ind,
                               d_coo_val,
                               index_type,
                               index_base,
                               data_type);

    rocsparse_create_dnmat_descr(&descr_B, k, n, k, d_B, data_type, rocsparse_order_column);
    rocsparse_create_dnmat_descr(&descr_C, m, n, m, d_C, data_type, rocsparse_order_column);

    // 5. Prepare device for rocSPARSE SpMM invocation.
    // Obtain required buffer size in bytes for analysis and solve stages.
    // This stage is non blocking and executed asynchronously with respect to the host.
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_spmm(handle,
                                   trans_A,
                                   trans_B,
                                   &alpha,
                                   descr_A,
                                   descr_B,
                                   &beta,
                                   descr_C,
                                   data_type,
                                   alg,
                                   rocsparse_spmm_stage_buffer_size,
                                   &buffer_size,
                                   nullptr));
    // No synchronization with the device is needed because for scalar results, when using host
    // pointer mode (the default pointer mode) this function blocks the CPU till the GPU has copied
    // the results back to the host. See rocsparse_set_pointer_mode.

    // 6. Analysis.
    // Allocate temporary buffer.
    void* temp_buffer{};
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Perform analysis.
    ROCSPARSE_CHECK(rocsparse_spmm(handle,
                                   trans_A,
                                   trans_B,
                                   &alpha,
                                   descr_A,
                                   descr_B,
                                   &beta,
                                   descr_C,
                                   data_type,
                                   alg,
                                   rocsparse_spmm_stage_preprocess,
                                   &buffer_size,
                                   temp_buffer));

    // 7. Compute matrix-matrix multiplication.
    // This stage is non blocking and executed asynchronously with respect to the host.
    ROCSPARSE_CHECK(rocsparse_spmm(handle,
                                   trans_A,
                                   trans_B,
                                   &alpha,
                                   descr_A,
                                   descr_B,
                                   &beta,
                                   descr_C,
                                   data_type,
                                   alg,
                                   rocsparse_spmm_stage_compute,
                                   &buffer_size,
                                   temp_buffer));

    // 8. Copy C to host from device. This call synchronizes with the host.
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, size_C, hipMemcpyDeviceToHost));

    // 9. Clear rocSPARSE.
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(descr_A));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(descr_B));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(descr_C));

    // 10. Clear device memory.
    HIP_CHECK(hipFree(d_coo_row_ind));
    HIP_CHECK(hipFree(d_coo_col_ind));
    HIP_CHECK(hipFree(d_coo_val));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    // 11. Print result.
    //     (  70  40  21  44  53  35  32  32  58  70 )
    // C = (  87  95   4  60  54 104  38  57 113  78 )
    //     ( 121  61  70  21  64 127  86  60 122  50 )
    //     (  56  49  44  11  38  79  43  44  66  32 )
    std::cout << "C =" << std::endl;

    for(int i = 0; i < m; ++i)
    {
        std::cout << "    (";
        for(int j = 0; j < n; ++j)
        {
            std::printf("%5.0lf", h_C[i + j * ldc]);
        }

        std::cout << " )" << std::endl;
    }

    return 0;
}
