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
    // 1. Set up input data.
    // Number of rows and columns of the input matrices.
    constexpr rocsparse_int m = 6;
    constexpr rocsparse_int k = 4;
    constexpr rocsparse_int n = 8;

    // Dense matrix A (m x k)
    //     (  9  8 12  9 )
    // A = ( 24 13 11 10 )
    //     (  6 27 15  6 )
    //     ( 21  3 15  0 )
    //     ( 10 18 36  0 )
    //     ( 17  6 12 27 )

    // Matrix A elements.
    // clang-format off
    std::array<double, (m * k)> h_A
        = { 9,  8, 12,  9,
           24, 13, 11, 10,
            6, 27, 15,  6,
           21,  3, 15,  0,
           10, 18, 36,  0,
           17,  6, 12, 27};
    // clang-format on

    // Dense matrix B (k x n).
    //     (  9 11   13   15 17 10 12 14 )
    // B = (  8 10    1   10  6 11  7  3 )
    //     ( 12  6    0   18 12  9 12 15 )
    //     (  9 27   18   36 27 18 27  9 )

    // Matrix B elements.
    // clang-format off
    std::array<double, (k * n)> h_B
        = { 9, 11, 13, 15, 17, 10, 12, 14,
            8, 10,  1, 10,  6, 11,  7,  3,
           12,  6,  0, 18, 12,  9, 12, 15,
            9, 27, 18, 36, 27, 18, 27,  9};
    // clang-format on

    // Sparse matrix C (m x n).
    //     ( 1  2  0  3  0  0  0  0 )
    // C = ( 0  4  5  0  0  0  0  0 )
    //     ( 0  0  6  7  8  0  0  0 )
    //     ( 0  0  0  9 10 11  0 12 )
    //     ( 0 13  0  0 14  0  0  0 )
    //     ( 0  0 15  0  0 16  0  0 )
    // Transposition of the matrix

    // Number of non-zero entries.
    constexpr rocsparse_int nnz = 16;

    // CSR values.
    std::array<double, nnz> h_csr_val
        = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.};

    // CSR column indices.
    std::array<rocsparse_int, nnz> h_csr_col_ind = {0, 1, 3, 1, 2, 2, 3, 4, 3, 4, 5, 7, 1, 4, 2, 5};

    // CSR row indices.
    std::array<rocsparse_int, (m + 1)> h_csr_row_ptr = {0, 3, 5, 8, 12, 14, nnz};

    // Transposition of the matrix.
    constexpr rocsparse_order     order   = rocsparse_order_row;
    constexpr rocsparse_operation trans_A = rocsparse_operation_none;
    constexpr rocsparse_operation trans_B = rocsparse_operation_none;

    // Scalars alpha and beta.
    constexpr double alpha = 1.0;
    constexpr double beta  = 1.0;

    // Index and data type.
    constexpr rocsparse_indextype index_type = rocsparse_indextype_i32;
    constexpr rocsparse_datatype  data_type  = rocsparse_datatype_f64_r;

    // Index base.
    constexpr rocsparse_index_base index_base = rocsparse_index_base_zero;

    // 2. Prepare device for calculation.
    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // 3. Allocate device memory and offload data to device.
    rocsparse_int* d_csr_col_ind;
    rocsparse_int* d_csr_row_ptr;
    double*        d_csr_val;
    double*        d_A;
    double*        d_B;

    constexpr size_t col_ind_size = sizeof(*d_csr_col_ind) * nnz;
    constexpr size_t row_ptr_size = sizeof(*d_csr_row_ptr) * (m + 1);
    constexpr size_t val_size     = sizeof(*d_csr_val) * nnz;
    constexpr size_t A_size       = sizeof(*d_A) * m * k;
    constexpr size_t B_size       = sizeof(*d_B) * k * n;

    HIP_CHECK(hipMalloc(&d_csr_col_ind, col_ind_size));
    HIP_CHECK(hipMalloc(&d_csr_row_ptr, row_ptr_size));
    HIP_CHECK(hipMalloc(&d_csr_val, val_size));
    HIP_CHECK(hipMalloc(&d_A, A_size));
    HIP_CHECK(hipMalloc(&d_B, B_size));

    HIP_CHECK(hipMemcpy(d_csr_col_ind, h_csr_col_ind.data(), col_ind_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_row_ptr, h_csr_row_ptr.data(), row_ptr_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_val, h_csr_val.data(), val_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), A_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), B_size, hipMemcpyHostToDevice));

    // 4. Create matrix descriptors.
    // Matrix descriptor.
    rocsparse_dnmat_descr mat_A_desc{};
    rocsparse_dnmat_descr mat_B_desc{};
    rocsparse_spmat_descr mat_C_desc{};

    // Create sparse csr descriptor.
    rocsparse_create_csr_descr(&mat_C_desc,
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

    // Create dense descriptors.
    rocsparse_create_dnmat_descr(&mat_A_desc, m, k, k, d_A, data_type, order);
    rocsparse_create_dnmat_descr(&mat_B_desc, k, n, n, d_B, data_type, order);

    // 5. Calculate size and allocate temp buffer.
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_sddmm_buffer_size(handle,
                                                trans_A,
                                                trans_B,
                                                &alpha,
                                                mat_A_desc,
                                                mat_B_desc,
                                                &beta,
                                                mat_C_desc,
                                                data_type,
                                                rocsparse_sddmm_alg::rocsparse_sddmm_alg_default,
                                                &buffer_size));
    // No synchronization with the device is needed because for scalar results, when using host
    // pointer mode (the default pointer mode) this function blocks the CPU till the GPU has copied
    // the results back to the host. See rocsparse_set_pointer_mode.

    // Allocate the temp buffer.
    void* d_temp_buffer{};
    HIP_CHECK(hipMalloc(&d_temp_buffer, buffer_size));

    // 6. Do the necessary pre calculations.
    // The result is written in the temp buffer.
    ROCSPARSE_CHECK(rocsparse_sddmm_preprocess(handle,
                                               trans_A,
                                               trans_B,
                                               &alpha,
                                               mat_A_desc,
                                               mat_B_desc,
                                               &beta,
                                               mat_C_desc,
                                               data_type,
                                               rocsparse_sddmm_alg::rocsparse_sddmm_alg_default,
                                               d_temp_buffer));

    // 7. Compute the solution.
    ROCSPARSE_CHECK(rocsparse_sddmm(handle,
                                    trans_A,
                                    trans_B,
                                    &alpha,
                                    mat_A_desc,
                                    mat_B_desc,
                                    &beta,
                                    mat_C_desc,
                                    data_type,
                                    rocsparse_sddmm_alg::rocsparse_sddmm_alg_default,
                                    d_temp_buffer));

    // 8. Copy C from device to host. These calls synchronize with the host.
    HIP_CHECK(hipMemcpy(h_csr_val.data(), d_csr_val, val_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_csr_col_ind.data(), d_csr_col_ind, col_ind_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_csr_row_ptr.data(), d_csr_row_ptr, row_ptr_size, hipMemcpyDeviceToHost));

    // 9. Clear rocSPARSE.
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(mat_A_desc));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(mat_B_desc));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(mat_C_desc));

    // 10. Clear device memory.
    HIP_CHECK(hipFree(d_csr_row_ptr));
    HIP_CHECK(hipFree(d_csr_col_ind));
    HIP_CHECK(hipFree(d_csr_val));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_temp_buffer));

    // 11. Print result.
    std::cout << "C =" << std::endl;

    for(rocsparse_int i = 0; i < m; ++i)
    {
        int row_pos = h_csr_row_ptr[i];
        int row_end = h_csr_row_ptr[i + 1];

        std::cout << "    (";
        for(rocsparse_int j = 0; j < n; ++j)
        {
            if(row_pos < row_end && j == h_csr_col_ind[row_pos])
            {
                std::cout << std::setw(5) << h_csr_val[row_pos];
                ++row_pos;
            }
            else
            {
                std::cout << std::setw(5) << 0;
            }
        }

        std::cout << " )" << std::endl;
    }

    return 0;
}
