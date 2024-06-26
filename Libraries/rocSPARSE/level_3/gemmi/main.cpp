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
    // Solve C =  alpha * op_a(A) * op_b(B) + beta * C, with an m x k dense matrix A,
    // a k x n sparse matrix B and  alpha beta scalars.
    // Number of rows and columns of the input matrix.
    constexpr rocsparse_int m = 4;
    constexpr rocsparse_int k = 6;
    constexpr rocsparse_int n = 5;

    // A matrix (m x k) values, stored in column-major order.
    // clang-format off
    constexpr std::array<double, m * k>
        h_A{ 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,
             7.0,  8.0,  9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    // clang-format on

    // Sparse matrix B (k x n)
    //     (  1  0  2  0  0 )
    //     (  0  3  0  0  0 )
    // B = (  0  4  0  5  0 )
    //     (  6  0  7  0  8 )
    //     (  0  9  0  0  0 )
    //     (  0  0 10  0 11 )

    // Number of non-zero values.
    constexpr rocsparse_int nnz = 11;

    // CSR row pointers.
    constexpr std::array<rocsparse_int, k + 1> h_csr_row_ptr{0, 2, 3, 5, 8, 9, nnz};

    // CSR column indices.
    constexpr std::array<rocsparse_int, nnz> h_csr_col_ind{0, 2, 1, 1, 3, 0, 2, 4, 1, 2, 3};

    // CSR values.
    constexpr std::array<double, nnz>
        h_csr_val{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};

    // C matrix
    // clang-format off
    std::array<double, m * n>
        h_C{4.0, 4.0, 3.0, 3.0, 2.0,
            4.0, 3.0, 3.0, 2.0, 2.0,
            3.0, 3.0, 2.0, 2.0, 1.0,
            3.0, 2.0, 2.0, 1.0, 1.0};
    // clang-format on

    // Scalar alpha and beta.
    constexpr double alpha = 2.0;
    constexpr double beta  = 0.5;

    // Operations applied to the matrices.
    // Operation on A must be none and operation on B must be transpose. The other combinations aren't supported
    constexpr rocsparse_operation trans_A = rocsparse_operation_none;
    constexpr rocsparse_operation trans_B = rocsparse_operation_transpose;

    // 2. Allocate device memory and copy input data to device.
    rocsparse_int* d_csr_row_ptr{};
    rocsparse_int* d_csr_col_ind{};
    double*        d_csr_val{};
    double*        d_A{};
    double*        d_C{};

    constexpr size_t csr_row_ptr_size = sizeof(*d_csr_row_ptr) * (k + 1);
    constexpr size_t csr_col_ind_size = sizeof(*d_csr_col_ind) * nnz;
    constexpr size_t csr_val_size     = sizeof(*d_csr_val) * nnz;
    constexpr size_t A_size           = sizeof(*d_A) * m * k;
    constexpr size_t C_size           = sizeof(*d_C) * m * n;

    HIP_CHECK(hipMalloc(&d_csr_row_ptr, csr_row_ptr_size));
    HIP_CHECK(hipMalloc(&d_csr_col_ind, csr_col_ind_size));
    HIP_CHECK(hipMalloc(&d_csr_val, csr_val_size));
    HIP_CHECK(hipMalloc(&d_A, A_size));
    HIP_CHECK(hipMalloc(&d_C, C_size));

    HIP_CHECK(
        hipMemcpy(d_csr_row_ptr, h_csr_row_ptr.data(), csr_row_ptr_size, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_csr_col_ind, h_csr_col_ind.data(), csr_col_ind_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_val, h_csr_val.data(), csr_val_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), A_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C.data(), C_size, hipMemcpyHostToDevice));

    // 3. Prepare for rocSPARSE function call by creating a handle and matrix descriptor.
    // Create rocSPARSE handle.
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // Create matrix descriptor.
    rocsparse_mat_descr mat_B_desc{};

    // Create sparse matrix descriptor
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&mat_B_desc));

    // 4. Perform the computation.
    // This function is non blocking and executed asynchronously with respect to the host.
    ROCSPARSE_CHECK(rocsparse_dgemmi(handle,
                                     trans_A,
                                     trans_B,
                                     m,
                                     n,
                                     k,
                                     nnz,
                                     &alpha,
                                     d_A,
                                     m,
                                     mat_B_desc,
                                     d_csr_val,
                                     d_csr_row_ptr,
                                     d_csr_col_ind,
                                     &beta,
                                     d_C,
                                     m));

    // 5. Copy result from device. This call synchronizes with the host.
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, C_size, hipMemcpyDeviceToHost));

    // Print solution and compare with expected results.
    std::cout << "Solution successfully computed: " << std::endl;
    std::cout << "C = " << std::endl;
    for(rocsparse_int i = 0; i < m; ++i)
    {
        std::cout << "    (";
        for(rocsparse_int j = 0; j < n; ++j)
        {
            std::cout << std::setw(7) << h_C[i + j * m];
        }

        std::cout << " )" << std::endl;
    }

    // 6. Free rocSPARSE resources and device memory.
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(mat_B_desc));

    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_csr_val));
    HIP_CHECK(hipFree(d_csr_col_ind));
    HIP_CHECK(hipFree(d_csr_row_ptr));

    return 0;
}
