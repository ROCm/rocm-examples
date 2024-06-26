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
// 'rocsparse_dspmv' is added in rocSPARSE 3.0. In lower versions use
// 'rocsparse_dspmv_ex' instead.
#if ROCSPARSE_VERSION_MAJOR < 3
    #define rocsparse_spmv(...) rocsparse_spmv_ex(__VA_ARGS__)
#endif

    // 1. Set up input data
    //
    // alpha *       op(A)       *    x    + beta *    y    =      y
    //
    //   3.7 * ( 1.0  0.0  2.0 ) * ( 1.0 ) +  1.3 * ( 4.0 ) = (  31.1 )
    //         ( 3.0  0.0  4.0 ) * ( 2.0 )          ( 5.0 ) = (  62.0 )
    //         ( 5.0  6.0  0.0 ) * ( 3.0 )          ( 6.0 ) = (  70.7 )
    //         ( 7.0  0.0  8.0 ) *                  ( 7.0 ) = ( 123.8 )

    // Set up COO matrix

    // Number of rows and columns
    constexpr rocsparse_int m = 4;
    constexpr rocsparse_int n = 3;

    // Number of non-zero elements
    constexpr rocsparse_int nnz = 8;

    // COO values
    constexpr std::array<double, nnz> h_coo_val = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    // COO row indices
    constexpr std::array<rocsparse_int, nnz> h_coo_row_ind = {0, 0, 1, 1, 2, 2, 3, 3};

    // COO column indices
    constexpr std::array<rocsparse_int, nnz> h_coo_col_ind = {0, 2, 0, 2, 0, 1, 0, 2};

    // Transposition of the matrix
    constexpr rocsparse_operation trans = rocsparse_operation_none;

    // Set up scalars
    constexpr double alpha = 3.7;
    constexpr double beta  = 1.3;

    // Set up x and y vectors
    constexpr std::array<double, n> h_x = {1.0, 2.0, 3.0};
    std::array<double, m>           h_y = {4.0, 5.0, 6.0, 7.0};

    // Index and data type.
    constexpr rocsparse_indextype index_type = rocsparse_indextype_i32;
    constexpr rocsparse_datatype  data_type  = rocsparse_datatype_f64_r;

    // Index base.
    constexpr rocsparse_index_base index_base = rocsparse_index_base_zero;

    // Use default algorithm.
    constexpr rocsparse_spmv_alg alg = rocsparse_spmv_alg_default;

    // 2. Allocate device memory and offload input data to device.
    double*        d_x{};
    double*        d_y{};
    double*        d_coo_val{};
    rocsparse_int* d_coo_row_ind{};
    rocsparse_int* d_coo_col_ind{};

    constexpr size_t size_val = sizeof(*d_coo_val) * nnz;
    constexpr size_t size_ind = sizeof(*d_coo_row_ind) * nnz;
    constexpr size_t size_x   = sizeof(*d_x) * n;
    constexpr size_t size_y   = sizeof(*d_y) * m;

    HIP_CHECK(hipMalloc(&d_x, size_x));
    HIP_CHECK(hipMalloc(&d_y, size_y));
    HIP_CHECK(hipMalloc(&d_coo_val, size_val));
    HIP_CHECK(hipMalloc(&d_coo_row_ind, size_ind));
    HIP_CHECK(hipMalloc(&d_coo_col_ind, size_ind));

    HIP_CHECK(hipMemcpy(d_x, h_x.data(), size_x, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y.data(), size_y, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_coo_val, h_coo_val.data(), size_val, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_coo_row_ind, h_coo_row_ind.data(), size_ind, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_coo_col_ind, h_coo_col_ind.data(), size_ind, hipMemcpyHostToDevice));

    // 3. Initialize rocSPARSE by creating a handle.
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // Create sparse matrix and dense vector descriptors.
    rocsparse_spmat_descr descr_A{};
    rocsparse_dnvec_descr descr_x{};
    rocsparse_dnvec_descr descr_y{};

    rocsparse_create_coo_descr(&descr_A,
                               m,
                               n,
                               nnz,
                               d_coo_row_ind,
                               d_coo_col_ind,
                               d_coo_val,
                               index_type,
                               index_base,
                               data_type);

    rocsparse_create_dnvec_descr(&descr_x, n, d_x, data_type);
    rocsparse_create_dnvec_descr(&descr_y, m, d_y, data_type);

    // 4. Prepare device for rocSPARSE SpMV invocation.
    // Obtain required buffer size in bytes for preprocess and compute stages.
    // This stage is non blocking and executed asynchronously with respect to the host.
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_spmv(handle,
                                   trans,
                                   &alpha,
                                   descr_A,
                                   descr_x,
                                   &beta,
                                   descr_y,
                                   data_type,
                                   alg,
                                   rocsparse_spmv_stage_buffer_size,
                                   &buffer_size,
                                   nullptr));
    // No synchronization with the device is needed because for scalar results, when using host
    // pointer mode (the default pointer mode) this function blocks the CPU till the GPU has copied
    // the results back to the host. See rocsparse_set_pointer_mode.

    // 5. Preprocess.
    // Allocate temporary buffer.
    void* temp_buffer{};
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Perform preprocessing.
    ROCSPARSE_CHECK(rocsparse_spmv(handle,
                                   trans,
                                   &alpha,
                                   descr_A,
                                   descr_x,
                                   &beta,
                                   descr_y,
                                   data_type,
                                   alg,
                                   rocsparse_spmv_stage_preprocess,
                                   &buffer_size,
                                   temp_buffer));

    // 6. Compute matrix-vector multiplication.
    // This stage is non blocking and executed asynchronously with respect to the host.
    ROCSPARSE_CHECK(rocsparse_spmv(handle,
                                   trans,
                                   &alpha,
                                   descr_A,
                                   descr_x,
                                   &beta,
                                   descr_y,
                                   data_type,
                                   alg,
                                   rocsparse_spmv_stage_compute,
                                   &buffer_size,
                                   temp_buffer));

    // 7. Copy result from device to host. This call synchronizes with the host.
    HIP_CHECK(hipMemcpy(h_y.data(), d_y, size_y, hipMemcpyDeviceToHost));

    // 8. Free rocSPARSE resources and device memory.
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(descr_A));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(descr_x));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(descr_y));

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(d_coo_val));
    HIP_CHECK(hipFree(d_coo_row_ind));
    HIP_CHECK(hipFree(d_coo_col_ind));
    HIP_CHECK(hipFree(temp_buffer));

    // 9. Print result.
    std::cout << "y = " << format_range(h_y.begin(), h_y.end()) << std::endl;

    return 0;
}
