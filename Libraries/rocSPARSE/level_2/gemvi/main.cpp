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

#include <hip/hip_runtime.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>

int main()
{
    // 1. Setup input data
    //
    // Given:
    // - scalars      : alpha, beta
    // - sparse vector: x
    // - dense vector : y
    // - dense matrix : A
    // Calculate dense vector y' such that:
    //    y'    = alpha *            op(A)            *    x    + beta *    y
    // Or more concretely with default alpha and beta:
    //                                                  ( 1.0 )
    // (245.7)           (  9.0 10.0 11.0 12.0 13.0 )   ( 2.0 )          ( 4.0 )
    // (358.0) =  3.7  * ( 14.0 15.0 16.0 17.0 18.0 ) * ( 0.0 ) +  1.3 * ( 5.0 )
    // (470.3)           ( 19.0 20.0 21.0 22.0 23.0 )   ( 3.0 )          ( 6.0 )
    //                                                  ( 0.0 )

    // Dense matrix A in column-major
    constexpr rocsparse_int                       A_rows = 3;
    constexpr rocsparse_int                       A_cols = 5;
    constexpr std::array<double, A_rows * A_cols> A
        = {9, 14, 19, 10, 15, 20, 11, 16, 21, 12, 17, 22, 13, 18, 23};

    constexpr rocsparse_int lda = A_rows;

    // Sparse vector x
    constexpr rocsparse_int                  x_non_zero = 3;
    constexpr std::array<double, x_non_zero> x_values   = {1, 2, 3};

    constexpr std::array<rocsparse_int, x_non_zero> x_indices = {0, 1, 3};

    // Dense vector y
    constexpr std::array<double, A_rows> y = {4, 5, 6};

    constexpr double alpha = 3.7;
    constexpr double beta  = 1.3;

    // Matrix operation
    constexpr rocsparse_operation trans = rocsparse_operation_none;

    // Index base
    constexpr rocsparse_index_base base = rocsparse_index_base_zero;

    // 2. Prepare device for calculation

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // 3. Offload data to device
    double*        d_A;
    rocsparse_int* d_x_indices;
    double*        d_x_values;
    double*        d_y;

    HIP_CHECK(hipMalloc(&d_A, sizeof(*d_A) * A_rows * A_cols));
    HIP_CHECK(hipMalloc(&d_x_indices, sizeof(*d_x_indices) * x_non_zero));
    HIP_CHECK(hipMalloc(&d_x_values, sizeof(*d_x_values) * x_non_zero));
    HIP_CHECK(hipMalloc(&d_y, sizeof(*d_y) * A_rows));

    HIP_CHECK(hipMemcpy(d_A, A.data(), sizeof(*d_A) * A_rows * A_cols, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x_indices,
                        x_indices.data(),
                        sizeof(*d_x_indices) * x_non_zero,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x_values,
                        x_values.data(),
                        sizeof(*d_x_values) * x_non_zero,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, y.data(), sizeof(*d_y) * A_rows, hipMemcpyHostToDevice));

    // Obtain buffer size
    // This function is non blocking and executed asynchronously with respect to the host.
    size_t buffer_size;
    ROCSPARSE_CHECK(
        rocsparse_dgemvi_buffer_size(handle, trans, A_rows, A_cols, x_non_zero, &buffer_size));
    // No synchronization with the device is needed because for scalar results, when using host
    // pointer mode (the default pointer mode) this function blocks the CPU till the GPU has copied
    // the results back to the host. See rocsparse_set_pointer_mode.

    // Allocate temporary buffer.
    void* buffer;
    HIP_CHECK(hipMalloc(&buffer, buffer_size));

    // 4. Call dgemvi to perform y' = alpha * op(A) * x + beta * y
    ROCSPARSE_CHECK(rocsparse_dgemvi(handle,
                                     trans,
                                     A_rows,
                                     A_cols,
                                     &alpha,
                                     d_A,
                                     lda,
                                     x_non_zero,
                                     d_x_values,
                                     d_x_indices,
                                     &beta,
                                     d_y,
                                     base,
                                     buffer));

    // Copy y' from device to host
    std::array<double, A_rows> y_prime;
    HIP_CHECK(hipMemcpy(y_prime.data(), d_y, sizeof(*d_y) * A_rows, hipMemcpyDeviceToHost));

    // 5. Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // 6. Free device memory
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_x_indices));
    HIP_CHECK(hipFree(d_x_values));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(buffer));

    // 7. Print results
    std::cout << "y = " << format_range(std::begin(y_prime), std::end(y_prime)) << std::endl;

    return 0;
}
