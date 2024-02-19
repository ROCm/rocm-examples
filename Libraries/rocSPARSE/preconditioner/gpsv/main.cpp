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
    // 1. Set up input data.

    // Number of unknowns
    constexpr rocsparse_int m = 5;

    // Number of batches and stride
    constexpr rocsparse_int batch_count  = 2;
    constexpr rocsparse_int batch_stride = 543;

    constexpr rocsparse_int N_b = m * batch_stride * batch_count;

    std::array<double, N_b> h_s;
    std::array<double, N_b> h_l;
    std::array<double, N_b> h_d;
    std::array<double, N_b> h_u;
    std::array<double, N_b> h_w;
    std::array<double, N_b> h_b;

    //           A_1          *  x_1  =  b_1
    //  ( 6  -2   1   0   0 )   ( 1 )   (  5 )
    //  ( 8   3   4  -3   0 )   ( 2 )   ( 14 )
    //  ( 2   0  -2   3   2 ) * ( 3 ) = ( 18 )
    //  ( 0  10   7   1   4 )   ( 4 )   ( 65 )
    //  ( 0   0   3  -1   5 )   ( 5 )   ( 30 )

    // Pentadiagonal matrix A_1
    rocsparse_int batch_id    = 0;
    rocsparse_int batch_index = batch_id;

    h_s[batch_index] = 0; // 0 by definition
    h_l[batch_index] = 0; // 0 by definition
    h_d[batch_index] = 6;
    h_u[batch_index] = -2;
    h_w[batch_index] = 1;
    h_b[batch_index] = 5; // solution vector

    batch_index += batch_stride;
    h_s[batch_index] = 0; // 0 by definition
    h_l[batch_index] = 8;
    h_d[batch_index] = 3;
    h_u[batch_index] = 4;
    h_w[batch_index] = -3;
    h_b[batch_index] = 14; // solution vector

    batch_index += batch_stride;
    h_s[batch_index] = 2;
    h_l[batch_index] = 0;
    h_d[batch_index] = -2;
    h_u[batch_index] = 3;
    h_w[batch_index] = 2;
    h_b[batch_index] = 18; // solution vector

    batch_index += batch_stride;
    h_s[batch_index] = 10;
    h_l[batch_index] = 7;
    h_d[batch_index] = 1;
    h_u[batch_index] = 4;
    h_w[batch_index] = 0; // 0 by definition
    h_b[batch_index] = 65; // solution vector

    batch_index += batch_stride;
    h_s[batch_index] = 3;
    h_l[batch_index] = -1;
    h_d[batch_index] = 5;
    h_u[batch_index] = 0; // 0 by definition
    h_w[batch_index] = 0; // 0 by definition
    h_b[batch_index] = 30; // solution vector

    //           A_2           *  x_2   =   b_2
    //  (  1  -2   3   0   0 )   (  3 )   (   20 )
    //  ( -4   5  -6   7   0 )   ( 14 )   (  612 )
    //  ( -8   9  10  -9   8 ) * ( 15 ) = (  -56 )
    //  (  0  -7   6  -5   4 )   ( 92 )   ( -208 )
    //  (  0   0  -3   2  -1 )   ( 65 )   (   74 )

    // Pentadiagonal matrix A_2

    batch_id    = 1;
    batch_index = batch_id;

    h_s[batch_index] = 0; // 0 by definition
    h_l[batch_index] = 0; // 0 by definition
    h_d[batch_index] = 1;
    h_u[batch_index] = -2;
    h_w[batch_index] = 3;
    h_b[batch_index] = 20; // solution vector

    batch_index += batch_stride;
    h_s[batch_index] = 0; // 0 by definition
    h_l[batch_index] = -4;
    h_d[batch_index] = 5;
    h_u[batch_index] = -6;
    h_w[batch_index] = 7;
    h_b[batch_index] = 612; // solution vector

    batch_index += batch_stride;
    h_s[batch_index] = -8;
    h_l[batch_index] = 9;
    h_d[batch_index] = 10;
    h_u[batch_index] = -9;
    h_w[batch_index] = 8;
    h_b[batch_index] = -56; // solution vector

    batch_index += batch_stride;
    h_s[batch_index] = -7;
    h_l[batch_index] = 6;
    h_d[batch_index] = -5;
    h_u[batch_index] = 4;
    h_w[batch_index] = 0; // 0 by definition
    h_b[batch_index] = -208; // solution vector

    batch_index += batch_stride;
    h_s[batch_index] = -3;
    h_l[batch_index] = 2;
    h_d[batch_index] = -1;
    h_u[batch_index] = 0; // 0 by definition
    h_w[batch_index] = 0; // 0 by definition
    h_b[batch_index] = 74; // solution vector

    // 2. Allocate device memory and offload input data to the device.
    double* d_w{};
    double* d_u{};
    double* d_d{};
    double* d_l{};
    double* d_s{};
    double* d_b{};

    constexpr size_t d_size = sizeof(*d_d) * N_b;

    HIP_CHECK(hipMalloc(&d_w, d_size));
    HIP_CHECK(hipMalloc(&d_u, d_size));
    HIP_CHECK(hipMalloc(&d_d, d_size));
    HIP_CHECK(hipMalloc(&d_l, d_size));
    HIP_CHECK(hipMalloc(&d_s, d_size));
    HIP_CHECK(hipMalloc(&d_b, d_size));

    HIP_CHECK(hipMemcpy(d_w, h_w.data(), d_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_u, h_u.data(), d_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_d, h_d.data(), d_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_l, h_l.data(), d_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_s, h_s.data(), d_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), d_size, hipMemcpyHostToDevice));

    // 3. Initialize rocSPARSE by creating a handle.
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // 4. Obtain the required buffer size.
    // This function is non blocking and executed asynchronously with respect to the host.
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_dgpsv_interleaved_batch_buffer_size(handle,
                                                                  rocsparse_gpsv_interleaved_alg_qr,
                                                                  m,
                                                                  d_s,
                                                                  d_l,
                                                                  d_d,
                                                                  d_u,
                                                                  d_w,
                                                                  d_b,
                                                                  batch_count,
                                                                  batch_stride,
                                                                  &buffer_size));
    // No synchronization with the device is needed because for scalar results, when using host
    // pointer mode (the default pointer mode) this function blocks the CPU till the GPU has copied
    // the results back to the host. See rocsparse_set_pointer_mode.

    // Allocate temporary buffer.
    void* temp_buffer{};
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // 5. Call interleaved batched pentadiagonal solver.
    // This function is non blocking and executed asynchronously with respect to the host.
    ROCSPARSE_CHECK(rocsparse_dgpsv_interleaved_batch(handle,
                                                      rocsparse_gpsv_interleaved_alg_qr,
                                                      m,
                                                      d_s,
                                                      d_l,
                                                      d_d,
                                                      d_u,
                                                      d_w,
                                                      d_b,
                                                      batch_count,
                                                      batch_stride,
                                                      temp_buffer));

    // 6. Copy result matrix to host. This call synchronizes with the host.
    HIP_CHECK(hipMemcpy(h_b.data(), d_b, d_size, hipMemcpyDeviceToHost));

    // 7. Free rocSPARSE resources and device memory.
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    HIP_CHECK(hipFree(temp_buffer));
    HIP_CHECK(hipFree(d_w));
    HIP_CHECK(hipFree(d_u));
    HIP_CHECK(hipFree(d_d));
    HIP_CHECK(hipFree(d_l));
    HIP_CHECK(hipFree(d_s));
    HIP_CHECK(hipFree(d_b));

    // 8. Check convergence.
    if(std::isnan(h_b[0]))
    {
        std::cout << "gpsv does not converge." << std::endl;
    }

    // 9. Print result matrix, check errors.
    // clang-format off
    constexpr std::array<double, N_b> expected = {
        1, 2, 3, 4, 5,
        3, 14, 15, 92, 65
    };
    // clang-format on

    constexpr double eps    = 1.0e5 * std::numeric_limits<double>::epsilon();
    rocsparse_int    errors = 0;
    for(rocsparse_int b = 0; b < batch_count; ++b)
    {
        std::cout << "batch #" << b << ":\t";
        for(rocsparse_int i = 0; i < m; ++i)
        {
            rocsparse_int j = batch_stride * i + b;
            errors += std::fabs(h_b[j] - expected[b * m + i]) > eps;
            std::cout << h_b[j] << " ";
        }
        std::cout << std::endl;
    }

    // Print validation result.
    return report_validation_result(errors);
}
