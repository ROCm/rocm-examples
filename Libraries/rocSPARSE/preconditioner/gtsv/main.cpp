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
    //      A         *       X        =       B
    // ( 1 -5  0  0 )   (  1  2  0   )   (  1  7 -2.5 )
    // ( 5 -2 -3  0 )   (  0 -1  0.5 )   ( -7  3  8   )
    // ( 0  3  1  4 ) * (  4  3 -3   ) = ( -4  0  6.5 )
    // ( 0  0  1  2 )   ( -2  0  2   )   (  0  3  1   )

    // Number of rows and columns of the input matrix.
    constexpr rocsparse_int m   = 4;
    constexpr rocsparse_int n   = 3;
    constexpr rocsparse_int ldb = m;

    // A matrix
    std::array<double, m> h_u = {-5, -3, 4, 0};
    std::array<double, m> h_d = {1, -2, 1, 2};
    std::array<double, m> h_l = {0, 5, 3, 1};

    // B matrix
    constexpr size_t N_B = ldb * n;

    // clang-format off
    std::array<double, N_B> h_B = {
        1, -7, -4, 0,
        7,  3, 0,  3,
        -2.5,  8,  6.5,  1
    };
    // clang-format on

    // 2. Allocate device memory and offload input data to the device.
    double* d_u{};
    double* d_d{};
    double* d_l{};
    double* d_B{};

    constexpr size_t d_size = sizeof(*d_d) * m;
    constexpr size_t B_size = sizeof(*d_B) * N_B;

    HIP_CHECK(hipMalloc(&d_u, d_size));
    HIP_CHECK(hipMalloc(&d_d, d_size));
    HIP_CHECK(hipMalloc(&d_l, d_size));
    HIP_CHECK(hipMalloc(&d_B, B_size));

    HIP_CHECK(hipMemcpy(d_u, h_u.data(), d_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_d, h_d.data(), d_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_l, h_l.data(), d_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), B_size, hipMemcpyHostToDevice));

    // 3. Initialize rocSPARSE by creating a handle.
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // 4. Obtain the required buffer size.
    // This function is non blocking and executed asynchronously with respect to the host.
    size_t buffer_size;
    ROCSPARSE_CHECK(
        rocsparse_dgtsv_buffer_size(handle, m, n, d_l, d_d, d_u, d_B, ldb, &buffer_size));
    // No synchronization with the device is needed because for scalar results, when using host
    // pointer mode (the default pointer mode) this function blocks the CPU till the GPU has copied
    // the results back to the host. See rocsparse_set_pointer_mode.

    // Allocate temporary buffer.
    void* temp_buffer{};
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // 5. Call GTSV tridiagonal solver.
    // This function is non blocking and executed asynchronously with respect to the host.
    ROCSPARSE_CHECK(rocsparse_dgtsv(handle, m, n, d_l, d_d, d_u, d_B, ldb, temp_buffer));

    // 6. Copy result matrix to host. This call synchronizes with the host.
    HIP_CHECK(hipMemcpy(h_B.data(), d_B, B_size, hipMemcpyDeviceToHost));

    // 7. Free rocSPARSE resources and device memory.
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    HIP_CHECK(hipFree(temp_buffer));
    HIP_CHECK(hipFree(d_u));
    HIP_CHECK(hipFree(d_d));
    HIP_CHECK(hipFree(d_l));
    HIP_CHECK(hipFree(d_B));

    // 8. Check convergence.
    if(std::isnan(h_B[0]))
    {
        std::cout << "GTSV does not converge." << std::endl;
    }

    // 9. Print result matrix, check errors.
    // clang-format off
    constexpr std::array<double, N_B> expected = {
        1, 0, 4, -2,
        2, -1, 3, 0,
        0, 0.5, -3, 2
    };
    // clang-format on

    constexpr double eps    = 1.0e5 * std::numeric_limits<double>::epsilon();
    int              errors = 0;

    std::cout << "X = " << std::endl;
    for(int i = 0; i < m; ++i)
    {
        std::cout << "    (";
        for(int j = 0; j < n; ++j)
        {
            int idx = i + j * m;
            std::printf("%7.2lf", h_B[idx]);

            errors += std::fabs(h_B[idx] - expected[idx]) > eps;
        }
        std::cout << ")" << std::endl;
    }

    // Print validation result.
    return report_validation_result(errors);
}
