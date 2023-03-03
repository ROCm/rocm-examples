// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "cmdparser.hpp"
#include "example_utils.hpp"
#include "hipsolver_utils.hpp"

#include <hipsolver/hipsolver.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

int main()
{
    int errors = 0;
    // The fill mode of the operations: specifies which triangular part, lower or upper, is
    // processed and replaced by the functions.
    constexpr hipsolverFillMode_t uplo = HIPSOLVER_FILL_MODE_LOWER;

    constexpr int nrhs = 1; // number of columns of B and X
    constexpr int n    = 3; // number of rows and columns of A
    constexpr int lda  = n; // leading dimension of A
    constexpr int ldbx = n; // leading dimension of B and X

    constexpr unsigned int size_a  = n * n;
    constexpr unsigned int size_bx = n * nrhs;

    // Matrices A to factorize.
    // clang-format off
    // A0 is not positive semi-definite, Cholesky factorization will fail.
    constexpr std::array<double, size_a> a0 = {1.0, 2.0,  3.0,
                                               2.0, 4.0,  5.0,
                                               3.0, 5.0, 12.0};
    // A1 is positive semi-definite, Cholesky factorization will succeed.
    constexpr std::array<double, size_a> a1 = {1.0, 2.0,  3.0,
                                               2.0, 5.0,  5.0,
                                               3.0, 5.0, 12.0};
    // Result of the linear system.
    constexpr std::array<double, size_bx> b = {1.0, 1.0, 1.0};

    // clang-format on
    double* d_a0{};
    double* d_a1{};
    HIP_CHECK(hipMalloc(&d_a0, sizeof(double) * size_a));
    HIP_CHECK(hipMalloc(&d_a1, sizeof(double) * size_a));
    HIP_CHECK(hipMemcpy(d_a0, a0.data(), sizeof(double) * size_a, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_a1, a1.data(), sizeof(double) * size_a, hipMemcpyHostToDevice));

    double* d_b{};
    HIP_CHECK(hipMalloc(&d_b, sizeof(double) * size_bx));
    HIP_CHECK(hipMemcpy(d_b, b.data(), sizeof(double) * size_bx, hipMemcpyHostToDevice));

    // Use the hipSOLVER API to create a handle.
    hipsolverHandle_t handle;
    HIPSOLVER_CHECK(hipsolverCreate(&handle));

    // Query working space size for potrf (A0 and A1 are the same size).
    int lwork_potrf{};
    HIPSOLVER_CHECK(hipsolverDpotrf_bufferSize(handle, uplo, n, d_a0, lda, &lwork_potrf));

    // Query working space size for potrs (A0 and A1 are the same size).
    int lwork_potrs{};
    HIPSOLVER_CHECK(
        hipsolverDpotrs_bufferSize(handle, uplo, n, nrhs, d_a1, lda, d_b, ldbx, &lwork_potrs));

    // Allocate working space.
    int     lwork = std::max(lwork_potrf, lwork_potrs);
    double* d_work{};
    HIP_CHECK(hipMalloc(&d_work, lwork));

    // Allocate space for device function return status.
    int* d_info{};
    HIP_CHECK(hipMalloc(&d_info, sizeof(int)));

    int info{};
    { // Process A0.
        // Perform Cholesky decomposition.
        HIPSOLVER_CHECK(hipsolverDpotrf(handle, uplo, n, d_a0, lda, d_work, lwork, d_info));

        // Copy device output data to host.
        HIP_CHECK(hipMemcpy(&info, d_info, sizeof(int), hipMemcpyDeviceToHost));
        if(info == 0)
        {
            std::cout << "Cholesky factorization of A0 succeeded, which should not be the case "
                         "since A0 is not positive semi-definite.\n";
            errors++;
        }
    }
    { // Process A1.
        // Perform Cholesky decomposition.
        HIPSOLVER_CHECK(hipsolverDpotrf(handle, uplo, n, d_a1, lda, d_work, lwork, d_info));

        // Copy device output data to host.
        HIP_CHECK(hipMemcpy(&info, d_info, sizeof(int), hipMemcpyDeviceToHost));
        if(info != 0)
        {
            std::cout << "Cholesky factorization of A1 failed, which should not be the case "
                         "since A1 is positive semi-definite.\n";
            errors++;
        }
    }

    // Solve A1 * X = B. Result will be placed in B.
    HIPSOLVER_CHECK(
        hipsolverDpotrs(handle, uplo, n, nrhs, d_a1, lda, d_b, ldbx, d_work, lwork, d_info));

    // Copy device output data to host.
    std::vector<double> x(size_bx);
    HIP_CHECK(hipMemcpy(x.data(), d_b, sizeof(double) * size_bx, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&info, d_info, sizeof(int), hipMemcpyDeviceToHost));

    // Print trace message.
    hipsolver_print_info(info);

    // Free all device resources.
    HIP_CHECK(hipFree(d_info));
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_a1));
    HIP_CHECK(hipFree(d_a0));
    HIPSOLVER_CHECK(hipsolverDestroy(handle));

    // Validate the result by performing A1 * X = B' and checking that B = B'.
    std::vector<double> b_inferred(size_bx);
    multiply_matrices(1.0,
                      0.0,
                      n,
                      1,
                      n,
                      a1.data(),
                      1,
                      lda,
                      x.data(),
                      1,
                      ldbx,
                      b_inferred.data(),
                      ldbx);
    constexpr double eps = 1.0e5 * std::numeric_limits<double>::epsilon();
    for(unsigned int i = 0; i < size_bx; i++)
    {
        errors += std::abs(b[i] - b_inferred[i]) > eps;
    }

    return report_validation_result(errors);
}
