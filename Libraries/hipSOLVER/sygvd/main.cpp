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

#include "example_utils.hpp"
#include "hipblas_utils.hpp"
#include "hipsolver_utils.hpp"

#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

int main()
{
    // Initialize dimensions, leading dimensions and number of elements of input matrices A and B.
    constexpr unsigned int n      = 3;
    constexpr unsigned int lda    = n;
    constexpr unsigned int ldb    = n;
    constexpr unsigned int size_A = lda * n;
    constexpr unsigned int size_B = ldb * n;

    // Initialize symmetric input matrix A.
    //     | 0.5 3.5 0.0 |
    // A = | 3.5 0.5 0.0 |
    //     | 0.0 0.0 2.0 |
    std::vector<double> A{0.5, 3.5, 0.0, 3.5, 0.5, 0.0, 0.0, 0.0, 2.0};

    // Initialize symmetric input matrix B as A but make it diagonally dominant
    // https://en.wikipedia.org/wiki/Diagonally_dominant_matrix so it's positive definite.
    //     | 4.0 3.5 0.0 |
    // B = | 3.5 4.0 0.0 |
    //     | 0.0 0.0 2.0 |
    std::vector<double> B(A);
    for(unsigned int i = 0; i < n; ++i)
    {
        double sum = 0;
        for(unsigned int j = 0; j < n; ++j)
        {
            sum += std::fabs(B[i + j * ldb]);
        }
        B[i + i * ldb] = sum;
    }

    // Allocate device memory for the inputs and outputs and copy input matrices from host to device.
    double* d_A{};
    double* d_B{};
    double* d_W{};
    double* d_X{}; /*Auxiliary device matrix for solution checking.*/
    int*    d_sygvd_info{};
    HIP_CHECK(hipMalloc(&d_A, sizeof(double) * size_A));
    HIP_CHECK(hipMalloc(&d_B, sizeof(double) * size_B));
    HIP_CHECK(hipMalloc(&d_W, sizeof(double) * n));
    HIP_CHECK(hipMalloc(&d_X, sizeof(double) * size_A));
    HIP_CHECK(hipMalloc(&d_sygvd_info, sizeof(int)));
    HIP_CHECK(hipMemcpy(d_A, A.data(), sizeof(double) * size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, B.data(), sizeof(double) * size_B, hipMemcpyHostToDevice));

    // Use the hipSOLVER API to create a handle.
    hipsolverHandle_t hipsolver_handle;
    HIPSOLVER_CHECK(hipsolverCreate(&hipsolver_handle));

    // Working space variables.
    int     lwork{};
    double* d_work{};

    // Query and allocate working space.
    HIPSOLVER_CHECK(hipsolverDsygvd_bufferSize(hipsolver_handle,
                                               HIPSOLVER_EIG_TYPE_1,
                                               HIPSOLVER_EIG_MODE_VECTOR,
                                               HIPSOLVER_FILL_MODE_UPPER,
                                               n,
                                               d_A,
                                               lda,
                                               d_B,
                                               ldb,
                                               d_W,
                                               &lwork));
    HIP_CHECK(hipMalloc(&d_work, sizeof(double) * lwork));

    // Compute the eigenvalues (written to d_W) and eigenvectors (written to d_A) of the pair (A, B).
    HIPSOLVER_CHECK(hipsolverDsygvd(hipsolver_handle,
                                    HIPSOLVER_EIG_TYPE_1,
                                    HIPSOLVER_EIG_MODE_VECTOR,
                                    HIPSOLVER_FILL_MODE_UPPER,
                                    n,
                                    d_A,
                                    lda,
                                    d_B,
                                    ldb,
                                    d_W,
                                    d_work,
                                    lwork,
                                    d_sygvd_info));

    // Check output info value.
    int sygvd_info{};
    HIP_CHECK(hipMemcpy(&sygvd_info, d_sygvd_info, sizeof(sygvd_info), hipMemcpyDeviceToHost));

    unsigned int errors{};

    if(sygvd_info < 0)
    {
        std::cout << -sygvd_info << "-th parameter is wrong.\n" << std::endl;
        errors++;
    }
    else if(sygvd_info > 0)
    {
        std::cout << "Computing eigenvalues did not converge.\n" << std::endl;
        errors++;
    }
    else
    {
        std::cout << "Eigenvalues successfully computed: ";

        // Copy the resulting vector of eigenvalues to the host and print it to standard output.
        std::vector<double> W(n);
        HIP_CHECK(hipMemcpy(W.data(), d_W, sizeof(double) * n, hipMemcpyDeviceToHost));

        if(!W.empty())
        {
            std::copy(W.begin(),
                      std::prev(W.end()),
                      std::ostream_iterator<double>(std::cout, ", "));
            std::cout << W.back() << "." << std::endl;
        }

        // Copy the resulting matrix of eigenvectors to the host.
        std::vector<double> X(size_A);
        HIP_CHECK(hipMemcpy(X.data(), d_A, sizeof(double) * size_A, hipMemcpyDeviceToHost));

        // Check the solution using the hipBLAS API.
        // Create a handle and enable passing scalar parameters from a pointer to host memory.
        hipblasHandle_t hipblas_handle;
        HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));
        HIPBLAS_CHECK(hipblasSetPointerMode(hipblas_handle, HIPBLAS_POINTER_MODE_HOST));

        // Validate the result by seeing if B * X * W - A * X is the zero matrix.
        const double eps         = 1.0e5 * std::numeric_limits<double>::epsilon();
        const double h_one       = 1;
        const double h_minus_one = -1;
        const double h_zero{};

        // Firstly, make A = A * X.
        HIP_CHECK(hipMemcpy(d_X, X.data(), sizeof(double) * size_A, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_A, A.data(), sizeof(double) * size_A, hipMemcpyHostToDevice));
        HIPBLAS_CHECK(hipblasDgemm(hipblas_handle,
                                   HIPBLAS_OP_N,
                                   HIPBLAS_OP_N,
                                   n,
                                   n,
                                   n,
                                   &h_one,
                                   d_A,
                                   lda,
                                   d_X,
                                   lda,
                                   &h_zero,
                                   d_A,
                                   lda));

        // Secondly, make X = X * diag(W).
        HIP_CHECK(hipMemcpy(d_X, X.data(), sizeof(double) * size_A, hipMemcpyHostToDevice));
        HIPBLAS_CHECK(
            hipblasDdgmm(hipblas_handle, HIPBLAS_SIDE_RIGHT, n, n, d_X, lda, d_W, 1, d_X, lda));

        // Thirdly, make A = B * X - A.
        HIP_CHECK(hipMemcpy(d_B, B.data(), sizeof(double) * size_B, hipMemcpyHostToDevice));
        HIPBLAS_CHECK(hipblasDgemm(hipblas_handle,
                                   HIPBLAS_OP_N,
                                   HIPBLAS_OP_N,
                                   n,
                                   n,
                                   n,
                                   &h_one,
                                   d_B,
                                   ldb,
                                   d_X,
                                   lda,
                                   &h_minus_one,
                                   d_A,
                                   lda))

        // Copy the result back to the host.
        HIP_CHECK(hipMemcpy(A.data(), d_A, sizeof(double) * size_A, hipMemcpyDeviceToHost));

        // Free hipBLAS handle.
        HIPBLAS_CHECK(hipblasDestroy(hipblas_handle));

        // Lastly, check if A is 0.
        for(unsigned int j = 0; j < n; ++j)
        {
            for(unsigned int i = 0; i < n; ++i)
            {
                errors += std::fabs(A[i + j * lda]) > eps;
            }
        }
    }

    // Free resources.
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_W));
    HIP_CHECK(hipFree(d_X));
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_sygvd_info));
    HIPSOLVER_CHECK(hipsolverDestroy(hipsolver_handle));

    // Print validation result.
    return report_validation_result(errors);
}
