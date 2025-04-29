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

#include "cmdparser.hpp"
#include "example_utils.hpp"
#include "hipblas_utils.hpp"
#include "hipsolver_utils.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>

#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

int main(const int argc, char* argv[])
{
    // 1. Parse command line arguments.
    cli::Parser parser(argc, argv);
    parser.set_optional<int>("n", "n", 3, "Size of n x n input matrices");
    parser.set_optional<int>("c", "batch_count", 2, "Number of matrices in the input batch");
    parser.run_and_exit_if_error();

    // Get the n x n matrices size.
    const int n = parser.get<int>("n");
    if(n <= 0)
    {
        std::cout << "Value of 'n' should be greater than 0" << std::endl;
        return error_exit_code;
    }
    const int lda         = n;
    const int size_matrix = n * lda;

    // Get the batch size.
    const int batch_count = parser.get<int>("c");
    if(batch_count <= 0)
    {
        std::cout << "Batch size should be at least 1" << std::endl;
        return error_exit_code;
    }

    // 2. Allocate and initialize the host side inputs.
    std::vector<double> A(size_matrix * batch_count); // Input batch and resulting eigenvectors
    std::vector<double> W(n * batch_count); // Resulting eigenvalues

    // Random and symmetric initialization of the input batch matrices.
    std::default_random_engine             generator;
    std::uniform_real_distribution<double> distribution(0., 2.);
    auto random_number = [&]() { return distribution(generator); };

    for(int k = 0; k < batch_count * size_matrix; k += size_matrix)
    {
        for(int i = 0; i < n; ++i)
        {
            A[k + (lda + 1) * i] = random_number();
            for(int j = 0; j < i; ++j)
            {
                A[k + i * lda + j] = A[k + j * lda + i] = random_number();
            }
        }
    }

    // 3. Allocate device memory and copy input data from host.
    double* d_A{};
    double* d_W{};
    int*    d_info{};

    HIP_CHECK(hipMalloc(&d_A, sizeof(double) * A.size()));
    HIP_CHECK(hipMalloc(&d_W, sizeof(double) * W.size()));
    HIP_CHECK(hipMalloc(&d_info, batch_count * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_A, A.data(), sizeof(double) * A.size(), hipMemcpyHostToDevice));

    // 4. Initialize hipSOLVER by creating a handle.
    hipsolverHandle_t hipsolver_handle;
    HIPSOLVER_CHECK(hipsolverCreate(&hipsolver_handle));

    // 5. Set parameters for hipSOLVER's syevjBatched function.
    const hipsolverEigMode_t  jobz = HIPSOLVER_EIG_MODE_VECTOR;
    const hipsolverFillMode_t uplo = HIPSOLVER_FILL_MODE_LOWER;

    hipsolverSyevjInfo_t params;
    HIPSOLVER_CHECK(hipsolverCreateSyevjInfo(&params));
    HIPSOLVER_CHECK(hipsolverXsyevjSetMaxSweeps(params, 15));
    HIPSOLVER_CHECK(hipsolverXsyevjSetTolerance(params, 1.e-12));
    HIPSOLVER_CHECK(hipsolverXsyevjSetSortEig(params, 1));

    // 6. Query and allocate working space.
    int     lwork{}; /* size of workspace in bytes */
    double* d_work{}; /* device workspace */
    HIPSOLVER_CHECK(hipsolverDsyevjBatched_bufferSize(hipsolver_handle,
                                                      jobz,
                                                      uplo,
                                                      n,
                                                      d_A,
                                                      lda,
                                                      d_W,
                                                      &lwork,
                                                      params,
                                                      batch_count));
    HIP_CHECK(hipMalloc(&d_work, sizeof(double) * lwork));

    // 7. Invoke hipsolverDsyevjBatched to compute the eigenvalues (written to d_W) and
    // eigenvectors (written to d_A) of the matrices in the batch.
    HIPSOLVER_CHECK(hipsolverDsyevjBatched(hipsolver_handle,
                                           jobz,
                                           uplo,
                                           n,
                                           d_A,
                                           lda,
                                           d_W,
                                           d_work,
                                           lwork,
                                           d_info,
                                           params,
                                           batch_count));
    // 8. Check returned info value.
    int info{};
    HIP_CHECK(hipMemcpy(&info, d_info, sizeof(int), hipMemcpyDeviceToHost));

    int errors{};

    if(info < 0)
    {
        std::cout << -info << "-th parameter is wrong.\n" << std::endl;
        errors++;
    }
    else if(info > 0)
    {
        std::cout << "Computing eigenvalues did not converge.\n" << std::endl;
        errors++;
    }
    else
    {
        // 9. Copy results back to host. Use auxiliary matrix X for copying eigenvectors.
        std::vector<double> X(size_matrix * batch_count);

        HIP_CHECK(hipMemcpy(X.data(), d_A, sizeof(double) * X.size(), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(W.data(), d_W, sizeof(double) * W.size(), hipMemcpyDeviceToHost));

        // 10. Print eigenvalues and check solution using the hipBLAS API for each matrix of the batch.
        // Copy original input matrix to device.
        HIP_CHECK(hipMemcpy(d_A, A.data(), sizeof(double) * A.size(), hipMemcpyHostToDevice));

        // Define necessary constants and auxiliary matrices.
        const double eps         = 1.0e5 * std::numeric_limits<double>::epsilon();
        const double h_one       = 1;
        const double h_minus_one = -1;
        double*      d_accum{}; /* cumulative device matrix */
        double*      d_X{};
        HIP_CHECK(hipMalloc(&d_accum, sizeof(double) * size_matrix));
        HIP_CHECK(hipMalloc(&d_X, sizeof(double) * size_matrix));

        // Create a handle and enable passing scalar parameters from a pointer to host memory.
        hipblasHandle_t hipblas_handle;
        HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));
        HIPBLAS_CHECK(hipblasSetPointerMode(hipblas_handle, HIPBLAS_POINTER_MODE_HOST));

        for(int i = 0; i < batch_count; ++i)
        {
            const int eigvals_offset = i * n;
            const int eigvect_offset = i * size_matrix;

            // 10a. Print eigenvalues of matrix i of the batch.
            std::cout << "Eigenvalues successfully computed for matrix " << i << " of the batch: "
                      << format_range(W.begin() + eigvals_offset, W.begin() + eigvals_offset + n)
                      << std::endl;

            // 10b. Check the solution by seeing if A_i * X_i - X_i * diag(W_i) is the zero matrix.
            // Firstly, make accum = X_i * diag(W_i).
            HIP_CHECK(hipMemcpy(d_X,
                                X.data() + eigvect_offset,
                                sizeof(double) * size_matrix,
                                hipMemcpyHostToDevice));
            HIPBLAS_CHECK(hipblasDdgmm(hipblas_handle,
                                       HIPBLAS_SIDE_RIGHT,
                                       n,
                                       n,
                                       d_X,
                                       lda,
                                       d_W + eigvals_offset,
                                       1,
                                       d_accum,
                                       lda));

            // Secondly, make accum = A_i * X_i - accum.
            HIPBLAS_CHECK(hipblasDgemm(hipblas_handle,
                                       HIPBLAS_OP_N,
                                       HIPBLAS_OP_N,
                                       n,
                                       n,
                                       n,
                                       &h_one,
                                       d_A + eigvect_offset,
                                       lda,
                                       d_X,
                                       lda,
                                       &h_minus_one,
                                       d_accum,
                                       lda));
            // Copy the result back to the host.
            HIP_CHECK(hipMemcpy(A.data() + eigvect_offset,
                                d_accum,
                                sizeof(double) * size_matrix,
                                hipMemcpyDeviceToHost));
        }
        // Free resources.
        HIP_CHECK(hipFree(d_X));
        HIP_CHECK(hipFree(d_accum));
        HIPBLAS_CHECK(hipblasDestroy(hipblas_handle));

        // Check if A is 0.
        for(size_t i = 0; i < A.size(); ++i)
        {
            errors += std::fabs(A[i]) > eps;
        }
    }

    // 11. Clean up device allocations and print validation result.
    HIPSOLVER_CHECK(hipsolverDestroy(hipsolver_handle));
    HIPSOLVER_CHECK(hipsolverDestroySyevjInfo(params));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_W));
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_info));

    return report_validation_result(errors);
}
