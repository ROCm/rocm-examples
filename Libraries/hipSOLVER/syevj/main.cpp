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
#include "hipsolver_utils.hpp"

#include <hip/hip_runtime.h>
#include <hipsolver/hipsolver.h>

#include <iostream>
#include <random>
#include <vector>

int main(const int argc, char* argv[])
{
    // 1. Parse user input
    cli::Parser parser(argc, argv);
    parser.set_optional<int>("n", "n", 3, "Size of n x n input matrix A");
    parser.run_and_exit_if_error();

    // Get the n x n matrix size
    const int n = parser.get<int>("n");
    if(n <= 0)
    {
        std::cout << "Value of 'n' should be greater than 0" << std::endl;
        return error_exit_code;
    }
    const int lda = n;

    // 2. Data vectors
    std::vector<double> A(n * lda); // Input matrix
    std::vector<double> V(n * lda); // Resulting eigenvectors
    std::vector<double> W(n); // resulting eigenvalues

    // 3. Generate a random symmetric matrix
    std::default_random_engine             generator;
    std::uniform_real_distribution<double> distribution(0., 2.);
    auto random_number = [&]() { return distribution(generator); };

    for(int i = 0; i < n; i++)
    {
        A[(lda + 1) * i] = random_number();
        for(int j = 0; j < i; j++)
        {
            A[i * lda + j] = A[j * lda + i] = random_number();
        }
    }

    // 4. Set hipsolver parameters
    const hipsolverEigMode_t  jobz = HIPSOLVER_EIG_MODE_VECTOR;
    const hipsolverFillMode_t uplo = HIPSOLVER_FILL_MODE_LOWER;

    hipsolverSyevjInfo_t params;
    HIPSOLVER_CHECK(hipsolverCreateSyevjInfo(&params));
    HIPSOLVER_CHECK(hipsolverXsyevjSetMaxSweeps(params, 100));
    HIPSOLVER_CHECK(hipsolverXsyevjSetTolerance(params, 1.e-7));
    HIPSOLVER_CHECK(hipsolverXsyevjSetSortEig(params, 1));

    // 5. Reserve and copy data to device
    double* d_A    = nullptr;
    double* d_W    = nullptr;
    int*    d_info = nullptr;

    HIP_CHECK(hipMalloc(&d_A, sizeof(double) * A.size()));
    HIP_CHECK(hipMalloc(&d_W, sizeof(double) * W.size()));
    HIP_CHECK(hipMalloc(&d_info, sizeof(int)));
    HIP_CHECK(hipMemcpy(d_A, A.data(), sizeof(double) * A.size(), hipMemcpyHostToDevice));

    // 6. Initialize hipsolver
    hipsolverHandle_t hipsolver_handle;
    HIPSOLVER_CHECK(hipsolverCreate(&hipsolver_handle));

    // 7. Get and reserve the working space on device.
    int     lwork  = 0;
    double* d_work = nullptr;
    HIPSOLVER_CHECK(
        hipsolverDsyevj_bufferSize(hipsolver_handle, jobz, uplo, n, d_A, lda, d_W, &lwork, params));

    HIP_CHECK(hipMalloc(&d_work, sizeof(double) * lwork));

    // 8. Compute eigenvectors and eigenvalues
    HIPSOLVER_CHECK(hipsolverDsyevj(hipsolver_handle,
                                    jobz,
                                    uplo,
                                    n,
                                    d_A,
                                    lda,
                                    d_W,
                                    d_work,
                                    lwork,
                                    d_info,
                                    params));
    // 9. Get results from host.
    int info = 0;
    HIP_CHECK(hipMemcpy(V.data(), d_A, sizeof(double) * V.size(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(W.data(), d_W, sizeof(double) * W.size(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&info, d_info, sizeof(int), hipMemcpyDeviceToHost));

    // 10. Print results
    if(info == 0)
    {
        std::cout << "SYEVJ converges." << std::endl;
    }
    else if(info > 0)
    {
        std::cout << "SYEVJ does not converge (" << info << " elements did not converge)."
                  << std::endl;
    }

    std::cout << "\nGiven the n x n square input matrix A; we computed the linearly independent "
                 "eigenvectors V and the associated eigenvalues W."
              << std::endl;
    std::cout << "A = " << format_range(A.begin(), A.end()) << std::endl;
    std::cout << "W = " << format_range(W.begin(), W.end()) << std::endl;
    std::cout << "V = " << format_range(V.begin(), V.end()) << std::endl;

    int    sweeps   = 0;
    double residual = 0;
    HIPSOLVER_CHECK(hipsolverXsyevjGetSweeps(hipsolver_handle, params, &sweeps));
    HIPSOLVER_CHECK(hipsolverXsyevjGetResidual(hipsolver_handle, params, &residual));

    std::cout << "\nWhich was computed in " << sweeps << " sweeps, with a residual of " << residual
              << std::endl;

    // 11. Validate that 'AV == VD' and that 'AV - VD == 0'.
    std::cout << "\nLet D be the diagonal constructed from W.\n"
              << "The right multiplication of A * V should result in V * D [AV == VD]:"
              << std::endl;

    // Right multiplication of the input matrix with the eigenvectors.
    std::vector<double> AV(n * lda);
    multiply_matrices(1.0, 0.0, n, n, n, A.data(), lda, 1, V.data(), 1, lda, AV.data(), lda);
    std::cout << "AV = " << format_range(AV.begin(), AV.end()) << std::endl;

    // Construct the diagonal D from eigenvalues W.
    std::vector<double> D(n * n);
    for(int i = 0; i < n; i++)
    {
        D[(n + 1) * i] = W[i];
    }

    // Scale eigenvectors V with W by multiplying V with D.
    std::vector<double> VD(n * lda);
    multiply_matrices(1.0, 0.0, n, n, n, V.data(), 1, lda, D.data(), lda, 1, VD.data(), lda);
    std::cout << "VD = " << format_range(VD.begin(), VD.end()) << std::endl;

    double epsilon = 1.0e5 * std::numeric_limits<double>::epsilon();
    int    errors  = 0;
    double mse     = 0;
    for(int i = 0; i < n * n; i++)
    {
        double diff = (AV[i] - VD[i]);
        diff *= diff;
        mse += diff;

        errors += (diff > epsilon);
    }
    mse /= n * n;
    std::cout << "\nMean Square Error of [AV == VD]:\n  " << mse << std::endl;

    // 12. Clean up device allocations.
    HIPSOLVER_CHECK(hipsolverDestroy(hipsolver_handle));
    HIPSOLVER_CHECK(hipsolverDestroySyevjInfo(params));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_W));
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_info));

    return report_validation_result(errors);
}
