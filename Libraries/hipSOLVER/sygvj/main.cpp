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
#include "hipsolver_utils.hpp"

#include <hipsolver/hipsolver.h>

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

int main(const int /*argc*/, char* /*argv*/[])
{
    // Initialize leading dimensions of input matrices A and B.
    constexpr int n   = 3;
    constexpr int lda = n;
    constexpr int ldb = n;

    // Initialize vectors with elements of A and B:
    //     | 3.5 0.5 0.0 |
    // A = | 0.5 3.5 0.0 |
    //     | 0.0 0.0 2.0 |
    //     | 10   2   3 |
    // B = |  2  10   5 |
    //     |  3   5  10 |
    const std::vector<double> A{3.5, 0.5, 0.0, 0.5, 3.5, 0.0, 0.0, 0.0, 2.0};
    const std::vector<double> B{10.0, 2.0, 3.0, 2.0, 10.0, 5.0, 3.0, 5.0, 10.0};

    // Define input matrices size.
    const unsigned int size_A = lda * n;
    const unsigned int size_B = ldb * n;

    // Allocate device memory for the input and outputs.
    double* d_A{};
    double* d_B{};
    double* d_W{};
    int*    d_sygvj_info{};
    HIP_CHECK(hipMalloc(&d_A, sizeof(double) * size_A));
    HIP_CHECK(hipMalloc(&d_B, sizeof(double) * size_B));
    HIP_CHECK(hipMalloc(&d_W, sizeof(double) * n));
    HIP_CHECK(hipMalloc(&d_sygvj_info, sizeof(int)));

    // Copy input matrices A and B from host to device.
    HIP_CHECK(hipMemcpy(d_A, A.data(), sizeof(double) * size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, B.data(), sizeof(double) * size_B, hipMemcpyHostToDevice));

    // Use the hipSOLVER API to create a handle.
    hipsolverHandle_t hipsolver_handle;
    HIPSOLVER_CHECK(hipsolverCreate(&hipsolver_handle));

    // Working space variables.
    int     lwork{};
    double* d_work{};

    // Configuration of syevj.
    hipsolverSyevjInfo_t syevj_params = nullptr;

    const double              tol        = 1.e-10;
    const int                 max_sweeps = 15;
    const hipsolverEigType_t  itype      = HIPSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
    const hipsolverEigMode_t  jobz       = HIPSOLVER_EIG_MODE_VECTOR; // compute eigenvectors
    const hipsolverFillMode_t uplo       = HIPSOLVER_FILL_MODE_LOWER;

    HIPSOLVER_CHECK(hipsolverCreateSyevjInfo(&syevj_params));
    // Default value of tolerance is machine zero.
    HIPSOLVER_CHECK(hipsolverXsyevjSetTolerance(syevj_params, tol));
    // Default value of max. sweeps is 100.
    HIPSOLVER_CHECK(hipsolverXsyevjSetMaxSweeps(syevj_params, max_sweeps));

    // Query and allocate working space.
    HIPSOLVER_CHECK(hipsolverDsygvj_bufferSize(hipsolver_handle,
                                               itype,
                                               jobz,
                                               uplo,
                                               n,
                                               d_A,
                                               lda,
                                               d_B,
                                               ldb,
                                               d_W,
                                               &lwork,
                                               syevj_params));
    HIP_CHECK(hipMalloc(&d_work, sizeof(double) * lwork));

    // Compute spectrum (written to d_W) and eigenvectors (writtten to d_A).
    HIPSOLVER_CHECK(hipsolverDsygvj(hipsolver_handle,
                                    itype,
                                    jobz,
                                    uplo,
                                    n,
                                    d_A,
                                    lda,
                                    d_B,
                                    ldb,
                                    d_W,
                                    d_work,
                                    lwork,
                                    d_sygvj_info,
                                    syevj_params));

    // Check returned info value.
    int syevj_info{};
    HIP_CHECK(hipMemcpy(&syevj_info, d_sygvj_info, sizeof(syevj_info), hipMemcpyDeviceToHost));
    int errors{};
    if(syevj_info == 0)
    {
        std::cout << "Eigenvalues successfully computed: ";

        // Copy the resulting vector of eigenvalues to the host.
        std::vector<double> W(n, 0);
        HIP_CHECK(hipMemcpy(W.data(), d_W, sizeof(double) * n, hipMemcpyDeviceToHost));

        // Print eigenvalues and compare them with the expected values.
        const std::vector<double> expected_eigenvalues{0.158660256604, 0.370751508101882, 0.6};
        for(int i = 0; i < n; ++i)
        {
            std::cout << W[i] << (i < n - 1 ? ", " : "\n");
            errors += std::abs(expected_eigenvalues[i] - W[i]) > tol;
        }

        // Copy the resulting vector of eigenvectors to the host.
        std::vector<double> V(size_A, 0);
        HIP_CHECK(hipMemcpy(V.data(), d_A, sizeof(double) * size_A, hipMemcpyDeviceToHost));

        std::cout << "Eigenvectors:" << std::endl;
        for(int i = 0; i < n; ++i)
        {
            std::cout << "{ ";
            for(int j = 0; j < n; ++j)
            {
                std::cout << V[i * lda + j] << (j < n - 1 ? ", " : " }\n");
            }
        }

        // Numerical results of syevj.
        double residual{};
        int    executed_sweeps{};

        HIPSOLVER_CHECK(hipsolverXsyevjGetResidual(hipsolver_handle, syevj_params, &residual));
        HIPSOLVER_CHECK(hipsolverXsyevjGetSweeps(hipsolver_handle, syevj_params, &executed_sweeps));

        std::cout << "Residual = " << residual << std::endl;
        std::cout << "Number of executed sweeps = " << executed_sweeps << std::endl;
    }
    else
    {
        if(syevj_info < 0)
        {
            std::cout << "Parameter number " << -syevj_info << " is wrong.";
        }
        else if(syevj_info <= n)
        {
            std::cout << "Leading minor of order " << syevj_info
                      << " of B is not positive definite.";
        }
        else if(syevj_info == n + 1)
        {
            std::cout << "Sygvj does not converge, error " << syevj_info << ".";
        }
        else
        {
            std::cout << "Unknown error " << syevj_info << ".";
        }
        std::cout << std::endl;
        ++errors;
    }

    // Free resources.
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_W));
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_sygvj_info));
    HIPSOLVER_CHECK(hipsolverDestroySyevjInfo(syevj_params));
    HIPSOLVER_CHECK(hipsolverDestroy(hipsolver_handle));

    // Print validation result.
    return report_validation_result(errors);
}
