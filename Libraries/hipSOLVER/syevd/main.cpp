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
#include "hipsolver_utils.hpp"

#include <hipsolver/hipsolver.h>

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

int main(const int /*argc*/, char* /*argv*/[])
{
    // Initialize leading dimensions of input matrix A.
    constexpr int n   = 3;
    constexpr int lda = n;

    // Initialize vector with elements of A:
    //     | 3.5 0.5 0.0 |
    // A = | 0.5 3.5 0.0 |
    //     | 0.0 0.0 2.0 |
    const std::vector<double> A{3.5, 0.5, 0.0, 0.5, 3.5, 0.0, 0.0, 0.0, 2.0};

    // Define input matrix size.
    const unsigned int size_A = lda * n;

    // Allocate device memory for the input and outputs and copy input matrix A from host to device.
    double* d_A{};
    double* d_W{};
    int*    d_syevd_info{};
    HIP_CHECK(hipMalloc(&d_A, sizeof(double) * size_A));
    HIP_CHECK(hipMalloc(&d_W, sizeof(double) * n));
    HIP_CHECK(hipMalloc(&d_syevd_info, sizeof(int)));
    HIP_CHECK(hipMemcpy(d_A, A.data(), sizeof(double) * size_A, hipMemcpyHostToDevice));

    // Use the hipSOLVER API to create a handle.
    hipsolverHandle_t hipsolver_handle;
    HIPSOLVER_CHECK(hipsolverCreate(&hipsolver_handle));

    // Working space variables.
    int     lwork{};
    double* d_work{};

    // Query and allocate working space.
    HIPSOLVER_CHECK(hipsolverDsyevd_bufferSize(hipsolver_handle,
                                               HIPSOLVER_EIG_MODE_NOVECTOR,
                                               HIPSOLVER_FILL_MODE_UPPER,
                                               n,
                                               d_A,
                                               lda,
                                               d_W,
                                               &lwork));
    HIP_CHECK(hipMalloc(&d_work, sizeof(double) * lwork));

    // Compute the eigenvalues (written to d_W).
    HIPSOLVER_CHECK(hipsolverDsyevd(hipsolver_handle,
                                    HIPSOLVER_EIG_MODE_NOVECTOR,
                                    HIPSOLVER_FILL_MODE_UPPER,
                                    n,
                                    d_A,
                                    lda,
                                    d_W,
                                    d_work,
                                    lwork,
                                    d_syevd_info));

    // Check returned info value.
    int syevd_info{};
    HIP_CHECK(hipMemcpy(&syevd_info, d_syevd_info, sizeof(syevd_info), hipMemcpyDeviceToHost));
    int errors{};
    if(syevd_info == 0)
    {
        std::cout << "Eigenvalues successfully computed: ";

        // Copy the resulting vector of eigenvalues to the host.
        std::vector<double> W(n);
        HIP_CHECK(hipMemcpy(W.data(), d_W, sizeof(double) * n, hipMemcpyDeviceToHost));

        // Print eigenvalues and compare them with the expected values.
        const std::vector<double> expected_eigenvalues{2.0, 3.0, 4.0};
        auto                      expected_it = expected_eigenvalues.begin();
        const double              eps         = 1.0e5 * std::numeric_limits<double>::epsilon();
        for(const auto eigenvalue : W)
        {
            std::cout << eigenvalue << ", ";
            errors += std::abs(*expected_it++ - eigenvalue) > eps;
        }
        std::cout << std::endl;
    }
    else
    {
        std::cout << "Computing eigenvalues did not converge.";
        ++errors;
    }

    // Free resources.
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_W));
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_syevd_info));
    HIPSOLVER_CHECK(hipsolverDestroy(hipsolver_handle));

    // Print validation result.
    return report_validation_result(errors);
}
