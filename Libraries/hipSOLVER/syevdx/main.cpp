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
#include <vector>

int main(const int /*argc*/, char* /*argv*/[])
{
    // Initialize leading dimensions of input matrix A.
    constexpr int n   = 3;
    constexpr int lda = n;

    // Define input matrix size (in number of elements).
    const unsigned int size_A = lda * n;

    // Initialize symmetric input matrix A.
    //     | 3.5 0.5 0.0 |
    // A = | 0.5 3.5 0.0 |
    //     | 0.0 0.0 2.0 |
    std::vector<double> A{3.5, 0.5, 0.0, 0.5, 3.5, 0.0, 0.0, 0.0, 2.0};

    // Allocate device memory for the input and outputs and copy input matrix A from host to device.
    double* d_A{};
    double* d_W{};
    double* d_X{}; /* auxiliary device matrix for solution checking */
    int*    d_syevdx_info{};
    HIP_CHECK(hipMalloc(&d_A, sizeof(double) * size_A));
    HIP_CHECK(hipMalloc(&d_W, sizeof(double) * n));
    HIP_CHECK(hipMalloc(&d_X, sizeof(double) * size_A));
    HIP_CHECK(hipMalloc(&d_syevdx_info, sizeof(int)));
    HIP_CHECK(hipMemcpy(d_A, A.data(), sizeof(double) * size_A, hipMemcpyHostToDevice));

    // Initialize the hipSOLVER Compatibility API by creating a handle.
    hipsolverDnHandle_t hipsolver_handle;
    HIPSOLVER_CHECK(hipsolverDnCreate(&hipsolver_handle));

    // Working space variables.
    int     lwork{}; /* size of workspace in bytes */
    double* d_work{}; /* device workspace */

    // Search interval variables.
    int nev{}; /* number of eigenvalues found */
    // The search interval can be selected to be (vl, vu].
    int vl{};
    int vu{};
    // It's also possible to obtain only the eigenvalues from the il-th to the iu-th found.
    int il = 1;
    int iu = 2;

    // Query and allocate working space.
    HIPSOLVER_CHECK(hipsolverDnDsyevdx_bufferSize(hipsolver_handle,
                                                  HIPSOLVER_EIG_MODE_VECTOR,
                                                  HIPSOLVER_EIG_RANGE_I,
                                                  HIPSOLVER_FILL_MODE_UPPER,
                                                  n,
                                                  d_A,
                                                  lda,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  &nev,
                                                  d_W,
                                                  &lwork));
    HIP_CHECK(hipMalloc(&d_work, sizeof(double) * lwork));

    // Compute the eigenvalues (written to d_W) and eigenvectors (written to d_A) of matrix A.
    HIPSOLVER_CHECK(hipsolverDnDsyevdx(hipsolver_handle,
                                       HIPSOLVER_EIG_MODE_VECTOR,
                                       HIPSOLVER_EIG_RANGE_I,
                                       HIPSOLVER_FILL_MODE_UPPER,
                                       n,
                                       d_A,
                                       lda,
                                       vl,
                                       vu,
                                       il,
                                       iu,
                                       &nev,
                                       d_W,
                                       d_work,
                                       lwork,
                                       d_syevdx_info));

    // Check returned info value.
    int syevdx_info{};
    HIP_CHECK(hipMemcpy(&syevdx_info, d_syevdx_info, sizeof(syevdx_info), hipMemcpyDeviceToHost));

    int errors{};

    if(syevdx_info < 0)
    {
        std::cout << -syevdx_info << "-th parameter is wrong.\n" << std::endl;
        errors++;
    }
    else if(syevdx_info > 0)
    {
        std::cout << "Computing eigenvalues did not converge.\n" << std::endl;
        errors++;
    }
    else
    {

        // Copy the resulting vector of eigenvalues to the host and print it to standard output.
        std::vector<double> W(nev, 0);
        HIP_CHECK(hipMemcpy(W.data(), d_W, sizeof(double) * nev, hipMemcpyDeviceToHost));

        std::cout << nev
                  << " eigenvalues successfully computed: " << format_range(W.begin(), W.end())
                  << std::endl;

        // Copy the resulting matrix of eigenvectors to the host.
        std::vector<double> X(size_A);
        HIP_CHECK(hipMemcpy(X.data(), d_A, sizeof(double) * size_A, hipMemcpyDeviceToHost));

        // Check the solution using the hipBLAS API.
        // Create a handle and enable passing scalar parameters from a pointer to host memory.
        hipblasHandle_t hipblas_handle;
        HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));
        HIPBLAS_CHECK(hipblasSetPointerMode(hipblas_handle, HIPBLAS_POINTER_MODE_HOST));

        // Validate the result by seeing if A * X - X * W is the zero matrix.
        // Define necessary constants.
        const double eps         = 1.0e5 * std::numeric_limits<double>::epsilon();
        const double h_one       = 1;
        const double h_minus_one = -1;
        double*      d_accum{}; /* cumulative device matrix */
        HIP_CHECK(hipMalloc(&d_accum, sizeof(double) * size_A));

        // Firstly, make accum = X * diag(W)
        HIP_CHECK(hipMemcpy(d_X, X.data(), sizeof(double) * size_A, hipMemcpyHostToDevice));
        HIPBLAS_CHECK(
            hipblasDdgmm(hipblas_handle, HIPBLAS_SIDE_RIGHT, n, n, d_X, lda, d_W, 1, d_accum, lda));

        // Secondly, make accum = A * X - accum.
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
                                   &h_minus_one,
                                   d_accum,
                                   lda));
        // Copy the result back to the host.
        HIP_CHECK(hipMemcpy(A.data(), d_accum, sizeof(double) * size_A, hipMemcpyDeviceToHost));

        // Free resources.
        HIP_CHECK(hipFree(d_accum));
        HIPBLAS_CHECK(hipblasDestroy(hipblas_handle));

        // Lastly, check if A is 0 for the eigenvalues and eigenvectors computed (i.e. check if
        // the first nev columns are 0).
        for(int i = 0; i < n * nev; ++i)
        {
            errors += std::fabs(A[i]) > eps;
        }
    }

    // Free resources.
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_W));
    HIP_CHECK(hipFree(d_X));
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_syevdx_info));
    HIPSOLVER_CHECK(hipsolverDnDestroy(hipsolver_handle));

    // Print validation result.
    return report_validation_result(errors);
}
