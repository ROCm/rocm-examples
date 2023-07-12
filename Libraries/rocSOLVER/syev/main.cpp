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
#include "rocblas_utils.hpp"

#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include <hip/hip_runtime.h>

#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

int main(const int argc, char* argv[])
{
    // 1. Parse user input
    cli::Parser parser(argc, argv);
    parser.set_optional<rocblas_int>("n", "n", 3, "Size of n x n input matrix A");
    parser.run_and_exit_if_error();

    // Get the n x n matrix size
    const rocblas_int n = parser.get<rocblas_int>("n");
    if(n <= 0)
    {
        std::cout << "Value of 'n' should be greater than 0" << std::endl;
        return error_exit_code;
    }
    const rocblas_int lda = n;

    // 2. Data vectors
    std::vector<rocblas_double> A(n * lda); // Input matrix
    std::vector<rocblas_double> V(n * lda); // Resulting eigenvectors
    std::vector<rocblas_double> W(n); // Resulting eigenvalues

    // 3. Generate a random symmetric matrix
    std::default_random_engine                     generator;
    std::uniform_real_distribution<rocblas_double> distribution(0., 2.);
    auto random_number = [&]() { return distribution(generator); };

    for(int i = 0; i < n; i++)
    {
        A[(lda + 1) * i] = random_number();
        for(int j = 0; j < i; j++)
        {
            A[i * lda + j] = A[j * lda + i] = random_number();
        }
    }

    // 4. Set rocSOLVER parameters
    const rocblas_evect evect = rocblas_evect::rocblas_evect_original;
    const rocblas_fill  uplo  = rocblas_fill::rocblas_fill_lower;

    // 5. Reserve and copy data to device
    rocblas_double* d_A    = nullptr;
    rocblas_double* d_W    = nullptr;
    rocblas_int*    d_info = nullptr;

    HIP_CHECK(hipMalloc(&d_info, sizeof(rocblas_int)));
    HIP_CHECK(hipMalloc(&d_A, sizeof(rocblas_double) * A.size()));
    HIP_CHECK(hipMalloc(&d_W, sizeof(rocblas_double) * W.size()));
    HIP_CHECK(hipMemcpy(d_A, A.data(), sizeof(rocblas_double) * A.size(), hipMemcpyHostToDevice));

    // 6. Initialize rocBLAS.
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    // 7. Get and reserve the working space on device. Only n - 1 elements (super and subdiagonal's
    // size) are needed, as the matrix is symmetric and the leading diagonal is stored in d_W
    // (given that it converges to the eigenvalues).
    rocblas_int     d_work_size = n - 1;
    rocblas_double* d_work      = nullptr;
    HIP_CHECK(hipMalloc(&d_work, sizeof(rocblas_double) * d_work_size));

    // 8. Compute eigenvectors and eigenvalues
    ROCBLAS_CHECK(rocsolver_dsyev(handle, evect, uplo, n, d_A, lda, d_W, d_work, d_info));

    // 9. Get results from device.
    int info = 0;
    HIP_CHECK(hipMemcpy(V.data(), d_A, sizeof(rocblas_double) * V.size(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(W.data(), d_W, sizeof(rocblas_double) * W.size(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&info, d_info, sizeof(rocblas_int), hipMemcpyDeviceToHost));

    // 10. Print results.
    if(info == 0)
    {
        std::cout << "SYEV converges." << std::endl;
    }
    else if(info > 0)
    {
        std::cout << "SYEV does not converge (" << info << " elements did not converge)."
                  << std::endl;
    }

    std::cout << "\nGiven the n x n square input matrix A; we computed the linearly independent, "
                 "orthonormal eigenvectors V and the associated eigenvalues W."
              << std::endl;
    std::cout << "A = " << format_range(A.begin(), A.end()) << std::endl;
    std::cout << "W = " << format_range(W.begin(), W.end()) << std::endl;
    std::cout << "V = " << format_range(V.begin(), V.end()) << std::endl;

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
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_W));
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_info));

    return report_validation_result(errors);
}
