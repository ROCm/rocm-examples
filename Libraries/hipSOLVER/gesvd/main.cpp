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

#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

int main(const int argc, char* argv[])
{
    // Parse user inputs.
    cli::Parser parser(argc, argv);
    parser.set_optional<int>("m", "m", 3, "Number of rows of input matrix A");
    parser.set_optional<int>("n", "n", 2, "Number of columns of input matrix A");
    parser.run_and_exit_if_error();

    // Get input matrix rows (m) and columns (n).
    const int m = parser.get<int>("m");
    if(m <= 0)
    {
        std::cout << "Value of 'm' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    const int n = parser.get<int>("n");
    if(n <= 0)
    {
        std::cout << "Value of 'n' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    // Initialize leading dimensions of input matrix A and output singular vector matrices.
    const int lda = m;
    const int ldu = m;
    const int ldv = n;

    // Define input and output matrices' sizes.
    const unsigned int size_A   = lda * n;
    const unsigned int size_U   = ldu * m;
    const unsigned int size_V_H = ldv * n;
    const unsigned int size_S   = std::min(m, n);

    // Initialize input matrix with sequence 1, 2, 3, ... .
    std::vector<double> A(size_A);
    std::iota(A.begin(), A.end(), 1.0);

    // We want to obtain the decomposition A = U * S * V_H. Initialize the right-hand matrices:
    // - U is an m x m unitary matrix, whose columns are the "left singular vectors".
    // - S is an m x n diagonal matrix, whose diagonal values are the "singular values". We store
    //     a vector with min(m,n) values instead of the whole matrix.
    // - V_H is an n x n unitary matrix, whose rows are the "right singular vectors".
    std::vector<double> U(size_U, 0);
    std::vector<double> S(size_S, 0);
    std::vector<double> V_H(size_V_H, 0);

    // Convergence information for the BDSQR algorithm used internally, it specifies how many
    // superdiagonals of the intermediate bidiagonal form did not converge to zero.
    int* d_bdsqr_info{};
    int  bdsqr_info{};

    // Allocate device memory for the matrices needed and copy input matrix A from host to device.
    double* d_A{};
    double* d_S{};
    double* d_U{};
    double* d_V_H{};
    double* d_W{}; // W = S * V_H, for solution checking.
    HIP_CHECK(hipMalloc(&d_A, sizeof(double) * size_A));
    HIP_CHECK(hipMalloc(&d_S, sizeof(double) * size_S));
    HIP_CHECK(hipMalloc(&d_U, sizeof(double) * size_U));
    HIP_CHECK(hipMalloc(&d_V_H, sizeof(double) * size_V_H));
    HIP_CHECK(hipMalloc(&d_W, sizeof(double) * size_A));
    HIP_CHECK(hipMalloc(&d_bdsqr_info, sizeof(int)));
    HIP_CHECK(hipMemcpy(d_A, A.data(), sizeof(double) * size_A, hipMemcpyHostToDevice));

    // Define how left and right singular vectors are calculated and stored.
    const signed char left_svect
        = 'A'; // All m columns of U (left singular vectors) are calculated.
    const signed char right_svect
        = 'A'; // All n columns of V_H (right singular vectors) are calculated.

    // Use the hipSOLVER API to create a handle.
    hipsolverHandle_t hipsolver_handle;
    HIPSOLVER_CHECK(hipsolverCreate(&hipsolver_handle));

    // Working space variables.
    int     lwork   = 0; /*Size of working space*/
    double* d_work  = nullptr; /*Working space*/
    double* d_rwork = nullptr; /*Unconverged superdiagonal elements of an upper bidiagonal matrix*/

    // Query working space.
    HIPSOLVER_CHECK(
        hipsolverDgesvd_bufferSize(hipsolver_handle, left_svect, right_svect, m, n, &lwork));
    HIP_CHECK(hipMalloc(&d_work, sizeof(double) * lwork));

    // Compute the singular values (vector S) and singular vectors (matrices U and V_H) of A.
    HIPSOLVER_CHECK(hipsolverDgesvd(hipsolver_handle,
                                    left_svect,
                                    right_svect,
                                    m,
                                    n,
                                    d_A,
                                    lda,
                                    d_S,
                                    d_U,
                                    ldu,
                                    d_V_H,
                                    ldv,
                                    d_work,
                                    lwork,
                                    d_rwork,
                                    d_bdsqr_info));
    // Copy device output data to host.
    HIP_CHECK(hipMemcpy(U.data(), d_U, sizeof(double) * size_U, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(S.data(), d_S, sizeof(double) * size_S, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(V_H.data(), d_V_H, sizeof(double) * size_V_H, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&bdsqr_info, d_bdsqr_info, sizeof(int), hipMemcpyDeviceToHost));

    // Print trace message for BDSQR.
    if(bdsqr_info == 0)
    {
        std::cout << "Internal BDSQR converges." << std::endl;
    }
    else if(bdsqr_info > 0)
    {
        std::cout << "Internal BDSQR does not converge (" << bdsqr_info
                  << "elements did not converge to 0)." << std::endl;
    }

    // Check the solution using the hipBLAS API.
    // Create a handle and enable passing scalar parameters from a pointer to host memory.
    hipblasHandle_t hipblas_handle;
    HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));
    HIPBLAS_CHECK(hipblasSetPointerMode(hipblas_handle, HIPBLAS_POINTER_MODE_HOST));

    // Validate the result by seeing if U * S * V_H - A is the zero matrix.
    const double eps         = 1.0e5 * std::numeric_limits<double>::epsilon();
    const double h_one       = 1;
    const double h_minus_one = -1;
    unsigned int errors      = 0;

    // Firstly, compute W = S * V_H.
    HIPBLAS_CHECK(
        hipblasDdgmm(hipblas_handle, HIPBLAS_SIDE_LEFT, n, n, d_V_H, ldv, d_S, 1, d_W, ldv));

    // Secondly, make A = U * W - A.
    HIP_CHECK(hipMemcpy(d_A, A.data(), sizeof(double) * size_A, hipMemcpyHostToDevice));
    HIPBLAS_CHECK(hipblasDgemm(hipblas_handle,
                               HIPBLAS_OP_N,
                               HIPBLAS_OP_N,
                               m /*rows A*/,
                               n /*cols A*/,
                               n /*cols U*/,
                               &h_one,
                               d_U,
                               ldu,
                               d_W,
                               ldv,
                               &h_minus_one,
                               d_A,
                               lda));
    // Copy the result back to the host.
    HIP_CHECK(hipMemcpy(A.data(), d_A, sizeof(double) * size_A, hipMemcpyDeviceToHost));

    // Lastly, check if A is 0.
    for(int j = 0; j < n; ++j)
    {
        for(int i = 0; i < m; ++i)
        {
            errors += std::fabs(A[i + j * lda]) > eps;
        }
    }

    // Free resources.
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_U));
    HIP_CHECK(hipFree(d_V_H));
    HIP_CHECK(hipFree(d_S));
    HIP_CHECK(hipFree(d_W));
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_rwork));
    HIP_CHECK(hipFree(d_bdsqr_info));
    HIPBLAS_CHECK(hipblasDestroy(hipblas_handle));
    HIPSOLVER_CHECK(hipsolverDestroy(hipsolver_handle));

    // Print validation result.
    return report_validation_result(errors);
}
