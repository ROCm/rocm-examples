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
#include <vector>

int main(const int argc, char* argv[])
{
    // Parse user inputs
    cli::Parser parser(argc, const_cast<const char**>(argv));
    parser.set_optional("n", "n", 5, "Number of rows and columns of input matrix A");
    parser.run_and_exit_if_error();

    // Get input matrix size.
    const rocblas_int n = parser.get<rocblas_int>("n");
    if(n <= 0)
    {
        std::cerr << "Value of 'n' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    // All input, intermediary, and output matrices are N by N, and have the same leading dimension and size.
    const rocblas_int lda  = n;
    const rocblas_int size = lda * n;

    // Initialize a 'diagonally dominant' matrix A, which is always invertible.
    // See https://en.wikipedia.org/wiki/Diagonally_dominant_matrix
    std::vector<double> A(size);
    // Begin by filling the matrix with the sequence 1, 2, ..., n * n
    std::iota(A.begin(), A.end(), 1.0);
    // Replace the diagonal of the matrix with the sum of its row.
    for(rocblas_int i = 0; i < n; ++i)
    {
        double sum = 0;
        for(rocblas_int j = 0; j < n; ++j)
        {
            sum += std::fabs(A[i + j * lda]);
        }
        A[i + i * lda] = sum;
    }

    // Allocate device memory for the matrices and copy the input to the device.
    // The computation of A^{-1} is in place. In order to compare the result though,
    // we need to keep around a copy of A, so allocate a separate matrix to compute
    // A^{-1} in.
    double*      d_A{};
    double*      d_Ainv{};
    rocblas_int* d_p{};
    HIP_CHECK(hipMalloc(&d_A, sizeof(double) * size));
    HIP_CHECK(hipMalloc(&d_Ainv, sizeof(double) * size));
    HIP_CHECK(hipMalloc(&d_p, sizeof(rocblas_int) * n));
    HIP_CHECK(hipMemcpy(d_A, A.data(), sizeof(double) * size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Ainv, d_A, sizeof(double) * size, hipMemcpyDeviceToDevice));

    // This variable will be used to store information about the result of the getrf operation:
    // If this value is set to zero after the operation, it was successful and the factorization is written
    // to d_A and d_P.
    // If nonzero, U[j, j] is the first non-zero pivot of matrix A, where `j = *d_getrf_info` (1-indexed).
    rocblas_int* d_getrf_info{};
    HIP_CHECK(hipMalloc(&d_getrf_info, sizeof(rocblas_int)));

    // Create a rocBLAS API handle.
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    // Invoke getrf() to compute the LU factorization.
    ROCBLAS_CHECK(rocsolver_dgetrf(handle, n, n, d_Ainv, lda, d_p, d_getrf_info));
    // From this point on, d_Ainv contains the LU factorization of matrix A. Note that both the L and U matrices are
    // stored in d_Ainv: these are lower and upper triangular respectively, and when stored in a square matrix they do
    // not overlap. The diagonal elements of L are not stored.

    // Copy the info value back to the host so that we can read it.
    rocblas_int getrf_info;
    HIP_CHECK(hipMemcpy(&getrf_info, d_getrf_info, sizeof(rocblas_int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_getrf_info));

    if(getrf_info > 0)
    {
        std::cout << "getrf: matrix A is not invertible" << std::endl;
        std::cout << "  first nonzero pivot: " << getrf_info << std::endl;
        std::cout << "  (proceeding with invert anyway)" << std::endl;

        // One can either terminate here, or ignore this status. The getri operation
        // will also report whether the matrix is invertible or not.
    }

    // Continue with computing the actual inversion.

    // This variable will be used to store information about the result of getri:
    // If the value is 0 after the operation, then it was successful and the inverted
    // matrix is written to d_Ainv.
    // If nonzero, then A is not invertible and U[j, j] is the first zero pivot, where `j = *d_getri_info` (1-indexed).
    rocblas_int* d_getri_info{};
    HIP_CHECK(hipMalloc(&d_getri_info, sizeof(rocblas_int)));

    // Compute the inversion.
    ROCBLAS_CHECK(rocsolver_dgetri(handle, n, d_Ainv, lda, d_p, d_getri_info));
    // From this point on, d_Ainv contains the inversion of the original A matrix.

    // Copy the info value back to the host so that we can read it.
    rocblas_int getri_info;
    HIP_CHECK(hipMemcpy(&getri_info, d_getri_info, sizeof(rocblas_int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_getri_info));

    unsigned int errors = 0;

    if(getri_info > 0)
    {
        std::cout << "getri: matrix A is not invertible" << std::endl;
        std::cout << "  first nonzero pivot: " << getrf_info << std::endl;
        errors = 1;
    }
    else
    {
        // Verify the results by checking whether A * A^{-1} = I using rocBLAS.
        // First, construct an identity matrix and upload it to the GPU.
        std::vector<double> C(size);
        generate_identity_matrix(C.data(), n, n, lda);

        double* d_C{};
        HIP_CHECK(hipMalloc(&d_C, sizeof(double) * C.size()));
        HIP_CHECK(hipMemcpy(d_C, C.data(), sizeof(double) * C.size(), hipMemcpyHostToDevice));

        // Perform the check by computing C = 1 * A * A^{-1} + -1 * C.
        // By subtracting the identity matrix, we can check the validity of the inversion by
        // comparing the result to a zero matrix.
        const double h_one       = 1;
        const double h_minus_one = -1;
        ROCBLAS_CHECK(rocblas_dgemm(handle,
                                    rocblas_operation_none,
                                    rocblas_operation_none,
                                    n, // m
                                    n, // n
                                    n, // k
                                    &h_one, // alpha
                                    d_Ainv,
                                    lda,
                                    d_A,
                                    lda,
                                    &h_minus_one, // beta
                                    d_C,
                                    lda));
        // Copy the results to the host.
        HIP_CHECK(hipMemcpy(C.data(), d_C, sizeof(double) * C.size(), hipMemcpyDeviceToHost));

        // Check if the matrix is zero by comparing every element to zero.
        const double eps = n * n * 10.0f * std::numeric_limits<double>::epsilon();
        for(int j = 0; j < n; ++j)
        {
            for(int i = 0; i < n; ++i)
            {
                errors += std::fabs(C[i + j * lda]) > eps;
            }
        }

        HIP_CHECK(hipFree(d_C));
    }

    HIP_CHECK(hipFree(d_p));
    HIP_CHECK(hipFree(d_Ainv));
    HIP_CHECK(hipFree(d_A));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return report_validation_result(errors);
}
