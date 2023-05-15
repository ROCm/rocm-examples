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

inline bool check_dev_info(int* d_dev_info)
{
    bool failed = false;
    int  info{};
    HIP_CHECK(hipMemcpy(&info, d_dev_info, sizeof(int), hipMemcpyDeviceToHost));
    if(info < 0)
    {
        std::cerr << "Parameter #" << -info << " for hipsolverDgeqrf is wrong." << std::endl;
        failed = true;
    }
    return failed;
}

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
        std::cerr << "Value of 'm' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    const int n = parser.get<int>("n");
    if(n <= 0)
    {
        std::cerr << "Value of 'n' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    if(m < n)
    {
        std::cerr << "'m' must be greater than or equal to 'n'" << std::endl;
        return error_exit_code;
    }

    // Initialize host-side constants and variables for matrix A
    const int          lda    = m; // leading dimension of matrix A
    const unsigned int size_a = lda * n; // size of matrix A

    // Initialize input matrix with sequence 1, 2, 3, ... .
    std::vector<double> A(size_a);
    std::iota(A.begin(), A.end(), 1.0);

    // Allocate and initialize device-side parameters
    // Only one matrix has to be allocated on the device-side, as the result is stored in-place.
    // R is stored in the upper right part of d_a.
    // Q is stored in the form of householder vectors in the lower left part.
    // The householder vectors are normalized in a way that the first non-zero element is 1 so that it need not be stored.
    // The scaling factors for those vectors are stored in d_tau.

    // Allocate device memory for the matrix and copy input matrix A from host to device.
    double* d_a{};
    double* d_tau{};
    int*    d_dev_info{}; // status of QR factorization
    HIP_CHECK(hipMalloc(&d_a, sizeof(double) * size_a));
    HIP_CHECK(hipMalloc(&d_tau, sizeof(double) * n))
    HIP_CHECK(hipMalloc(&d_dev_info, sizeof(int)));
    HIP_CHECK(hipMemcpy(d_a, A.data(), sizeof(double) * size_a, hipMemcpyHostToDevice));

    // Create hipSOLVER handle.
    hipsolverHandle_t hipsolver_handle;
    HIPSOLVER_CHECK(hipsolverCreate(&hipsolver_handle));

    // Set up temporary working space variables
    int lwork{}; // size of working array d_work
    int lwork_geqrf{};
    int lwork_orgqr{};
    // Query amount of required working space
    HIPSOLVER_CHECK(hipsolverDgeqrf_bufferSize(hipsolver_handle, m, n, d_a, lda, &lwork_geqrf));
    HIPSOLVER_CHECK(
        hipsolverDorgqr_bufferSize(hipsolver_handle, m, n, n, d_a, lda, d_tau, &lwork_orgqr));

    // Allocate the amount of working space so that it can be used for both hipsolver calls.
    lwork = std::max(lwork_geqrf, lwork_orgqr);
    double* d_work{}; // Temporary working space
    HIP_CHECK(hipMalloc(&d_work, lwork));

    // Compute the QR factorization.
    HIPSOLVER_CHECK(
        hipsolverDgeqrf(hipsolver_handle, m, n, d_a, lda, d_tau, d_work, lwork, d_dev_info));

    // Check success of hipsolverDgeqrf.
    bool failed = check_dev_info(d_dev_info);

    if(!failed)
    {
        // Calculate Q from the householder vectors represented in d_a and d_tau.
        // The resulting matrix is stored in place of d_a.
        HIPSOLVER_CHECK(
            hipsolverDorgqr(hipsolver_handle, m, n, n, d_a, lda, d_tau, d_work, lwork, d_dev_info));

        // Check success of hipsolverDorgqr.
        failed = check_dev_info(d_dev_info);
    }

    if(!failed)
    {
        // Validate the result using hipBLAS, by calculating the residual Q^T*Q - I, to check whether Q is orthogonal.
        // Create a handle and enable passing scalar parameters from a pointer to host memory.
        hipblasHandle_t hipblas_handle;
        HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));
        HIPBLAS_CHECK(hipblasSetPointerMode(hipblas_handle, HIPBLAS_POINTER_MODE_HOST));

        // Create an nxn identity matrix on the host and copy it to the device.
        const unsigned int  size_residual = n * n;
        std::vector<double> residual(size_residual);
        generate_identity_matrix(residual.data(), n, n, n);

        double* d_residual;
        HIP_CHECK(hipMalloc(&d_residual, sizeof(double) * size_residual));
        HIP_CHECK(hipMemcpy(d_residual,
                            residual.data(),
                            sizeof(double) * size_residual,
                            hipMemcpyHostToDevice));

        // Set up scalar factors for multiplication and addition.
        constexpr double h_one       = 1.0;
        constexpr double h_minus_one = -1.0;

        // Calculate Q^T*Q - I.
        HIPBLAS_CHECK(hipblasDgemm(
            hipblas_handle,
            HIPBLAS_OP_T, // transpose first input matrix
            HIPBLAS_OP_N, // don't transpose second input matrix
            n, // number of rows of result
            n, // number of columns of result
            m, // number of columns of Q^T
            &h_one, // scalar to multiply the product of the matrices, address to 1.0 on host
            d_a, // first multiplication input matrix (Q, is transposed by setting HIP_BLAS_T)
            lda, // leading dimension of first input matrix
            d_a, // second multiplication input matrix (Q)
            lda, // leading dimension of second input matrix
            &h_minus_one, // scalar to multiply the matrix to be added to the product
            d_residual, // matrix to be added and output of result
            n)); // leading dimension of result matrix

        // Calculate the RMS of elements in Q^T*Q - I and print it.
        double res_rms;
        HIPBLAS_CHECK(hipblasDnrm2(hipblas_handle, size_residual, d_residual, 1, &res_rms));
        res_rms /= size_residual;
        std::cout << "The RMS of the elements of Q^T*Q - I is " << res_rms << std::endl;

        // Free hipBLAS resources
        HIP_CHECK(hipFree(d_residual));
        HIPBLAS_CHECK(hipblasDestroy(hipblas_handle));
    }

    // Free hipSOLVER resources.
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_tau));
    HIP_CHECK(hipFree(d_dev_info));
    HIP_CHECK(hipFree(d_work));
    HIPSOLVER_CHECK(hipsolverDestroy(hipsolver_handle));

    if(failed)
        return error_exit_code;
}
