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
#include "rocblas_utils.hpp"

#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

int main(const int argc, const char* argv[])
{
    // Parse user inputs.
    cli::Parser parser(argc, argv);
    parser.set_optional<int>("m", "m", 3, "Number of rows of input matrix A");
    parser.set_optional<int>("n", "n", 3, "Number of columns of input matrix A");
    parser.run_and_exit_if_error();

    // Get input matrix rows (m) and columns (n).
    const rocblas_int m = parser.get<int>("m");
    if(m <= 0)
    {
        std::cout << "Value of 'm' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    const rocblas_int n = parser.get<int>("n");
    if(n <= 0)
    {
        std::cout << "Value of 'n' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    // Initialize leading dimensions of input matrix A and output matrix LU.
    const rocblas_int lda = m;

    // Define input and output matrices' sizes.
    const unsigned int size_A    = lda * n;
    const unsigned int size_Ipiv = std::min(m, n);

    // Initialize input matrix with sequence 1, 2, 3, ... .
    std::vector<double> A(size_A);
    std::iota(A.begin(), A.end(), 1.0);

    // We want to obtain the factorization P * A = L * U. Initialize the right-hand matrices:
    // - LU is an m x n matrix consisting of a lower triangular and upper triangular matrix, the lower triangular diagonal values are the "unit elements".
    // - Ipiv is an min(m,n) vector, the vector of pivot indices. The full matrix P of the factorization can be derived from Ipiv.
    std::vector<double> LU(size_A, 0);
    std::vector<int>    Ipiv(size_Ipiv, 0);

    // Allocate host and device memory for the info variable.
    rocblas_int  info{};
    rocblas_int* d_info{};
    HIP_CHECK(hipMalloc(&d_info, sizeof(rocblas_int)));

    // Allocate device memory for the matrices needed and copy input matrix A from host to device.
    double*      d_A{};
    rocblas_int* d_Ipiv{}; // Array of dimension min(m,n).
    HIP_CHECK(hipMalloc(&d_A, sizeof(double) * size_A));
    HIP_CHECK(hipMalloc(&d_Ipiv, sizeof(rocblas_int) * size_Ipiv));
    HIP_CHECK(hipMemcpy(d_A, A.data(), sizeof(double) * size_A, hipMemcpyHostToDevice));

    // Use the rocBLAS API to create a handle and enable passing scalar parameters from a pointer
    // to host memory.
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    // Compute the permutation matrix (vector Ipiv) and the LU matrix.
    ROCBLAS_CHECK(rocsolver_dgetf2(handle, m, n, d_A, lda, d_Ipiv, d_info));

    // Copy device output data to host.
    HIP_CHECK(hipMemcpy(LU.data(), d_A, sizeof(double) * size_A, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(Ipiv.data(), d_Ipiv, sizeof(int) * size_Ipiv, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&info, d_info, sizeof(int), hipMemcpyDeviceToHost));

    // Print trace message for LU factorization.
    if(info > 0)
    {
        std::cout << "U is singular. U[" << info << "," << info << "] is the first zero pivot."
                  << std::endl;
    }
    else if(info == 0)
    {
        std::cout << "Successful exit." << std::endl;
    }

    // Free resources.
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_Ipiv));
    HIP_CHECK(hipFree(d_info));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
}
