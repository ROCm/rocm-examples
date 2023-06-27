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
#include <string>
#include <vector>

int main(const int argc, char* argv[])
{
    // 1. Parse user input.
    cli::Parser parser(argc, argv);
    parser.set_optional<rocblas_int>("n", "n", 3, "Size of n x n input matrix A");
    parser.set_optional<int>("c", "batch_size", 3, "Number of matrices in the input batch");
    parser.set_optional<int>("p", "padding", 3, "Number of elements used for padding");

    parser.run_and_exit_if_error();

    // Input sanity checks.
    const rocblas_int n          = parser.get<rocblas_int>("n");
    const rocblas_int batch_size = parser.get<rocblas_int>("c");
    const rocblas_int padding    = parser.get<rocblas_int>("p");

    if(n <= 0)
    {
        std::cout << "Value of 'n' should be greater or equal to 1" << std::endl;
        return error_exit_code;
    }
    if(batch_size <= 0)
    {
        std::cout << "Value of 'c' (batch size) should be greater or equal to 1" << std::endl;
        return error_exit_code;
    }
    if(padding <= 0)
    {
        std::cout << "Value of 'p' (padding) should be greater or equal to 1" << std::endl;
        return error_exit_code;
    }

    // Calculate the number of elements needed for the batch.
    const rocblas_int lda                               = n;
    const rocblas_int single_matrix_num_elements        = lda * n;
    const rocblas_int single_matrix_num_elements_padded = single_matrix_num_elements + padding;
    const rocblas_int batch_matrix_num_elements_padded
        = single_matrix_num_elements_padded * batch_size;

    // Calculate the number of elements for the output eigen vectors.
    const rocblas_int single_eigenvector_num_elements        = single_matrix_num_elements;
    const rocblas_int single_eigenvector_num_elements_padded = single_matrix_num_elements_padded;
    const rocblas_int batch_eigenvector_num_elements_padded  = batch_matrix_num_elements_padded;

    // Calculate the number of eigenvalues.
    const rocblas_int single_eigenvalue_num_elements        = n;
    const rocblas_int single_eigenvalue_num_elements_padded = n + padding;
    const rocblas_int batch_eigenvalue_num_elements_padded
        = single_eigenvalue_num_elements_padded * batch_size;

    // Calculate the number of elements needed for the internal tridiagonal matrices.
    const rocblas_int single_tridiagonal_num_elements = n - 1;
    const rocblas_int single_tridiagonal_num_elements_padded
        = single_tridiagonal_num_elements + padding;
    const rocblas_int batch_tridiagonal_num_elements_padded
        = single_tridiagonal_num_elements_padded * batch_size;

    // 2. Data vectors.
    std::vector<rocblas_double> A(batch_matrix_num_elements_padded); // Input matrix.
    std::vector<rocblas_double> V(batch_eigenvector_num_elements_padded); // Resulting eigenvectors.
    std::vector<rocblas_double> W(batch_eigenvalue_num_elements_padded); // Resulting eigenvalues.
    std::vector<rocblas_int>    info(batch_size); // Convergence information.

    // 3. Generate a random symmetric matrix.
    std::default_random_engine                     generator;
    std::uniform_real_distribution<rocblas_double> distribution(0., 2.);
    auto random_number     = [&]() { return distribution(generator); };
    auto initialize_matrix = [&](std::vector<rocblas_double>& A, rocblas_int offset)
    {
        for(int i = 0; i < n; ++i)
        {
            // Populate the diagonal.
            A[offset + (lda + 1) * i] = random_number();
            for(int j = 0; j < i; ++j)
            {
                // Populate the upper and the lower triangular matrix.
                A[offset + i * lda + j] = A[offset + j * lda + i] = random_number();
            }
        }
    };
    for(int i = 0; i < batch_size; ++i)
    {
        initialize_matrix(A, i * single_matrix_num_elements_padded);
    }

    // 4. Set rocSOLVER parameters.
    const rocblas_evect evect = rocblas_evect::rocblas_evect_original;
    const rocblas_fill  uplo  = rocblas_fill::rocblas_fill_lower;

    // 5. Reserve and copy data to device.
    rocblas_double* d_A    = nullptr;
    rocblas_double* d_W    = nullptr;
    rocblas_int*    d_info = nullptr;

    const rocblas_int info_size              = sizeof(rocblas_int) * batch_size;
    const rocblas_int batch_matrix_size      = sizeof(double) * batch_matrix_num_elements_padded;
    const rocblas_int batch_eigenvector_size = batch_matrix_size;
    const rocblas_int batch_eigenvalue_size = sizeof(double) * batch_eigenvalue_num_elements_padded;

    HIP_CHECK(hipMalloc(&d_info, info_size));
    HIP_CHECK(hipMalloc(&d_A, batch_matrix_size));
    HIP_CHECK(hipMalloc(&d_W, batch_eigenvalue_size));

    HIP_CHECK(hipMemcpy(d_A, A.data(), batch_matrix_size, hipMemcpyHostToDevice));

    // 6. Initialize rocBLAS.
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    // 7. Get and reserve the working space on device.
    const rocblas_int batch_tridiagonal_size
        = batch_tridiagonal_num_elements_padded * sizeof(rocblas_double);
    rocblas_double* d_work = nullptr;
    HIP_CHECK(hipMalloc(&d_work, batch_tridiagonal_size));

    // 8. Compute eigenvectors and eigenvalues.
    ROCBLAS_CHECK(rocsolver_dsyev_strided_batched(handle,
                                                  evect,
                                                  uplo,
                                                  n,
                                                  d_A,
                                                  lda,
                                                  single_matrix_num_elements_padded,
                                                  d_W,
                                                  single_eigenvalue_num_elements_padded,
                                                  d_work,
                                                  single_tridiagonal_num_elements_padded,
                                                  d_info,
                                                  batch_size));

    // 9. Get results from device.
    HIP_CHECK(hipMemcpy(V.data(), d_A, batch_eigenvector_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(W.data(), d_W, batch_eigenvalue_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(info.data(), d_info, info_size, hipMemcpyDeviceToHost));

    // 10. Print results.
    auto print_results = [&](rocblas_int batch_id)
    {
        if(info[batch_id] == 0)
        {
            std::cout << "SYEV converges." << std::endl;
        }
        else if(info[batch_id] > 0)
        {
            std::cout << "SYEV does not converge (" << info[batch_id]
                      << " elements did not converge)." << std::endl;
        }

        std::cout << "\nGiven the n x n square input matrix A; we computed the orthonormal "
                     "eigenvectors V and the associated eigenvalues W."
                  << std::endl;
        auto A_begin = A.begin() + single_matrix_num_elements_padded * batch_id;
        auto W_begin = W.begin() + single_eigenvalue_num_elements_padded * batch_id;
        auto V_begin = V.begin() + single_eigenvector_num_elements_padded * batch_id;
        std::cout << "A = " << format_range(A_begin, A_begin + single_matrix_num_elements)
                  << std::endl;
        std::cout << "W = " << format_range(W_begin, W_begin + single_eigenvalue_num_elements)
                  << std::endl;
        std::cout << "V = " << format_range(V_begin, V_begin + single_eigenvector_num_elements)
                  << std::endl;
    };
    // 11. Validate that 'AV == VD' and that 'AV - VD == 0'.
    auto validate_results = [&](rocblas_int batch_id)
    {
        // Calculate offsets for validation.
        const int matrix_offset      = batch_id * single_matrix_num_elements_padded;
        const int eigenvector_offset = batch_id * single_eigenvector_num_elements_padded;
        const int eigenvalue_offset  = batch_id * single_eigenvalue_num_elements_padded;

        std::cout << "\nLet D be the diagonal constructed from W.\n"
                  << "The right multiplication of A * V should result in V * D [AV == VD]:"
                  << std::endl;

        // Right multiplication of the input matrix with the eigenvectors.
        std::vector<double> AV(n * n);
        multiply_matrices(1.0,
                          0.0,
                          n,
                          n,
                          n,
                          A.data() + matrix_offset,
                          lda,
                          1,
                          V.data() + eigenvector_offset,
                          1,
                          lda,
                          AV.data(),
                          lda);
        std::cout << "AV = " << format_range(AV.begin(), AV.end()) << std::endl;

        // Construct the diagonal D from eigenvalues W.
        std::vector<double> D(n * n);
        for(int i = 0; i < n; ++i)
        {
            D[(n + 1) * i] = W[eigenvalue_offset + i];
        }

        // Scale eigenvectors V with W by multiplying V with D.
        std::vector<double> VD(n * n);
        multiply_matrices(1.0,
                          0.0,
                          n,
                          n,
                          n,
                          V.data() + eigenvector_offset,
                          1,
                          lda,
                          D.data(),
                          lda,
                          1,
                          VD.data(),
                          lda);
        std::cout << "VD = " << format_range(VD.begin(), VD.end()) << std::endl;

        constexpr double epsilon = 1.0e5 * std::numeric_limits<double>::epsilon();
        int              errors  = 0;
        double           mse     = 0;
        for(int i = 0; i < n * n; ++i)
        {
            double diff = (AV[i] - VD[i]);
            diff *= diff;
            mse += diff;

            errors += (diff > epsilon);
        }
        mse /= n * n;
        std::cout << "\nMean Square Error of [AV == VD]:\n  " << mse << std::endl;
        return errors;
    };
    int errors = 0;
    for(int i = 0; i < batch_size; ++i)
    {
        std::cout << "===== batch no. " << i << "=====" << std::endl;
        print_results(i);
        const int batch_errors = validate_results(i);
        errors += batch_errors;
        std::cout << "Number of errors: " << batch_errors << std::endl;
        std::cout << "================" << std::endl;
    }

    // 12. Clean up device allocations.
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_W));
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_info));

    return report_validation_result(errors);
}
