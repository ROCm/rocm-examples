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

#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

int main(const int argc, char* argv[])
{
    // Parse user inputs.
    cli::Parser parser(argc, argv);
    parser.set_optional<int>("m", "m", 3, "Number of rows of input matrix A.");
    parser.set_optional<int>("n", "n", 2, "Number of columns of input matrix A.");
    parser.run_and_exit_if_error();

    // Get input matrix rows (m) and columns (n).
    const int m = parser.get<int>("m");
    if(m <= 0)
    {
        std::cout << "Value of 'm' should be greater than 0." << std::endl;
        return error_exit_code;
    }

    const int n = parser.get<int>("n");
    if(n <= 0)
    {
        std::cout << "Value of 'n' should be greater than 0." << std::endl;
        return error_exit_code;
    }

    // Initialize leading dimensions of the input- and output matrices,
    // as well as the number of right-hand sides to solve.
    const int lda  = m;
    const int ldb  = m;
    const int ldx  = n;
    const int nrhs = 1;

    // Define input and output matrices' sizes.
    const unsigned int size_a = lda * n;
    const unsigned int size_b = ldb * nrhs;
    const unsigned int size_x = ldx * nrhs;

    // Allocate input- and output matrices.
    std::vector<double> a(size_a);
    std::vector<double> b(size_b);
    double*             d_a{};
    double*             d_b{};
    double*             d_x{};
    HIP_CHECK(hipMalloc(&d_a, sizeof(double) * size_a));
    HIP_CHECK(hipMalloc(&d_b, sizeof(double) * size_b));
    HIP_CHECK(hipMalloc(&d_x, sizeof(double) * size_x));

    // Initialize matrices A and B as the sequence 1, 2, 3, etc.
    std::iota(a.begin(), a.end(), 1.0);
    HIP_CHECK(hipMemcpy(d_a, a.data(), sizeof(double) * size_a, hipMemcpyHostToDevice));
    std::iota(b.begin(), b.end(), 1.0);
    HIP_CHECK(hipMemcpy(d_b, b.data(), sizeof(double) * size_b, hipMemcpyHostToDevice));

    // Use the hipSOLVER API to create a handle.
    hipsolverHandle_t hipsolver_handle;
    HIPSOLVER_CHECK(hipsolverCreate(&hipsolver_handle));

    // Size of and pointer to working space.
    size_t lwork{};
    void*  d_work{};

    // Query working space size and allocate.
    HIPSOLVER_CHECK(hipsolverDDgels_bufferSize(hipsolver_handle,
                                               m,
                                               n,
                                               nrhs,
                                               d_a,
                                               lda,
                                               d_b,
                                               ldb,
                                               d_x,
                                               ldx,
                                               &lwork));
    HIP_CHECK(hipMalloc(&d_work, lwork));

    // Allocate device function output.
    int  niters{};
    int* d_info{};
    HIP_CHECK(hipMalloc(&d_info, sizeof(int)));

    // Solve the linear least squares problem.
    HIPSOLVER_CHECK(hipsolverDDgels(hipsolver_handle,
                                    m,
                                    n,
                                    nrhs,
                                    d_a,
                                    lda,
                                    d_b,
                                    ldb,
                                    d_x,
                                    ldx,
                                    d_work,
                                    lwork,
                                    &niters,
                                    d_info));

    // Copy device output data to host.
    std::vector<double> x(size_x);
    int                 info{};
    HIP_CHECK(hipMemcpy(x.data(), d_x, sizeof(double) * size_x, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&info, d_info, sizeof(int), hipMemcpyDeviceToHost));

    // Print trace message.
    hipsolver_print_info(info);
    std::cout << "Required " << niters << " iterations.\n";

    // Free all device resources.
    HIP_CHECK(hipFree(d_info));
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_a));
    HIPSOLVER_CHECK(hipsolverDestroy(hipsolver_handle));

    // Validate the result.
    std::vector<double> b_inferred(size_b);
    multiply_matrices(1.0,
                      0.0,
                      m,
                      1,
                      n,
                      a.data(),
                      1,
                      lda,
                      x.data(),
                      1,
                      ldx,
                      b_inferred.data(),
                      ldb);
    double max_error = 0.0;
    for(unsigned int i = 0; i < size_b; i++)
    {
        max_error = std::max(max_error, std::abs(b[i] - b_inferred[i]));
    }

    const double eps = 1.0e5 * std::numeric_limits<double>::epsilon();
    if(max_error > eps)
    {
        std::cout << "Validation failed. Maximum element-wise error " << max_error
                  << " is larger than allowed error " << eps << std::endl;
        return error_exit_code;
    }

    std::cout << "Validation passed." << std::endl;
}
