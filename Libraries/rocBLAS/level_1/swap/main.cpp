// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocblas/rocblas.h>

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

/// \brief Checks if the provided status code is \p rocblas_status_success and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define ROCBLAS_CHECK(condition)                                                             \
    {                                                                                        \
        const rocblas_status status = condition;                                             \
        if(status != rocblas_status_success)                                                 \
        {                                                                                    \
            std::cerr << "rocBLAS error encountered: \"" << rocblas_status_to_string(status) \
                      << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;               \
            std::exit(error_exit_code);                                                      \
        }                                                                                    \
    }

int main(const int argc, const char** argv)
{
    // Parse user inputs
    cli::Parser parser(argc, argv);
    parser.set_optional<int>("x", "incx", 1, "Increment for x vector");
    parser.set_optional<int>("y", "incy", 1, "Increment for y vector");
    parser.set_optional<int>("n", "n", 5, "Size of input vectors");
    parser.run_and_exit_if_error();

    // Increment between consecutive values of input vector X.
    const rocblas_int incx = parser.get<int>("x");
    if(incx <= 0)
    {
        std::cout << "Value of 'x' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    // Increment between consecutive values of input vector Y.
    const rocblas_int incy = parser.get<int>("y");
    if(incy <= 0)
    {
        std::cout << "Value of 'y' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    // Number of elements in input vector X and input vector Y.
    const rocblas_int n = parser.get<int>("n");
    if(n <= 0)
    {
        std::cout << "Value of 'n' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    // Allocate memory for both the host input vectors X and Y.
    std::vector<float> h_x(n * incx);
    std::vector<float> h_y(n * incy);

    // Initialize the host X vector with the increasing sequence 0, 1, 2, ...
    // and the host Y vector with 1000, 1001, 1002, ...
    std::iota(h_x.begin(), h_x.end(), 0.f);
    std::iota(h_y.begin(), h_y.end(), 1000.f);

    // Display the input vectors.
    std::cout << "Input Vector X: " << format_range(h_x.begin(), h_x.end()) << "\n";
    std::cout << "Input Vector Y: " << format_range(h_y.begin(), h_y.end()) << std::endl;

    // Compute reference result on host.
    std::vector<float> h_x_gold(h_x);
    std::vector<float> h_y_gold(h_y);
    for(int i = 0; i < n; ++i)
    {
        std::swap(h_x[i * incx], h_y[i * incy]);
    }

    // Allocate device memory using hipMalloc.
    float* d_x{};
    float* d_y{};
    HIP_CHECK(hipMalloc(&d_x, n * incx * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, n * incy * sizeof(float)));

    // Copy the input from the host to the device.
    ROCBLAS_CHECK(rocblas_set_vector(n, sizeof(float), h_x.data(), incx, d_x, incx));
    ROCBLAS_CHECK(rocblas_set_vector(n, sizeof(float), h_y.data(), incy, d_y, incy));

    // Initialize a rocBLAS API handle.
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    // Asynchronously invoke the computation on the device. This function returns before the result is
    // actually calculated, after which we can check if there were any launch errors.
    // The leading 's' in sswap stands for single-precision float.
    ROCBLAS_CHECK(rocblas_sswap(handle, n, d_x, incx, d_y, incy));

    // Fetch the results from the device to the host. These functions automatically wait until the above
    // computation is finished.
    ROCBLAS_CHECK(rocblas_get_vector(n, sizeof(float), d_x, incx, h_x.data(), incx));
    ROCBLAS_CHECK(rocblas_get_vector(n, sizeof(float), d_y, incy, h_y.data(), incy));

    // Destroy the rocBLAS handle.
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    // Free device memory.
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));

    // Display the results.
    std::cout << "Output Vector X: " << format_range(h_x.begin(), h_x.end()) << "\n";
    std::cout << "Output Vector Y: " << format_range(h_y.begin(), h_y.end()) << std::endl;

    // Check the relative error between output generated by the rocBLAS API and the CPU.
    constexpr float eps    = 10.f * std::numeric_limits<float>::epsilon();
    unsigned int    errors = 0;
    for(int i = 0; i < n; i++)
    {
        errors += std::fabs(h_x[i] - h_x_gold[i]) > eps;
        errors += std::fabs(h_y[i] - h_y_gold[i]) > eps;
    }

    if(errors)
    {
        std::cout << "Validation failed. Errors: " << errors << std::endl;
        return error_exit_code;
    }

    std::cout << "Validation passed." << std::endl;
}
