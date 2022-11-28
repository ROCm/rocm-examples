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

#include <iostream>
#include <numeric>
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
    parser.set_optional<int>("n", "n", 5, "Size of vector");
    parser.run_and_exit_if_error();

    // Stride between consecutive values of input vector X.
    rocblas_int incx = parser.get<int>("x");

    // Stride between consecutive values of input vector Y.
    rocblas_int incy = parser.get<int>("y");

    // Number of elements in input vector X and input vector Y.
    rocblas_int n = parser.get<int>("n");

    // Check input values validity
    if(incx <= 0)
    {
        std::cout << "Value of 'x' should be greater than 0" << std::endl;
        return 0;
    }

    if(incy <= 0)
    {
        std::cout << "Value of 'y' should be greater than 0" << std::endl;
        return 0;
    }

    if(n <= 0)
    {
        std::cout << "Value of 'n' should be greater than 0" << std::endl;
        return 0;
    }

    // Adjust the size of input vector X for values of stride (incx) not equal to 1.
    size_t size_x = n * incx;

    // Adjust the size of input vector Y for values of stride (incy) not equal to 1.
    size_t size_y = n * incy;

    // Allocate memory for both the host input vectors X and Y
    std::vector<float> h_x(size_x);
    std::vector<float> h_y(size_y);

    // Initialize the values to both the host vectors X and Y to the
    // increasing sequence 0, 1, 2, ...
    std::iota(h_x.begin(), h_x.end(), 0.f);
    std::iota(h_y.begin(), h_y.end(), 0.f);

    std::cout << "Input Vector X: " << format_range(h_x.begin(), h_x.end()) << "\n";
    std::cout << "Input Vector Y: " << format_range(h_y.begin(), h_y.end()) << std::endl;

    // Initialize gold_result as a gold standard to compare our result from rocBLAS dot product funtion
    float gold_result = 0.0;

    //CPU function for dot product
    for(int i = 0; i < n; i++)
    {
        gold_result += h_x[i * incx] * h_y[i * incy];
    }

    // Use the rocBLAS API to create a handle.
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    // Allocate memory for both the both device vectors X and Y.
    float* d_x{};
    float* d_y{};
    HIP_CHECK(hipMalloc(&d_x, size_x * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, size_y * sizeof(float)));

    //Enable passing h_result parameter from pointer to host memory
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    // Transfer data from host vectors to device vectors.
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), sizeof(float) * size_x, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y.data(), sizeof(float) * size_y, hipMemcpyHostToDevice));

    // Initialize h_result for the result of dot product
    float h_result = 0.0;

    // Asynchronous single precision dot product calculation on device
    ROCBLAS_CHECK(rocblas_sdot(handle, n, d_x, incx, d_y, incy, &h_result));

    // Transfer the result from device vector Y to host vector Y,
    // which halts host execution until results ready.
    HIP_CHECK(hipMemcpy(h_y.data(), d_y, sizeof(float) * size_y, hipMemcpyDeviceToHost));

    // Destroy the rocBLAS handle and release device memory.
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(d_x));

    // Print rocBLAS and CPU output.
    std::cout << "Output result:               " << h_result << std::endl;
    std::cout << "Output gold standard result: " << gold_result << std::endl;

    // Check the relative error between output generated by the rocBLAS API and the CPU.
    constexpr float eps = 10.f * std::numeric_limits<float>::epsilon();

    if(std::fabs(h_result - gold_result) > eps)
    {
        std::cout << "Validation failed." << std::endl;
        return error_exit_code;
    }

    std::cout << "Validation passed." << std::endl;
}
