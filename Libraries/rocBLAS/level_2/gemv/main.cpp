// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

/// \brief Computes a general matrix-vector product:
/// y := alpha * op(A) * x + beta * y
/// where A is optionally transposed before the multiplication.
/// The result is computed in-place, and stored in `y`.
void gemv_reference(const rocblas_operation    transpose_a,
                    const rocblas_int          rows,
                    const rocblas_int          cols,
                    const rocblas_float        alpha,
                    const rocblas_float* const a,
                    const rocblas_int          lda,
                    const rocblas_float* const x,
                    const rocblas_int          incx,
                    const rocblas_float        beta,
                    rocblas_float* const       y,
                    const rocblas_int          incy)
{
    if(transpose_a != rocblas_operation_none)
    {
        for(rocblas_int i = 0; i < cols; ++i)
        {
            rocblas_float result = beta * y[incy * i];
            for(rocblas_int j = 0; j < rows; ++j)
                result += alpha * a[i * lda + j] * x[incx * j];
            y[incy * i] = result;
        }
    }
    else
    {
        for(rocblas_int i = 0; i < rows; ++i)
        {
            rocblas_float result = beta * y[incy * i];
            for(rocblas_int j = 0; j < cols; ++j)
                result += alpha * a[j * lda + i] * x[incx * j];
            y[incy * i] = result;
        }
    }
}

int main(const int argc, const char** argv)
{
    // Parse user inputs
    cli::Parser parser(argc, argv);
    parser.set_optional<float>("a", "alpha", 1.f, "Alpha scalar");
    parser.set_optional<float>("b", "beta", 1.f, "Beta scalar");
    parser.set_optional<int>("x", "incx", 1, "Increment for x vector");
    parser.set_optional<int>("y", "incy", 1, "Increment for y vector");
    parser.set_optional<int>("n", "n", 5, "Number of matrix columns");
    parser.set_optional<int>("m", "m", 5, "Number of matrix rows");
    parser.run_and_exit_if_error();

    // Increment between consecutive values of the x vector.
    const rocblas_int incx = parser.get<int>("x");
    if(incx <= 0)
    {
        std::cout << "Value of 'x' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    // Increment between consecutive values of the y vector.
    const rocblas_int incy = parser.get<int>("y");
    if(incy <= 0)
    {
        std::cout << "Value of 'y' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    // Number of columns in the matrix.
    const rocblas_int cols = parser.get<int>("n");
    if(cols <= 0)
    {
        std::cout << "Value of 'n' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    // Number of rows in the matrix.
    const rocblas_int rows = parser.get<int>("m");
    if(rows <= 0)
    {
        std::cout << "Value of 'm' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    // Alpha scalar multiplier.
    const rocblas_float h_alpha = parser.get<float>("a");

    // Beta scalar multiplier.
    const rocblas_float h_beta = parser.get<float>("b");

    // The leading dimension (the stride between column starts) of the matrix A.
    // The matrix is packed into memory, so the leading dimension is equal to
    // the actual dimension.
    const size_t lda = rows;
    // The size of the A matrix, in elements.
    const size_t size_a = lda * cols;

    // Specify whether the matrix is transposed before the matrix-vector product is computed.
    const rocblas_operation transpose_a = rocblas_operation_none;

    size_t dim_x = cols;
    size_t dim_y = rows;

    // If the matrix is transposed before the operation, the required
    // dimensions of the x and y vector are swapped.
    if(transpose_a != rocblas_operation_none)
    {
        std::swap(dim_x, dim_y);
    }

    // The size of the x vector, in elements (including stride).
    const size_t size_x = dim_x * incx;

    // The size of the y vector, in elements (including stride).
    const size_t size_y = dim_y * incy;

    // Initialize identity matrix h_a.
    std::vector<float> h_a(size_a);
    for(rocblas_int i = 0; i < cols; ++i)
    {
        for(rocblas_int j = 0; j < rows; ++j)
        {
            h_a[i * lda + j] = i == j;
        }
    }

    // Initialize x vector with the sequence 1, 2, 3, ...
    std::vector<float> h_x(size_x);
    std::iota(h_x.begin(), h_x.end(), 1.0f);

    // Initalize the y vector with the sequence -1, -2, -3, ...
    std::vector<float> h_y(size_y);
    std::iota(h_y.begin(), h_y.end(), 1.0f);
    std::transform(h_y.begin(), h_y.end(), h_y.begin(), std::negate<float>());

    // Compute reference on CPU.
    std::vector<float> h_y_gold(h_y);
    gemv_reference(transpose_a,
                   rows,
                   cols,
                   h_alpha,
                   h_a.data(),
                   lda,
                   h_x.data(),
                   incx,
                   h_beta,
                   h_y_gold.data(),
                   incy);

    // Allocate device memory using hipMalloc.
    rocblas_float* d_a{};
    rocblas_float* d_x{};
    rocblas_float* d_y{};
    HIP_CHECK(hipMalloc(&d_a, size_a * sizeof(rocblas_float)));
    HIP_CHECK(hipMalloc(&d_x, size_x * sizeof(rocblas_float)));
    HIP_CHECK(hipMalloc(&d_y, size_y * sizeof(rocblas_float)));

    // Copy the input from the host to the device.
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), size_a * sizeof(rocblas_float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), size_x * sizeof(rocblas_float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y.data(), size_y * sizeof(rocblas_float), hipMemcpyHostToDevice));

    // Initialize a rocBLAS API handle.
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    // Invoke the GEMV operation on the device.
    ROCBLAS_CHECK(rocblas_sgemv(handle,
                                transpose_a,
                                rows,
                                cols,
                                &h_alpha,
                                d_a,
                                lda,
                                d_x,
                                incx,
                                &h_beta,
                                d_y,
                                incy));

    // Fetch the results from the device to the host. These functions automatically wait until the above
    // computation is finished.
    HIP_CHECK(hipMemcpy(h_y.data(), d_y, size_y * sizeof(rocblas_float), hipMemcpyDeviceToHost));

    // Destroy the rocBLAS handle.
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    // Free device memory as it is no longer required.
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));

    // Check the relative error between output generated by the rocBLAS API and the CPU.
    constexpr float eps    = 10.f * std::numeric_limits<float>::epsilon();
    unsigned int    errors = 0;
    for(size_t i = 0; i < size_y; i++)
    {
        errors += std::fabs(h_y[i] - h_y_gold[i]) > eps;
    }

    if(errors)
    {
        std::cout << "Validation failed. Errors: " << errors << std::endl;
        return error_exit_code;
    }

    std::cout << "Validation passed." << std::endl;
}
