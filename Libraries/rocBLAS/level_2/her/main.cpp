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

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

/// \brief Computes the HER matrix vector operation on complex floats:
/// A := A + alpha * x * x ** H
/// The result is computed in-place, and stored in `a`.
void cher_reference(const rocblas_int                n,
                    const float                      alpha,
                    const rocblas_int                incx,
                    const std::complex<float>* const x,
                    hipFloatComplex* const           a,
                    const rocblas_int                lda)
{
    for(rocblas_int i = 0; i < n; ++i)
    {
        for(rocblas_int j = 0; j < n; ++j)
        {
            const std::complex<float> r
                = x[j * incx] * std::conj(x[i * incx]) * std::complex<float>(alpha, 0);
            a[i * lda + j] = hipCaddf(hipFloatComplex(r.real(), r.imag()), a[i * lda + j]);
        }
    }
}

int main(const int argc, const char** argv)
{
    // Parse user inputs
    cli::Parser parser(argc, argv);
    parser.set_optional<float>("a", "alpha", 1.f, "Alpha scalar");
    parser.set_optional<int>("x", "incx", 1, "Increment for x vector");
    parser.set_optional<int>("n", "n", 5, "Size of input vectors");
    parser.run_and_exit_if_error();

    // Increment between consecutive values of the input vector x.
    const rocblas_int incx = parser.get<int>("x");
    if(incx <= 0)
    {
        std::cout << "Value of 'x' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    // Number of elements in the input vector.
    const rocblas_int n = parser.get<int>("n");
    if(n <= 0)
    {
        std::cout << "Value of 'n' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    // Scalar multiplier.
    const rocblas_float h_alpha = parser.get<float>("a");

    // The leading dimension (the stride between column starts) of the matrix A.
    // The matrix is packed into memory, so the leading dimension is equal to
    // the actual dimension.
    const size_t lda = n;
    // The size of the A matrix, in elements.
    const size_t size_a = lda * lda;
    // The size of the A matrix, in byes.
    const size_t size_a_bytes = size_a * sizeof(rocblas_float_complex);

    // The size of the x vector, in elements (including stride).
    const size_t size_x = incx * n;
    // The size of the x vector, in bytes.
    const size_t size_x_bytes = size_x * sizeof(rocblas_float_complex);

    const rocblas_fill fill = rocblas_fill_upper;

    // Initialize identity matrix h_a.
    std::vector<hipFloatComplex> h_a(size_a);
    for(rocblas_int i = 0; i < n; ++i)
    {
        for(rocblas_int j = 0; j < n; ++j)
        {
            h_a[i * lda + j] = hipFloatComplex(i == j, 0);
        }
    }

    // Initialize vector h_x.
    std::vector<std::complex<float>> h_x(size_x);
    for(size_t i = 0; i < size_x; ++i)
    {
        h_x[i] = std::complex<float>(static_cast<float>(i) * 0.1f,
                                     static_cast<float>(size_x - i) * -0.2f);
    }

    // Compute reference on CPU.
    std::vector<hipFloatComplex> h_a_gold(h_a);
    cher_reference(n, h_alpha, incx, h_x.data(), h_a_gold.data(), lda);

    // Allocate device memory using hipMalloc.
    rocblas_float_complex* d_a{};
    rocblas_float_complex* d_x{};
    HIP_CHECK(hipMalloc(&d_a, size_a_bytes));
    HIP_CHECK(hipMalloc(&d_x, size_x_bytes));

    // Copy the input from the host to the device.
    // The layout of std::complex<float> is compatible with that of rocblas_float_complex.
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), size_a_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), size_x_bytes, hipMemcpyHostToDevice));

    // Initialize a rocBLAS API handle.
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    ROCBLAS_CHECK(rocblas_cher(handle, fill, n, &h_alpha, d_x, incx, d_a, lda));

    // Fetch the results from the device to the host. These functions automatically wait until the above
    // computation is finished.
    HIP_CHECK(hipMemcpy(h_a.data(), d_a, size_a_bytes, hipMemcpyDeviceToHost));

    // Destroy the rocBLAS handle.
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    // Free device memory as it is no longer required.
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_x));

    // Check the relative error between output generated by the rocBLAS API and the CPU.
    const float  eps    = 10 * std::numeric_limits<float>::epsilon();
    unsigned int errors = 0;
    for(rocblas_int i = 0; i < n; ++i)
    {
        for(rocblas_int j = 0; j < n; ++j)
        {
            if((fill == rocblas_fill_upper && j > i) || (fill == rocblas_fill_lower && j < i))
            {
                continue;
            }

            const hipFloatComplex diff = h_a[i * lda + j] - h_a_gold[i * lda + j];
            errors += diff.x > eps;
            errors += diff.y > eps;
        }
    }
    if(errors)
    {
        std::cout << "Validation failed. Errors: " << errors << std::endl;
        return error_exit_code;
    }

    std::cout << "Validation passed." << std::endl;
}
