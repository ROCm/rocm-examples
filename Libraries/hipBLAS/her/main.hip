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
#include "hipblas_utils.hpp"

#if hipblasVersionMajor >= 3
#define HIPBLAS_CHER2 hipblasCher2
#else
#define HIPBLAS_CHER2 hipblasCher2_v2
#endif
#include <hipblas/hipblas.h>

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

/// \brief Computes the Hermitian rank-2 update (HER2) operation on complex floats:
/// A := A + alpha * x * y ** H + conj(alpha) * y * x ** H.
/// The result is computed in-place, and stored in `a`.
void cher_reference(const int                        n,
                    const float                      alpha,
                    const int                        incx,
                    const std::complex<float>* const x,
                    const int                        incy,
                    const std::complex<float>* const y,
                    hipFloatComplex* const           a,
                    const int                        lda)
{
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            const std::complex<float> r
                = x[j * incx] * std::conj(y[i * incy]) * std::complex<float>(alpha, 0)
                  + y[j * incy] * std::conj(x[i * incx]) * std::complex<float>(alpha, 0);

            a[i * lda + j] = hipCaddf(make_hipFloatComplex(r.real(), r.imag()), a[i * lda + j]);
        }
    }
}

int main(const int argc, const char** argv)
{
    // Parse user inputs.
    cli::Parser parser(argc, argv);
    parser.set_optional<float>("a", "alpha", 1.f, "Alpha scalar");
    parser.set_optional<int>("x", "incx", 1, "Increment for x vector");
    parser.set_optional<int>("y", "incy", 1, "Increment for y vector");
    parser.set_optional<int>("n", "n", 5, "Dimension of input vectors and matrix");
    parser.run_and_exit_if_error();

    // Increment between consecutive values of the input vector x.
    const int incx = parser.get<int>("x");
    if(incx <= 0)
    {
        std::cout << "Value of 'x' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    // Increment between consecutive values of the input vector y.
    const int incy = parser.get<int>("y");
    if(incy <= 0)
    {
        std::cout << "Value of 'y' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    // Number of elements in the input vectors and dimension of input matrix.
    const int n = parser.get<int>("n");
    if(n <= 0)
    {
        std::cout << "Value of 'n' should be greater than 0" << std::endl;
        return error_exit_code;
    }

    // Scalar multiplier.
    const float          alpha   = parser.get<float>("a");
    const hipComplex h_alpha = hipComplex(alpha, 0.0f);

    // The leading dimension (the stride between column starts) of the matrix A.
    // The matrix is packed into memory, so the leading dimension is equal to
    // the actual dimension.
    const size_t lda = n;

    // The size of the matrix A and vectors x and y in elements.
    const size_t size_a = lda * lda;
    const size_t size_x = incx * n;
    const size_t size_y = incy * n;

    // We will only update the upper triangle of the hermitian matrix.
    const hipblasFillMode_t fill = HIPBLAS_FILL_MODE_UPPER;

    // Initialize host vector h_x.
    std::vector<std::complex<float>> h_x(size_x);
    for(size_t i = 0; i < size_x; ++i)
    {
        h_x[i] = std::complex<float>(static_cast<float>(i) * 0.1f,
                                     static_cast<float>(size_x - i) * -0.2f);
    }

    // Initialize host vector h_y.
    std::vector<std::complex<float>> h_y(size_y);
    for(size_t i = 0; i < size_y; ++i)
    {
        h_y[i] = std::complex<float>(static_cast<float>(i) * 0.1f,
                                     static_cast<float>(size_y - i) * 0.2f);
    }

    // Initialize host identity matrix h_a.
    std::vector<hipFloatComplex> h_a(size_a);
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            h_a[i * lda + j] = make_hipFloatComplex(float(i == j), 0);
        }
    }

    // Compute reference on CPU.
    std::vector<hipFloatComplex> h_a_gold(h_a);
    cher_reference(n, alpha, incx, h_x.data(), incy, h_y.data(), h_a_gold.data(), lda);

    // Initialize a hipBLAS API handle.
    hipblasHandle_t handle;
    HIPBLAS_CHECK(hipblasCreate(&handle));
    HIPBLAS_CHECK(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

    // Allocate device memory using hipMalloc.
    hipComplex* d_a{};
    hipComplex* d_x{};
    hipComplex* d_y{};
    HIP_CHECK(hipMalloc(&d_a, size_a * sizeof(hipComplex)));
    HIP_CHECK(hipMalloc(&d_x, size_x * sizeof(hipComplex)));
    HIP_CHECK(hipMalloc(&d_y, size_y * sizeof(hipComplex)));

    // Copy the input vectors from the host to the device. The layout of std::complex<float> is
    // compatible with that of hipFloatComplex.
    HIPBLAS_CHECK(hipblasSetVector(n, sizeof(std::complex<float>), h_x.data(), incx, d_x, incx));
    HIPBLAS_CHECK(hipblasSetVector(n, sizeof(std::complex<float>), h_y.data(), incy, d_y, incy));

    // Copy the input matrix from the host to the device, invoke hipBLAS HER2 function and
    // fetch the results from the device to the host.
    HIPBLAS_CHECK(hipblasSetMatrix(n, n, sizeof(hipFloatComplex), h_a.data(), lda, d_a, lda));
    HIPBLAS_CHECK(HIPBLAS_CHER2(handle, fill, n, &h_alpha, d_x, incx, d_y, incy, d_a, lda));
    HIPBLAS_CHECK(hipblasGetMatrix(n, n, sizeof(hipFloatComplex), d_a, lda, h_a.data(), lda));

    // Destroy the hipBLAS handle.
    HIPBLAS_CHECK(hipblasDestroy(handle));

    // Free device memory as it is no longer required.
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));

    // Check the relative error between output generated by the hipBLAS API and the CPU.
    const float  eps    = 10 * std::numeric_limits<float>::epsilon();
    unsigned int errors = 0;
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            if((fill == HIPBLAS_FILL_MODE_UPPER && j > i)
               || (fill == HIPBLAS_FILL_MODE_LOWER && j < i))
            {
                continue;
            }

            const hipFloatComplex diff = hipCsubf(h_a[i * lda + j], h_a_gold[i * lda + j]);
            errors += diff.x > eps;
            errors += diff.y > eps;
        }
    }
    return report_validation_result(errors);
}
