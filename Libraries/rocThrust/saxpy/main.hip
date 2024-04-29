// MIT License
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "example_utils.hpp"

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

// This example illustrates how to implement the SAXPY operation
// (Y[i] = a * X[i] + Y[i]) using rocThrust.

// The saxpy_slow function demonstrates the most straightforward implementation
// using a temporary array and two separate transformations, one multiplication
// and one addition.
void saxpy_slow(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    thrust::device_vector<float> temp(X.size());

    // temp <- A
    thrust::fill(temp.begin(), temp.end(), A);

    // temp <- A * X
    thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<float>());

    // Y <- A * X + Y
    thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<float>());
}

// The saxpy_fast function implements the operation with a single transformation
// and represents "best practice". The saxpy_functor makes use of fast and accurate
// Fused Multiply-Add (FMA) operation (see README.md).
struct saxpy_functor : public thrust::binary_function<float, float, float>
{
    float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__ float operator()(const float x, const float y) const
    {
        return fma(a, x, y);
    }
};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

int main()
{
    // initialize host arrays
    std::array x{4.0F, 3.0F, 2.0F, 1.0F};
    std::array y{1.0F, 2.0F, 3.0F, 4.0F};

    // print initial values
    std::cout << "Calculate linear transformation 2 * x + y in two ways:" << std::endl
              << "x = " << format_range(x.begin(), x.end()) << std::endl
              << "y = " << format_range(y.begin(), y.end()) << std::endl;

    {
        // transfer to device
        thrust::device_vector<float> X(x.begin(), x.end());
        thrust::device_vector<float> Y(y.begin(), y.end());

        // slow method
        saxpy_slow(2.0F, X, Y);

        // print result
        std::cout << "By multiplication and addition  = " << format_range(Y.begin(), Y.end())
                  << std::endl;
    }

    {
        // transfer to device
        thrust::device_vector<float> X(x.begin(), x.end());
        thrust::device_vector<float> Y(y.begin(), y.end());

        // fast method
        saxpy_fast(2.0F, X, Y);

        // print result
        std::cout << "By Fused Multiply-Add operation = " << format_range(Y.begin(), Y.end())
                  << std::endl;
    }
}
