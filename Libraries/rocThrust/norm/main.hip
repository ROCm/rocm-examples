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

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

// An anonymous namespace sets static linkage to its contents.
// This means that the contained function definitions will only be visible
// in the current compilation unit (i.e. cpp source file).
namespace
{

// Generic function object to compute the square of a number
template<typename T>
struct square
{
    __host__ __device__ T operator()(const T& x) const
    {
        return x * x;
    }
};

} // namespace

int main()
{
    // create a host vector
    const std::vector<double> host_vector = {0.5, 1.0, 2.0, 5.0};
    std::cout << "host_vector after initialization: "
              << format_range(host_vector.begin(), host_vector.end()) << std::endl;

    // transfer to device
    const thrust::device_vector<double> device_vector = host_vector;

    // initial value passed to the reduction
    constexpr double initial_value = 0;

    // compute the Euclidean norm
    const double norm = std::sqrt(thrust::transform_reduce(device_vector.begin(),
                                                           device_vector.end(),
                                                           square<double>{},
                                                           initial_value,
                                                           thrust::plus<double>{}));

    // print the Euclidean norm
    std::cout << "The Euclidean norm is: " << norm << std::endl;
}
