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

#include <cstddef>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

int main()
{
    // create a host vector with 4 elements
    thrust::host_vector<int> host_vector(4);
    host_vector[0] = 33;
    host_vector[1] = -1;
    host_vector[2] = 0;
    host_vector[3] = 10;

    // print the elements of the vector
    std::cout << "host_vector after setting elements: "
              << format_range(host_vector.begin(), host_vector.end()) << std::endl;

    // transfer to device and compute sum
    const thrust::device_vector<int> device_vector = host_vector;

    // initial value used in the summation
    constexpr int initial_value = 500;

    // compute sum
    const int sum = thrust::reduce(device_vector.begin(),
                                   device_vector.end(),
                                   initial_value,
                                   thrust::plus<int>{});

    // print the sum
    std::cout << "The sum is: " << sum << std::endl;
}
