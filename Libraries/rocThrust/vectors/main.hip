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
#include <thrust/host_vector.h>

int main()
{
    // Allocate a resizable vector in host memory.
    thrust::host_vector<int> host_vector(4);
    std::cout << "host_vector after initialization: "
              << format_range(host_vector.begin(), host_vector.end()) << std::endl;
    // Items can be accessed with `operator[]`.
    host_vector[0] = 45;
    host_vector[1] = 78;
    host_vector[2] = 0;
    host_vector[3] = -77;
    std::cout << "host_vector after setting elements: "
              << format_range(host_vector.begin(), host_vector.end()) << std::endl;

    // Resizing the vector keeps the values.
    host_vector.resize(2);
    std::cout << "host_vector after resize to 2: "
              << format_range(host_vector.begin(), host_vector.end()) << std::endl;

    // Copies the values from the host to the device (GPU).
    // Also, `device_vector` allocates the required amount of device memory.
    thrust::device_vector<int> device_vector = host_vector;
    std::cout << "device_vector after copy from host: "
              << format_range(device_vector.begin(), device_vector.end()) << std::endl;

    // Elements of device_vector can be modified and accessed from both host and device code.
    device_vector[0] = 99;
    device_vector[1] = 88;
    std::cout << "device_vector after setting elements: "
              << format_range(device_vector.begin(), device_vector.end()) << std::endl;

    // The allocated memory is freed in the destructor of both the host and the device vector.
}
