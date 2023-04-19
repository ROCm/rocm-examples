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

#include <cassert>
#include <iostream>
#include <sstream>

// includes for allocating and freeing memory for a device_ptr
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>

// thrust algorithms
#include <thrust/reduce.h>
#include <thrust/sequence.h>

int main()
{
    // Allocate memory buffer to store 10 integers on the device
    constexpr size_t              N     = 10;
    const thrust::device_ptr<int> d_ptr = thrust::device_malloc<int>(N);

    // thrust::device_ptr supports pointer arithmetic
    const thrust::device_ptr<int> first = d_ptr;
    const thrust::device_ptr<int> last  = d_ptr + N;
    std::cout << "Device array contains " << (last - first) << " uninitialized values" << std::endl;

    // thrust::device_ptr can be used when calling Thrust algorithms
    thrust::sequence(first, last);
    std::cout << "Device array after calling thrust::sequence(first, last): "
              << format_range(first, last) << std::endl;

    // Device memory pointed to by thrust::device_ptr can be read and written transparently from host
    d_ptr[0] = 1;
    d_ptr[1] = 11;
    d_ptr[2] = d_ptr[0] + d_ptr[1];
    std::cout << "Device array after setting first three values from host-side: "
              << format_range(first, last) << std::endl;

    // thrust::device_ptr can be converted to a "raw" device pointer for use in other APIs and kernels, etc.
    // Note: the "raw" device pointer should not be dereferenced in host code!
    // To make sure that it's accessible in host code, the memory has to be allocated by using `hipMallocManaged`
    std::cout << "Converting device_ptr to \"raw\" device pointer" << std::endl;
    int* raw_ptr = thrust::raw_pointer_cast(d_ptr);

    // Conversely, raw device pointers can be wrapped in thrust::device_ptr
    std::cout << "Converting \"raw\" device pointer back to device_ptr \"wrapped_ptr\""
              << std::endl;
    thrust::device_ptr<int> wrapped_ptr = thrust::device_pointer_cast(raw_ptr);
    // The wrapped pointer is expected to be equal with the original one
    assert((wrapped_ptr == d_ptr));

    std::cout << "The sum of values in \"wrapped_ptr\" is "
              << thrust::reduce(wrapped_ptr, wrapped_ptr + N) << std::endl;

    std::cout << "Freeing device memory pointed to by device_ptr" << std::endl;
    // device_ptr is not a smart pointer, the memory has to be freed manually
    thrust::device_free(wrapped_ptr);
}
