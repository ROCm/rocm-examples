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
#include <vector>

#include <hip/hip_runtime.h>
#include <hipcub/device/device_radix_sort.hpp>

int main()
{
    // Allocate and initialize data on the host
    const std::vector<float> h_keys{9.3f, 2.1f, 7.3f, 4.0f, 2.2f, 5.0f, 3.6f, 2.7f, 1.1f, 0.0f};
    const std::vector<int>   h_values{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    assert(h_keys.size() == h_values.size());
    const int num_elements = h_keys.size();

    std::cout << "Input (key, value) pairs: "
              << format_pairs(h_keys.begin(), h_keys.end(), h_values.begin(), h_values.end())
              << std::endl;
    std::cout << "Sorting " << num_elements
              << " elements on the device using hipcub::DeviceRadixSort::SortPairs()" << std::endl;

    // Allocate device arrays
    // DoubleBuffer is a convenient struct that provides two buffers of the same type for
    // algorithms that use separate in- and output buffers.
    // By using DoubleBuffers the user doesn't have to explicitly declare both the input
    // and output arrays. For example here the d_keys implicitly contains
    // the input d_keys.d_buffer[0] buffer and also the output buffer d_keys.d_buffer[1].
    // d_keys.d_buffer[1] is assigned the output by hipCUB functions such as the
    // hipcub::DeviceRadixSort::SortPairs shown in this example.
    // The user still has to allocate and deallocate memory.
    // There is another overloaded hipcub::DeviceRadixSort::SortPairs function that doesn't
    // take DoubleBuffer but separate input and output pointers. Please consult
    // the hipCUB documentation.
    hipcub::DoubleBuffer<float> d_keys;
    hipcub::DoubleBuffer<int>   d_values;

    const size_t d_keys_bytes   = sizeof(float) * num_elements;
    const size_t d_values_bytes = sizeof(int) * num_elements;
    HIP_CHECK(hipMalloc(&d_keys.d_buffers[0], d_keys_bytes));
    HIP_CHECK(hipMalloc(&d_keys.d_buffers[1], d_keys_bytes));

    HIP_CHECK(hipMalloc(&d_values.d_buffers[0], d_values_bytes));
    HIP_CHECK(hipMalloc(&d_values.d_buffers[1], d_values_bytes));

    // Initialize device arrays with values from corresponding host arrays
    HIP_CHECK(hipMemcpy(d_keys.d_buffers[d_keys.selector],
                        h_keys.data(),
                        d_keys_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_values.d_buffers[d_values.selector],
                        h_values.data(),
                        d_values_bytes,
                        hipMemcpyHostToDevice));

    // hipCUB algorithms require temporary storage on the device
    void*  d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;

    // Since d_temp_storage is set to null, this first call to
    // hipcub::DeviceRadixSort::SortPairs will provide the size in bytes for d_temp_storage
    // The size in bytes will be stored in temp_storage_bytes
    HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(d_temp_storage,
                                                 temp_storage_bytes,
                                                 d_keys,
                                                 d_values,
                                                 num_elements));

    // Allocate temporary storage on the device
    // temp_storage_bytes is used to allocate the amount of temporary storage on the device
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Run SortPairs on the device
    // SortPairs will sort based on keys
    HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(d_temp_storage,
                                                 temp_storage_bytes,
                                                 d_keys,
                                                 d_values,
                                                 num_elements));

    // Get results back to host
    std::vector<float> result_keys(num_elements);
    std::vector<int>   result_values(num_elements);

    HIP_CHECK(hipMemcpy(result_keys.data(),
                        d_keys.Current(),
                        sizeof(float) * result_keys.size(),
                        hipMemcpyDeviceToHost));

    HIP_CHECK(hipMemcpy(result_values.data(),
                        d_values.Current(),
                        sizeof(int) * result_values.size(),
                        hipMemcpyDeviceToHost));

    std::cout << "Sorted (key, value) pairs: "
              << format_pairs(result_keys.begin(),
                              result_keys.end(),
                              result_values.begin(),
                              result_values.end())
              << std::endl;

    HIP_CHECK(hipFree(d_keys.d_buffers[0]));
    HIP_CHECK(hipFree(d_keys.d_buffers[1]));
    HIP_CHECK(hipFree(d_values.d_buffers[0]));
    HIP_CHECK(hipFree(d_values.d_buffers[1]));

    HIP_CHECK(hipFree(d_temp_storage));
}
