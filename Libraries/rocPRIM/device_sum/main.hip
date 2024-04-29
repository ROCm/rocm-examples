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
#include <vector>

#include <hip/hip_runtime.h>
#include <rocprim/device/device_reduce.hpp>

int main()
{
    // Allocate and initialize data on the host
    const std::vector<int> h_in{0, 1, 2, 3, 4, 5, 6, 6, 7, 8};

    std::cout << "Input Data: " << format_range(h_in.begin(), h_in.end()) << std::endl;
    std::cout << "Performing reduction using rocprim::reduce on " << h_in.size() << " elements"
              << std::endl;

    // Allocate the storage on the device that is needed to copy h_in to the device
    const size_t h_in_bytes = sizeof(int) * h_in.size();
    int*         d_in       = nullptr;
    HIP_CHECK(hipMalloc(&d_in, h_in_bytes));

    // Allocate storage for device output
    // The reduce sum produces a single scalar
    int* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));

    // Initialize the device input data
    // Copy h_in from the host to device in d_in
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), h_in_bytes, hipMemcpyHostToDevice));

    // This example demonstrates summation
    const auto sum_operator = rocprim::plus<int>{};

    // rocPRIM device level algorithms require temporary storage on the device
    void*  d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;

    // Since d_temp_storage is set to null, this first call to rocprim::reduce
    // will provide the size in bytes for d_temp_storage.
    // The size in bytes will be stored in temp_storage_bytes
    HIP_CHECK(rocprim::reduce(d_temp_storage,
                              temp_storage_bytes,
                              d_in,
                              d_out,
                              h_in.size(),
                              sum_operator));

    // Allocate device memory for temporary storage
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Run Sum on the device
    HIP_CHECK(rocprim::reduce(d_temp_storage,
                              temp_storage_bytes,
                              d_in,
                              d_out,
                              h_in.size(),
                              sum_operator));
    // Get result back to host
    int result;
    HIP_CHECK(hipMemcpy(&result, d_out, sizeof(int), hipMemcpyDeviceToHost));

    std::cout << "Reduced summation result: " << result << std::endl;

    // Free memory
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_temp_storage));
}
