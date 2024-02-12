// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "hipfft_utils.hpp"

#include <hip/hip_runtime_api.h>
#include <hipfft/hipfft.h>

#include <algorithm>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main()
{
    std::cout << "hipFFT 2D double-precision complex-to-complex transform showing "
                 "the advanced interface\n";

    // Define parameters for the batch of matrices
    constexpr int    dimension = 2;
    std::vector<int> n{4, 5};
    constexpr int    batch_size = 3;

    constexpr int    stride = 1;
    std::vector<int> embed{stride * n[0], stride * n[1]};
    int              distance = embed[0] * embed[1];

    std::vector<std::complex<double>> h_data(batch_size * distance);
    const size_t total_bytes = h_data.size() * sizeof(decltype(h_data)::value_type);

    // Initialize input data on host
    std::mt19937                           generator{};
    std::uniform_real_distribution<double> distribution{};
    std::generate(h_data.begin(),
                  h_data.end(),
                  [&]() {
                      return std::complex<double>{distribution(generator), distribution(generator)};
                  });

    // Print input data per batch
    std::cout << "input:\n" << std::setprecision(3);
    for(int ibatch = 0; ibatch < batch_size; ++ibatch)
    {
        std::cout << "batch: " << ibatch << "\n";
        const auto                              dist = ibatch * distance;
        const std::vector<std::complex<double>> slice(h_data.begin() + dist, h_data.end());
        print_nd_data(slice, embed);
    }
    std::cout << std::endl;

    hipfftHandle plan;

    // Create a hipFFT plan with multiple batches
    HIPFFT_CHECK(hipfftPlanMany(
        &plan,
        dimension,
        n.data(), // Array of size rank, describing the size of each dimension.
        embed
            .data(), // Pointer of size rank that indicates the storage dimensions of the input data in memory.
        stride, // Input stride.
        distance, // Distance between input batches.
        embed
            .data(), // Pointer of size rank that indicates the storage dimensions of the output data in memory.
        stride, // Output stride.
        distance, // Distance between output batches.
        HIPFFT_Z2Z,
        batch_size));

    // Copy input data to device
    hipfftDoubleComplex* d_data;
    HIP_CHECK(hipMalloc((void**)&d_data, total_bytes));
    HIP_CHECK(hipMemcpy(d_data, (void*)h_data.data(), total_bytes, hipMemcpyHostToDevice));

    // Execute hipFFT plan
    HIPFFT_CHECK(hipfftExecZ2Z(plan, d_data, d_data, HIPFFT_FORWARD));

    // Copy output data back to host
    HIP_CHECK(hipMemcpy((void*)h_data.data(), d_data, total_bytes, hipMemcpyDeviceToHost));

    // Print output data in batches
    std::cout << "output:\n" << std::setprecision(3);
    for(int ibatch = 0; ibatch < batch_size; ++ibatch)
    {
        std::cout << "batch: " << ibatch << "\n";
        const auto                              dist = ibatch * distance;
        const std::vector<std::complex<double>> slice(h_data.begin() + dist, h_data.end());
        print_nd_data(slice, embed);
    }
    std::cout << std::endl;

    // Clean up
    HIPFFT_CHECK(hipfftDestroy(plan));
    HIP_CHECK(hipFree(d_data));

    return 0;
}
