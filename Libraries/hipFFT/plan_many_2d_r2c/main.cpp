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
    std::cout << "hipFFT 2D single-precision real-to-complex transform showing "
                 "the advanced interface\n";

    // Define parameters for the batch of matrices
    constexpr int    dimension = 2;
    std::vector<int> n{4, 5};
    constexpr int    batch_size = 3;

    const int n1_complex_elements      = n[1] / 2 + 1;
    const int n1_padding_real_elements = n1_complex_elements * 2;

    constexpr int    stride = 1;
    std::vector<int> inembed{stride * n[0], stride * n1_padding_real_elements};
    std::vector<int> onembed{stride * n[0], stride * n1_complex_elements};
    int              input_distance  = inembed[0] * inembed[1];
    int              output_distance = onembed[0] * onembed[1];

    std::vector<float> h_input(batch_size * input_distance);
    const auto         total_bytes = h_input.size() * sizeof(decltype(h_input)::value_type);

    // Initialize input data on host
    std::mt19937                          generator{};
    std::uniform_real_distribution<float> distribution{};
    std::generate(h_input.begin(), h_input.end(), [&]() { return distribution(generator); });

    // Print input data per batch
    std::cout << "input:\n" << std::setprecision(3);
    for(int ibatch = 0; ibatch < batch_size; ++ibatch)
    {
        std::cout << "batch: " << ibatch << "\n";
        const auto               dist = ibatch * input_distance;
        const std::vector<float> slice(h_input.begin() + dist, h_input.end());
        // n is matrix without padding
        print_nd_data(slice, n, 1);
    }
    std::cout << std::endl;

    hipfftHandle plan;

    // Create a hipFFT plan with multiple batches
    HIPFFT_CHECK(hipfftPlanMany(
        &plan,
        dimension,
        n.data(), // Array of size rank, describing the size of each dimension.
        inembed
            .data(), // Pointer of size rank that indicates the storage dimensions of the input data in memory.
        stride, // Input stride.
        input_distance, // Distance between input batches.
        onembed
            .data(), // Pointer of size rank that indicates the storage dimensions of the output data in memory.
        stride, // Output stride.
        output_distance, // Distance between output batches.
        HIPFFT_R2C,
        batch_size));

    // Copy input data to device
    hipfftReal* d_data;
    HIP_CHECK(hipMalloc((void**)&d_data, total_bytes));
    HIP_CHECK(hipMemcpy(d_data, (void*)h_input.data(), total_bytes, hipMemcpyHostToDevice));

    // Execute hipFFT plan
    HIPFFT_CHECK(hipfftExecR2C(plan, d_data, (hipfftComplex*)d_data));

    // Allocate and copy output data back to host
    std::vector<std::complex<float>> h_output(total_bytes);
    HIP_CHECK(hipMemcpy((void*)h_output.data(), d_data, total_bytes, hipMemcpyDeviceToHost));

    // Print output data in batches
    std::cout << "output:\n" << std::setprecision(3);
    for(int ibatch = 0; ibatch < batch_size; ++ibatch)
    {
        std::cout << "batch: " << ibatch << "\n";
        const auto                             dist = ibatch * output_distance;
        const std::vector<std::complex<float>> slice(h_output.begin() + dist, h_output.end());
        print_nd_data(slice, onembed);
    }

    // Clean up
    HIPFFT_CHECK(hipfftDestroy(plan));
    HIP_CHECK(hipFree(d_data));

    return 0;
}
