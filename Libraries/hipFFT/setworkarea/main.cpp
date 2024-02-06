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
#include <iostream>
#include <numeric>
#include <vector>

int main()
{
    std::cout << "hipFFT 1D single-precision real-to-complex transform showing "
                 "set work area usage\n";

    constexpr size_t N             = 9;
    constexpr size_t N_complex     = (N / 2 + 1);
    constexpr size_t real_bytes    = sizeof(float) * N;
    constexpr size_t complex_bytes = sizeof(std::complex<float>) * N_complex;

    std::cout << "input:\n";
    std::vector<float> h_input(N);
    std::iota(h_input.begin(), h_input.end(), 1);
    for(size_t i = 0; i < N; i++)
    {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;

    // Create HIP device object.
    hipfftReal* d_input;
    HIP_CHECK(hipMalloc(&d_input, real_bytes));

    hipfftComplex* d_output;
    HIP_CHECK(hipMalloc(&d_output, complex_bytes));

    // Copy input data to device
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), real_bytes, hipMemcpyHostToDevice));

    size_t work_size;
    HIPFFT_CHECK(hipfftEstimate1d(N, HIPFFT_R2C, 1, &work_size));
    std::cout << "hipfftEstimate 1d workSize: " << work_size << std::endl;

    hipfftHandle plan{};
    // Allocate a new plan
    HIPFFT_CHECK(hipfftCreate(&plan));
    // Set the plan’s auto-allocation flag to 0.
    HIPFFT_CHECK(hipfftSetAutoAllocation(plan, 0));
    // Assumes that the plan has been created already, and modifies the plan associated with the plan handle.
    HIPFFT_CHECK(hipfftMakePlan1d(plan, N, HIPFFT_R2C, 1, &work_size));

    // Set work buffer
    hipfftComplex* work_buf;
    HIP_CHECK(hipMalloc(&work_buf, work_size));
    HIPFFT_CHECK(hipfftSetWorkArea(plan, work_buf));
    HIPFFT_CHECK(hipfftGetSize(plan, &work_size));

    std::cout << "hipfftGetSize workSize: " << work_size << std::endl;

    // Execute plan
    HIPFFT_CHECK(hipfftExecR2C(plan, d_input, d_output));

    // Copy result back to host
    std::vector<std::complex<float>> h_output(N_complex);
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, complex_bytes, hipMemcpyDeviceToHost));

    std::cout << "output:\n";
    for(size_t i = 0; i < N_complex; i++)
    {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    HIPFFT_CHECK(hipfftDestroy(plan));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(work_buf));

    return 0;
}
