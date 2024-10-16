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

#include "cmdparser.hpp"
#include "example_utils.hpp"
#include "hipfft_utils.hpp"

#include <hip/driver_types.h>
#include <hipfft/hipfft.h>

#include <algorithm>
#include <complex>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

void fft_example(const int dimension, const int size = 4, const int direction = HIPFFT_FORWARD)
{
    using input_t  = std::complex<double>;
    using output_t = std::complex<double>;

    std::cout << "hipFFT " << dimension << "D double-precision real to complex transform."
              << std::endl;

    // 1. Define input dimensions, ordered as { Nx, Ny, Nz }
    std::vector<int> n(dimension);
    std::fill(n.begin(), n.end(), size);

    // 1c. Calculate size of arrays
    const int n_total = std::accumulate(n.begin(), n.end(), 1, std::multiplies<int>{});

    // 2. Generate input and print
    std::vector<input_t> input(n_total);

    std::mt19937                           generator{};
    std::uniform_real_distribution<double> distribution{};
    std::generate(input.begin(),
                  input.end(),
                  [&]() { return input_t{distribution(generator), distribution(generator)}; });

    std::cout << "Input:\n" << std::setprecision(3);
    print_nd_data(input, n, 16);

    // 3. Alocate device memory
    hipfftDoubleComplex* d_input;
    hipfftDoubleComplex* d_output;

    HIP_CHECK(hipMalloc(&d_input, n_total * sizeof(*d_input)));
    HIP_CHECK(hipMalloc(&d_output, n_total * sizeof(*d_output)));

    // 4. Copy host to device
    HIP_CHECK(hipMemcpy(d_input, input.data(), n_total * sizeof(*d_input), hipMemcpyHostToDevice));

    // 5. Define FFT plan
    hipfftHandle plan;

    // 5a. Create {1, 2, 3}-dimensional plan
    switch(dimension)
    {
        case 1: HIPFFT_CHECK(hipfftPlan1d(&plan, n[0], hipfftType::HIPFFT_Z2Z, 1)); break;
        case 2: HIPFFT_CHECK(hipfftPlan2d(&plan, n[0], n[1], hipfftType::HIPFFT_Z2Z)); break;
        case 3: HIPFFT_CHECK(hipfftPlan3d(&plan, n[0], n[1], n[2], hipfftType::HIPFFT_Z2Z)); break;
    }

    // 6. Execute plan
    HIPFFT_CHECK(hipfftExecZ2Z(plan, d_input, d_output, direction));

    // 7. Allocate output on host
    std::vector<output_t> output(n_total);

    // 8. Copy device to host
    HIP_CHECK(
        hipMemcpy(output.data(), d_output, n_total * sizeof(*d_output), hipMemcpyDeviceToHost));

    // 9. Print output
    std::cout << "Output:\n" << std::setprecision(3);
    print_nd_data(output, n, 16);

    // 10. Clean up
    HIPFFT_CHECK(hipfftDestroy(plan));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

int main(const int argc, const char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<std::vector<int>>("d",
                                          "dimensions",
                                          {1, 2, 3},
                                          "number of dimensions. must be {1, 2, 3}");
    parser.set_optional<int>("n", "size", 4, "size of each dimension");
    parser.run_and_exit_if_error();

    const std::vector<int> dimensions = parser.get<std::vector<int>>("d");
    const int              size       = parser.get<int>("n");

    // Verify passed dimensions
    for(const int dimension : dimensions)
    {
        if(dimension < 1 || dimension > 3)
        {
            std::cout << "Only 1D, 2D, and 3D FFT transformations are supported!" << std::endl;
            return -1;
        }
    }

    for(const int dimension : dimensions)
    {
        fft_example(dimension, size);
    }

    return 0;
}
