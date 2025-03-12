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

#include <hipfft/hipfft.h>
#include <hipfft/hipfftXt.h>

#include <hip/hip_runtime_api.h>

#include <complex>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char* argv[])
{
    // 1. Define the various input parameters.
    using data_t = std::complex<double>;

    std::cout << "hipFFT single-node multi-gpu complex-to-complex 3D FFT example\n";

    cli::Parser parser(argc, argv);
    parser.set_optional<std::vector<int>>("l", "length", {8, 8}, "2D FFT size (eg: -l 8 8).");
    parser.set_optional<std::vector<int>>("d", "devices", {0, 1}, "Devices to use (eg: -d 0 1)");
    parser.run_and_exit_if_error();

    const auto length  = parser.get<std::vector<int>>("l");
    auto       devices = parser.get<std::vector<int>>("d");

    int available_devices{};
    HIP_CHECK(hipGetDeviceCount(&available_devices));
    if(devices.size() < 2 || available_devices < 2)
    {
        std::cout << "This example is designed to run on two or more devices!" << std::endl;
        return 0;
    }

    if(length.size() != 2 || length[0] < 1 || length[1] < 1)
    {
        std::cout << "This example is designed to run on a 2D input!" << std::endl;
        return 0;
    }

    // 2. Generate the input data on host.
    std::vector<data_t> input(length[0] * length[1]);

    std::mt19937                           generator{};
    std::uniform_real_distribution<double> distribution{};
    std::generate(input.begin(),
                  input.end(),
                  [&]() { return data_t{distribution(generator), distribution(generator)}; });

    std::cout << "Input:\n" << std::setprecision(3);
    print_nd_data(input, length, 16);

    // 3. Initialize the FFT plan handle.
    hipfftHandle plan{};
    HIPFFT_CHECK(hipfftCreate(&plan));

    // 4. Set up multi GPU execution.
    hipStream_t stream{};
    HIP_CHECK(hipStreamCreate(&stream));
    HIPFFT_CHECK(hipfftSetStream(plan, stream));
    HIPFFT_CHECK(hipfftXtSetGPUs(plan, devices.size(), devices.data()));

    // 5. Make the 2D FFT plan.
    std::vector<size_t> work_size(devices.size());
    HIPFFT_CHECK(
        hipfftMakePlan2d(plan, length[0], length[1], hipfftType::HIPFFT_Z2Z, work_size.data()));

    // 6. Allocate memory on device.
    hipLibXtDesc* input_desc;
    hipLibXtDesc* output_desc;
    HIPFFT_CHECK(hipfftXtMalloc(plan, &input_desc, hipfftXtSubFormat::HIPFFT_XT_FORMAT_INPUT));
    HIPFFT_CHECK(hipfftXtMalloc(plan, &output_desc, hipfftXtSubFormat::HIPFFT_XT_FORMAT_OUTPUT));

    // 7. Copy data from host to device.
    HIPFFT_CHECK(hipfftXtMemcpy(plan,
                                input_desc,
                                input.data(),
                                hipfftXtCopyType::HIPFFT_COPY_HOST_TO_DEVICE));

    // 8. Execute multi GPU FFT from plan.
    HIPFFT_CHECK(hipfftXtExecDescriptor(plan, input_desc, output_desc, HIPFFT_FORWARD));

    // 9. Copy data from device to host.
    std::vector<data_t> output(length[0] * length[1]);
    HIPFFT_CHECK(hipfftXtMemcpy(plan,
                                output.data(),
                                output_desc,
                                hipfftXtCopyType::HIPFFT_COPY_DEVICE_TO_HOST));

    std::cout << "Output:\n" << std::setprecision(3);
    print_nd_data(output, length, 16);

    // 10. Clean up.
    HIPFFT_CHECK(hipfftXtFree(input_desc));
    HIPFFT_CHECK(hipfftXtFree(output_desc));
    HIPFFT_CHECK(hipfftDestroy(plan));
    HIP_CHECK(hipStreamDestroy(stream));

    return 0;
}
