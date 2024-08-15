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
#include "rocfft_utils.hpp"

#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include <rocfft/rocfft.h>

#include <algorithm>
#include <complex>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
    std::cout << "rocfft single-node multi-gpu complex-to-complex 3D FFT example\n";

    // Command-line options:
    cli::Parser parser(argc, argv);
    // Length of transform, first dimension must be greater than number of GPU devices
    parser.set_optional<std::vector<size_t>>("l",
                                             "length",
                                             {8, 8, 8},
                                             "3-D FFT size (eg: --length 8 8 8).");
    parser.set_optional<std::vector<size_t>>(
        "d",
        "devices",
        {0, 1},
        "List of devices to use separated by spaces (eg: --devices 0 1)");
    parser.run_and_exit_if_error();

    const auto length  = parser.get<std::vector<size_t>>("l");
    const auto devices = parser.get<std::vector<size_t>>("d");

    int deviceCount = devices.size();
    if(length.size() != 3 || deviceCount != 2)
    {
        std::cout << "This example is designed to run on two devices with 3-D inputs!" << std::endl;
        return 0;
    }

    const size_t fftSize = length[0] * length[1] * length[2]; // Must evenly divide deviceCount
    int          nDevices;
    HIP_CHECK(hipGetDeviceCount(&nDevices));

    std::cout << "Number of available GPUs: " << nDevices << " \n";
    if(nDevices <= static_cast<int>(*std::max_element(devices.begin(), devices.end())))
    {
        std::cout << "device ID greater than number of available devices" << std::endl;
        return 0;
    }

    ROCFFT_CHECK(rocfft_setup());

    rocfft_plan_description description = nullptr;
    ROCFFT_CHECK(rocfft_plan_description_create(&description));
    // Do not set stride information via the descriptor, they are to be defined during field creation below
    ROCFFT_CHECK(rocfft_plan_description_set_data_layout(description,
                                                         rocfft_array_type_complex_interleaved,
                                                         rocfft_array_type_complex_interleaved,
                                                         nullptr,
                                                         nullptr,
                                                         0,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         0));

    // Define infield geometry
    // First entry of upper dimension is the batch size
    const size_t              batch_size     = 1;
    const std::vector<size_t> inbrick0_lower = {0, 0, 0, 0};
    const std::vector<size_t> inbrick0_upper
        = {length[0] / deviceCount, length[1], length[2], batch_size};
    const std::vector<size_t> inbrick1_lower = {length[0] / deviceCount, 0, 0, 0};
    const std::vector<size_t> inbrick1_upper = {length[0], length[1], length[2], batch_size};

    // Row-major stride for brick data layout in memory
    const size_t        idist        = fftSize; // distance between batches
    std::vector<size_t> brick_stride = {1, length[0] * length[1], length[0], idist};

    rocfft_field infield = nullptr;
    ROCFFT_CHECK(rocfft_field_create(&infield));

    rocfft_brick inbrick0 = nullptr;
    ROCFFT_CHECK(rocfft_brick_create(&inbrick0,
                                     inbrick0_lower.data(),
                                     inbrick0_upper.data(),
                                     brick_stride.data(),
                                     inbrick0_lower.size(),
                                     devices[0])); // Device id
    ROCFFT_CHECK(rocfft_field_add_brick(infield, inbrick0));
    ROCFFT_CHECK(rocfft_brick_destroy(inbrick0));

    rocfft_brick inbrick1 = nullptr;
    ROCFFT_CHECK(rocfft_brick_create(&inbrick1,
                                     inbrick1_lower.data(),
                                     inbrick1_upper.data(),
                                     brick_stride.data(),
                                     inbrick1_lower.size(),
                                     devices[1])); // Device id
    ROCFFT_CHECK(rocfft_field_add_brick(infield, inbrick1));
    ROCFFT_CHECK(rocfft_brick_destroy(inbrick1));

    ROCFFT_CHECK(rocfft_plan_description_add_infield(description, infield));

    // Allocate and initialize GPU input
    std::vector<void*>                      gpu_in(2);
    const size_t                            bufferSize = fftSize / deviceCount;
    constexpr std::complex<double>          input_data = {0.1, 0.1};
    const std::vector<std::complex<double>> input(bufferSize, input_data); // Host test input
    const size_t                            memSize = sizeof(std::complex<double>) * bufferSize;

    HIP_CHECK(hipSetDevice(devices[0]));
    HIP_CHECK(hipMalloc(&gpu_in[0], memSize));
    HIP_CHECK(hipMemcpy(gpu_in[0], input.data(), memSize, hipMemcpyHostToDevice));

    HIP_CHECK(hipSetDevice(devices[1]));
    HIP_CHECK(hipMalloc(&gpu_in[1], memSize));
    HIP_CHECK(hipMemcpy(gpu_in[1], input.data(), memSize, hipMemcpyHostToDevice));

    // Data decomposition for output
    rocfft_field outfield = nullptr;
    ROCFFT_CHECK(rocfft_field_create(&outfield));

    std::vector<void*>        gpu_out(2);
    const std::vector<size_t> outbrick0_lower = {0, 0, 0, 0};
    const std::vector<size_t> outbrick0_upper = {length[0] / deviceCount, length[1], length[2], 1};
    const std::vector<size_t> outbrick1_lower = {length[0] / deviceCount, 0, 0, 0};
    const std::vector<size_t> outbrick1_upper = {length[0], length[1], length[2], 1};

    rocfft_brick outbrick0 = nullptr;
    ROCFFT_CHECK(rocfft_brick_create(&outbrick0,
                                     outbrick0_lower.data(),
                                     outbrick0_upper.data(),
                                     brick_stride.data(),
                                     outbrick0_lower.size(),
                                     devices[0])); // Device id
    ROCFFT_CHECK(rocfft_field_add_brick(outfield, outbrick0));
    ROCFFT_CHECK(rocfft_brick_destroy(outbrick0));

    rocfft_brick outbrick1 = nullptr;
    ROCFFT_CHECK(rocfft_brick_create(&outbrick1,
                                     outbrick1_lower.data(),
                                     outbrick1_upper.data(),
                                     brick_stride.data(),
                                     outbrick1_lower.size(),
                                     devices[1])); // Device id
    ROCFFT_CHECK(rocfft_field_add_brick(outfield, outbrick1));
    ROCFFT_CHECK(rocfft_brick_destroy(outbrick1));

    ROCFFT_CHECK(rocfft_plan_description_add_outfield(description, outfield));

    // Allocate GPU output
    HIP_CHECK(hipSetDevice(devices[0]));
    HIP_CHECK(hipMalloc(&gpu_out[0], memSize));
    HIP_CHECK(hipSetDevice(devices[1]));
    HIP_CHECK(hipMalloc(&gpu_out[1], memSize));

    // Create a multi-gpu plan
    HIP_CHECK(hipSetDevice(devices[0]));
    rocfft_plan gpu_plan = nullptr;
    ROCFFT_CHECK(rocfft_plan_create(&gpu_plan,
                                    rocfft_placement_notinplace, // Placeness for the transform
                                    rocfft_transform_type_complex_forward, // Direction of transform
                                    rocfft_precision_double,
                                    length.size(), // Dimension
                                    length.data(), // Lengths
                                    1, // Number of transforms
                                    description); // Description
    );

    // Get execution information and allocate work buffer

    size_t work_buf_size = 0;
    ROCFFT_CHECK(rocfft_plan_get_work_buffer_size(gpu_plan, &work_buf_size));

    void*                 work_buf = nullptr;
    rocfft_execution_info planinfo = nullptr;
    ROCFFT_CHECK(rocfft_execution_info_create(&planinfo));
    if(work_buf_size)
    {
        HIP_CHECK(hipMalloc(&work_buf, work_buf_size));
        ROCFFT_CHECK(rocfft_execution_info_set_work_buffer(planinfo, work_buf, work_buf_size));
    }

    // Execute plan
    ROCFFT_CHECK(rocfft_execute(gpu_plan, (void**)gpu_in.data(), (void**)gpu_out.data(), planinfo));

    // Get results from device
    std::complex<double> output;
    HIP_CHECK(hipSetDevice(devices[0]));
    HIP_CHECK(hipMemcpy(&output, gpu_out[0], sizeof(std::complex<double>), hipMemcpyDeviceToHost));

    const auto expected = static_cast<std::complex<double>>(fftSize) * input_data;
    std::cout << "Expected result: " << expected << std::endl;
    std::cout << "Actual   result: " << output << std::endl;

    // Destroy plan
    ROCFFT_CHECK(rocfft_execution_info_destroy(planinfo));
    ROCFFT_CHECK(rocfft_plan_description_destroy(description));
    ROCFFT_CHECK(rocfft_plan_destroy(gpu_plan));

    ROCFFT_CHECK(rocfft_cleanup());

    // Free device memory
    HIP_CHECK(hipFree(gpu_in[0]));
    HIP_CHECK(hipFree(gpu_in[1]));
    HIP_CHECK(hipFree(gpu_out[0]));
    HIP_CHECK(hipFree(gpu_out[1]));

    return 0;
}
