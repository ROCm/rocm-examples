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

#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include <rocfft/rocfft.h>

#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
    std::cout << "rocfft double-precision complex-to-complex transform\n" << std::endl;

    // Command-line options:
    cli::Parser parser(argc, argv);
    parser.set_optional<int>("d", "device", 0, "Select a specific device id");
    parser.set_optional<bool>("o", "outofplace", false, "Perform an out-of-place transform");
    parser.set_optional<bool>("i", "inverse", false, "Perform an inverse transform");
    parser.set_optional<std::vector<size_t>>(
        "l",
        "length",
        {4, 4},
        "Lengths of the transform separated by spaces (eg: --length 4 4).");
    parser.run_and_exit_if_error();

    const auto length     = parser.get<std::vector<size_t>>("l");
    const auto device_id  = parser.get<int>("d");
    const auto outofplace = parser.get<bool>("o");
    const auto inverse    = parser.get<bool>("i");

    if(device_id < 0)
    {
        std::cout << "Value of 'device_id' should be greater or equal to 0" << std::endl;
        return error_exit_code;
    }

    // Set up buffer size and stride for the input and output:
    std::vector<size_t> stride = {1};
    for(unsigned int i = 1; i < length.size(); ++i)
    {
        stride.push_back(length[i - 1] * stride[i - 1]);
    }

    const size_t size = length.back() * stride.back();

    // Allocate and initialize the host data
    std::vector<hipDoubleComplex> h_input(size);
    for(size_t i = 0; i < size; i++)
    {
        h_input[i] = {size - i, i};
    }

    std::cout << "input:\n";
    print_nd_data(h_input, length, 1, true);
    std::cout << "\n";

    ROCFFT_CHECK(rocfft_setup());

    // Set the device:
    HIP_CHECK(hipSetDevice(device_id));

    const unsigned int size_bytes = size * sizeof(hipDoubleComplex);

    // Create HIP device object and copy host data to the device
    hipDoubleComplex* d_input = nullptr;
    HIP_CHECK(hipMalloc(&d_input, size_bytes));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), size_bytes, hipMemcpyHostToDevice));

    // Create a description struct to set data layout:
    rocfft_plan_description d_description = nullptr;
    ROCFFT_CHECK(rocfft_plan_description_create(&d_description));
    ROCFFT_CHECK(rocfft_plan_description_set_data_layout(
        d_description,
        rocfft_array_type_complex_interleaved, // Input data format
        rocfft_array_type_complex_interleaved, // Output data format
        nullptr,
        nullptr,
        stride.size(), // Input stride length
        stride.data(), // Input stride data
        0, // Input batch distance
        stride.size(), // Output stride length
        stride.data(), // Output stride data
        0)); // Output batch distance

    // With in-place transforms, only input buffers are provided to the execution API,
    // and the resulting data is written to the same buffer, overwriting the input data.
    const rocfft_result_placement place
        = outofplace ? rocfft_placement_notinplace : rocfft_placement_inplace;

    // Direction of transform
    const rocfft_transform_type direction
        = inverse ? rocfft_transform_type_complex_inverse : rocfft_transform_type_complex_forward;

    // We can also pass "nullptr" instead of a description; rocFFT will use reasonable
    // default parameters.  If the data isn't contiguous, we need to set strides, etc,
    // using the description.

    // Create the plan
    rocfft_plan d_plan = nullptr;
    ROCFFT_CHECK(rocfft_plan_create(&d_plan,
                                    place,
                                    direction,
                                    rocfft_precision_double,
                                    length.size(), // Dimension
                                    length.data(), // Lengths
                                    1, // Number of transforms
                                    d_description)); // Description

    // Get the execution info for the fft plan (in particular, work memory requirements):
    rocfft_execution_info planinfo = nullptr;
    ROCFFT_CHECK(rocfft_execution_info_create(&planinfo));
    size_t workbuffersize = 0;
    ROCFFT_CHECK(rocfft_plan_get_work_buffer_size(d_plan, &workbuffersize));

    // If the transform requires work memory, allocate a work buffer:
    void* wbuffer = nullptr;
    if(workbuffersize > 0)
    {
        HIP_CHECK(hipMalloc(&wbuffer, workbuffersize));
        ROCFFT_CHECK(rocfft_execution_info_set_work_buffer(planinfo, wbuffer, workbuffersize));
    }

    // If the transform is out-of-place, allocate the output buffer as well:
    double2* d_output = outofplace ? nullptr : d_input;
    if(outofplace)
    {
        HIP_CHECK(hipMalloc(&d_output, size_bytes));
    }

    // Execute the GPU transform:
    ROCFFT_CHECK(rocfft_execute(d_plan, // plan
                                (void**)&d_input, // in_buffer
                                (void**)&d_output, // out_buffer
                                planinfo)); // execution_info

    // Get the result from the device and print it to standard output:
    std::cout << "output:\n";
    std::vector<hipDoubleComplex> h_output(size);
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, size_bytes, hipMemcpyDeviceToHost));

    print_nd_data(h_output, length, 1, true);

    // Clean up: free device memory
    HIP_CHECK(hipFree(d_input));

    if(outofplace)
    {
        HIP_CHECK(hipFree(d_output));
    }
    if(wbuffer != nullptr)
    {
        HIP_CHECK(hipFree(wbuffer));
    }

    // Clean up: destroy plans
    ROCFFT_CHECK(rocfft_execution_info_destroy(planinfo));
    ROCFFT_CHECK(rocfft_plan_description_destroy(d_description));
    ROCFFT_CHECK(rocfft_plan_destroy(d_plan));

    ROCFFT_CHECK(rocfft_cleanup());
    return 0;
}
