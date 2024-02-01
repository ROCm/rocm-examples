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
#include <iostream>
#include <numeric>
#include <vector>

int main(int argc, char* argv[])
{
    std::cout << "rocfft double-precision complex-real transform\n" << std::endl;

    // Command-line options:
    cli::Parser parser(argc, argv);
    parser.set_optional<int>("d", "device", 0, "Select a specific device id");
    parser.set_optional<bool>("o", "outofplace", false, "Perform an out-of-place transform");
    parser.set_optional<std::vector<size_t>>(
        "l",
        "length",
        {4, 4},
        "Lengths of the transform separated by spaces (eg: --length 4 4).");
    parser.run_and_exit_if_error();

    const auto length     = parser.get<std::vector<size_t>>("l");
    const auto device_id  = parser.get<int>("d");
    const auto outofplace = parser.get<bool>("o");

    if(device_id < 0)
    {
        std::cout << "Value of 'device_id' should be greater or equal to 0" << std::endl;
        return error_exit_code;
    }

    // The complex data length is half + 1 of the real data length in the contiguous
    // dimensions.  Since rocFFT is column-major, this is the first index.
    std::vector<size_t> cpy_length(length);
    cpy_length[0] = cpy_length[0] / 2 + 1;
    const std::vector<size_t> input_length(cpy_length);
    std::vector<size_t>       input_stride = {1};
    for(unsigned int i = 1; i < input_length.size(); ++i)
    {
        input_stride.push_back(input_length[i - 1] * input_stride[i - 1]);
    }
    const size_t input_size  = input_length.back() * input_stride.back();
    const size_t input_bytes = input_size * sizeof(hipDoubleComplex);

    // Set up the strides and buffer size for the real values:
    std::vector<size_t> output_stride = {1};
    for(unsigned int i = 1; i < length.size(); ++i)
    {
        // In-place transforms need space for two extra real values in the contiguous
        // direction.
        auto val = (length[i - 1] + ((!outofplace && i) ? 2 : 0)) * output_stride[i - 1];
        output_stride.push_back(val);
    }
    const size_t output_size  = length.back() * output_stride.back();
    const size_t output_bytes = output_size * sizeof(double);

    // Allocate and initialize the host data
    std::vector<hipDoubleComplex> h_input(input_size);
    for(size_t i = 0; i < input_size; i++)
    {
        h_input[i] = {i, i};
    }

    // Force the data to be Hermitian symmetric
    if(input_length.size() == 1)
    {
        // For 1D th 0th element and the N/2 element when N is even should be real
        if(length[0] % 2 == 0)
        {
            h_input.back().y = 0;
        }
    }
    else
    {
        std::vector<size_t> index(input_length.size(), 0);
        do
        {
            std::vector<size_t> negindex(index.size(), 0);
            // Skip double work
            if(index[0] <= input_length[0] / 2)
            {
                for(size_t idx = 0; idx < index.size(); ++idx)
                {
                    negindex[idx] = (input_length[idx] - index[idx]) % input_length[idx];
                }

                const auto i
                    = std::inner_product(index.begin(), index.end(), input_stride.begin(), 0);
                const auto j
                    = std::inner_product(negindex.begin(), negindex.end(), input_stride.begin(), 0);
                // Set the opposite value as conjugate
                if(i != j)
                {
                    h_input[j] = hipConj(h_input[i]);
                }
                else
                {
                    // Only real numbers can be conjugate of themselves
                    h_input[i].y = 0;
                }
            }
        }
        while(increment_cm(index, input_length));
    }

    // Print input
    std::cout << "input:\n";
    print_nd_data(h_input, input_length, 1, true);
    std::cout << "\n";

    ROCFFT_CHECK(rocfft_setup());

    // Set the device:
    HIP_CHECK(hipSetDevice(device_id));

    // Create HIP device object and initialize data
    void* d_input = nullptr;
    HIP_CHECK(hipMalloc(&d_input, outofplace ? input_bytes : std::max(output_bytes, input_bytes)));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), input_bytes, hipMemcpyHostToDevice));

    // Create a description struct to set data layout:
    rocfft_plan_description d_description = nullptr;
    ROCFFT_CHECK(rocfft_plan_description_create(&d_description));

    ROCFFT_CHECK(rocfft_plan_description_set_data_layout(
        d_description,
        rocfft_array_type_hermitian_interleaved, // Input data format
        rocfft_array_type_real, // Output data format
        nullptr,
        nullptr,
        input_stride.size(), // Input stride length
        input_stride.data(), // Input stride data
        0, // Input batch distance
        output_stride.size(), // Output stride length
        output_stride.data(), // Output stride data
        0)); // Output batch distance

    // With in-place transforms, only input buffers are provided to the execution API,
    // and the resulting data is written to the same buffer, overwriting the input data.
    const rocfft_result_placement place
        = outofplace ? rocfft_placement_notinplace : rocfft_placement_inplace;

    // We can also pass "nullptr" instead of a description; rocFFT will use reasonable
    // default parameters.  If the data isn't contiguous, we need to set strides, etc,
    // using the description.

    // Create the FFT plan:
    rocfft_plan d_plan = nullptr;
    ROCFFT_CHECK(rocfft_plan_create(&d_plan,
                                    place,
                                    rocfft_transform_type_real_inverse, // Direction
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
    void* d_output = outofplace ? nullptr : d_input;
    if(outofplace)
    {
        HIP_CHECK(hipMalloc(&d_output, output_bytes));
    }

    // Execute the GPU transform:
    ROCFFT_CHECK(rocfft_execute(d_plan, // plan
                                (void**)&d_input, // in_buffer
                                (void**)&d_output, // out_buffer
                                planinfo)); // execution_info

    // Get the result from the device and print it to standard output:
    std::vector<double> h_output(output_size);
    std::cout << "output:\n";
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, output_bytes, hipMemcpyDeviceToHost));
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

    rocfft_cleanup();
    return 0;
}
