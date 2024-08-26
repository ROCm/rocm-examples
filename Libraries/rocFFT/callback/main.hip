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
#include "rocfft_utils.hpp"

#include <rocfft/rocfft.h>

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>

#include <iostream>
#include <math.h>
#include <random>
#include <vector>

// example of using load/store callbacks with rocfft

struct load_callback_data
{
    double2* filter;
};

__device__ double2 load_callback(double2* input,
                                 size_t   offset,
                                 void*    callback_data,
                                 void* /*sharedMem*/)
{
    auto data = static_cast<load_callback_data*>(callback_data);

    // multiply each element by filter element
    return hipCmul(input[offset], data->filter[offset]);
}

// Can not give __device__ function to HIP_SYMBOL
__device__ auto load_callback_dev = load_callback;

int main()
{
    constexpr size_t N = 8;

    std::vector<double2> input(N), callback_filter(N);

    // Initialize data and filter
    std::random_device                     rd;
    std::default_random_engine             gen(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for(size_t i = 0; i < N; i++)
    {
        input[i].x           = i;
        input[i].y           = i;
        callback_filter[i].x = distribution(gen);
    }

    // Rocfft gpu compute
    ROCFFT_CHECK(rocfft_setup());

    const size_t Nbytes = N * sizeof(double2);

    // Create HIP device object.
    double2 *data_dev, *callback_filter_dev;

    // Create buffers
    HIP_CHECK(hipMalloc(&data_dev, Nbytes));
    HIP_CHECK(hipMalloc(&callback_filter_dev, Nbytes));

    // Copy data to device
    HIP_CHECK(hipMemcpy(data_dev, input.data(), Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(callback_filter_dev, callback_filter.data(), Nbytes, hipMemcpyHostToDevice));

    // Set up scaling
    rocfft_plan_description description  = nullptr;
    const double            scale_factor = 1.0 / static_cast<double>(N);
    ROCFFT_CHECK(rocfft_plan_description_create(&description));
    ROCFFT_CHECK(rocfft_plan_description_set_scale_factor(description, scale_factor));

    // Create plan
    rocfft_plan plan = nullptr;
    ROCFFT_CHECK(rocfft_plan_create(&plan,
                                    rocfft_placement_inplace,
                                    rocfft_transform_type_complex_forward,
                                    rocfft_precision_double,
                                    1,
                                    &N,
                                    1,
                                    description));

    // Check if the plan requires a work buffer
    size_t work_buf_size = 0;
    ROCFFT_CHECK(rocfft_plan_get_work_buffer_size(plan, &work_buf_size));
    rocfft_execution_info info = nullptr;
    ROCFFT_CHECK(rocfft_execution_info_create(&info));

    void* work_buf = nullptr;
    if(work_buf_size)
    {
        HIP_CHECK(hipMalloc(&work_buf, work_buf_size));
        ROCFFT_CHECK(rocfft_execution_info_set_work_buffer(info, work_buf, work_buf_size));
    }

    // Prepare callback
    load_callback_data callback_data_host;
    callback_data_host.filter = callback_filter_dev;

    void* callback_data_dev;
    HIP_CHECK(hipMalloc(&callback_data_dev, sizeof(load_callback_data)));
    HIP_CHECK(hipMemcpy(callback_data_dev,
                        &callback_data_host,
                        sizeof(load_callback_data),
                        hipMemcpyHostToDevice));

    // Get a properly-typed host pointer to the device function, as
    // rocfft_execution_info_set_load_callback expects void*.
    void* callback_ptr_host = nullptr;
    HIP_CHECK(
        hipMemcpyFromSymbol(&callback_ptr_host, HIP_SYMBOL(load_callback_dev), sizeof(void*)));

    // Set callback
    ROCFFT_CHECK(
        rocfft_execution_info_set_load_callback(info, &callback_ptr_host, &callback_data_dev, 0));

    // Execute plan
    ROCFFT_CHECK(rocfft_execute(plan, (void**)&data_dev, nullptr, info));

    // Clean up work buffer
    if(work_buf_size)
    {
        HIP_CHECK(hipFree(work_buf));
    }

    // Destroy description
    ROCFFT_CHECK(rocfft_plan_description_destroy(description));
    description = nullptr;

    // Destroy info
    ROCFFT_CHECK(rocfft_execution_info_destroy(info));
    info = nullptr;

    // Destroy plan
    ROCFFT_CHECK(rocfft_plan_destroy(plan));
    plan = nullptr;

    // Copy result back to host
    std::vector<double2> output(N);
    HIP_CHECK(hipMemcpy(output.data(), data_dev, Nbytes, hipMemcpyDeviceToHost));

    for(size_t i = 0; i < N; i++)
    {
        std::cout << "element " << i << " input:  (" << input[i].x << "," << input[i].y << ")"
                  << " output: (" << output[i].x << "," << output[i].y << ")" << std::endl;
    }

    HIP_CHECK(hipFree(callback_data_dev));
    HIP_CHECK(hipFree(callback_filter_dev));
    HIP_CHECK(hipFree(data_dev));

    ROCFFT_CHECK(rocfft_cleanup());

    return 0;
}
