// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>

/// \brief A kernel that keeps every thread block busy for a fixed number of
/// iterations. It is used as a proxy for a long-running workload. The volatile
/// accumulator prevents the compiler from optimizing the loop away.
__global__ void busy_kernel(unsigned int* out, const unsigned int iterations)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    volatile unsigned int acc = 0;
    for(unsigned int i = 0; i < iterations; ++i)
    {
        acc += i;
    }

    // Write the result so the loop has an observable side effect.
    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = acc;
    }
}

/// \brief Launches the busy kernel on a stream and returns the measured GPU
/// duration in milliseconds.
float time_kernel_on_stream(hipStream_t        stream,
                            unsigned int*      d_out,
                            const unsigned int grid_size,
                            const unsigned int block_size,
                            const unsigned int iterations)
{
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start, stream));
    busy_kernel<<<dim3(grid_size), dim3(block_size), 0, stream>>>(d_out, iterations);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return elapsed_ms;
}

int main()
{
    constexpr int device_id = 0;
    HIP_CHECK(hipSetDevice(device_id));

    // Determine the number of compute units (CUs) on the device. On AMD this
    // count drives how the CUs are partitioned; on other platforms it is only
    // used to size the launch grids.
    unsigned int total_cus = 0;

#ifdef __HIP_PLATFORM_AMD__
    // Query the total number of CUs through the device resource. The device
    // resource uses the field name smCount for CUDA source compatibility; on
    // AMD GPUs it represents compute units.
    hipDevResource all_cu_resources = {};
    HIP_CHECK(hipDeviceGetDevResource(device_id, &all_cu_resources, hipDevResourceTypeSm));
    total_cus = all_cu_resources.sm.smCount;
#else
    // Execution context resource partitioning is an AMD feature. On other
    // platforms this example falls back to two ordinary streams, so read the
    // CU-equivalent count from the device properties instead.
    hipDeviceProp_t props = {};
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));
    total_cus = static_cast<unsigned int>(props.multiProcessorCount);
#endif

    std::cout << "Compute units on device " << device_id << ": " << total_cus << std::endl;

    constexpr unsigned int min_required_cus = 4;
    if(total_cus < min_required_cus)
    {
        std::cout << "Device has too few compute units (" << total_cus << ") for this example. "
                  << "At least " << min_required_cus << " are required." << std::endl;
        return 0;
    }

    // Reserve a quarter of the CUs for the critical work and give the rest to
    // the long-running work.
    const unsigned int critical_cus = (total_cus >= 8) ? total_cus / 4 : 2;

    hipStream_t stream_long, stream_critical;

#ifdef __HIP_PLATFORM_AMD__
    // Split the CUs into two groups. The larger group runs the long-running
    // kernel; the smaller group is reserved for the critical kernel. The split
    // API can adjust the counts to satisfy architecture alignment requirements.
    hipDevResource              groups[2]       = {};
    hipDevSmResourceGroupParams group_params[2] = {
        {/*smCount=*/total_cus - critical_cus, 0, 0, 0},
        {/*smCount=*/critical_cus, 0, 0, 0}};

    HIP_CHECK(hipDevSmResourceSplit(&groups[0],
                                    2,
                                    &all_cu_resources,
                                    nullptr /* remainder */,
                                    0 /* flags */,
                                    &group_params[0]));

    std::cout << "Long-running execution context provisioned with " << groups[0].sm.smCount
              << " CUs" << std::endl;
    std::cout << "Critical execution context provisioned with " << groups[1].sm.smCount << " CUs"
              << std::endl;

    // Generate a resource descriptor for each group.
    hipDevResourceDesc_t desc_long     = {};
    hipDevResourceDesc_t desc_critical = {};
    HIP_CHECK(hipDevResourceGenerateDesc(&desc_long, &groups[0], 1));
    HIP_CHECK(hipDevResourceGenerateDesc(&desc_critical, &groups[1], 1));

    // Create an execution context from each descriptor.
    hipExecutionCtx_t ctx_long     = {};
    hipExecutionCtx_t ctx_critical = {};
    HIP_CHECK(hipGreenCtxCreate(&ctx_long, desc_long, device_id, 0));
    HIP_CHECK(hipGreenCtxCreate(&ctx_critical, desc_critical, device_id, 0));

    // Create a stream belonging to each execution context.
    HIP_CHECK(hipExecutionCtxStreamCreate(&stream_long, ctx_long, hipStreamDefault, 0));
    HIP_CHECK(hipExecutionCtxStreamCreate(&stream_critical, ctx_critical, hipStreamDefault, 0));
#else
    // Fallback path: without execution contexts the kernels share all CUs, so
    // create two independent non-blocking streams instead.
    std::cout << "Execution contexts are not available on this platform; "
              << "running on two ordinary streams without CU partitioning." << std::endl;
    HIP_CHECK(hipStreamCreateWithFlags(&stream_long, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&stream_critical, hipStreamNonBlocking));
#endif

    // Allocate device output for each kernel. Size to the larger grid.
    constexpr unsigned int block_size          = 256;
    const unsigned int     long_grid_size      = total_cus * 4; // oversubscribe the CUs
    const unsigned int     critical_grid_size  = critical_cus;
    const unsigned int     long_iterations     = 2000000;
    const unsigned int     critical_iterations = 200000;

    unsigned int* d_out_long     = nullptr;
    unsigned int* d_out_critical = nullptr;
    HIP_CHECK(hipMalloc(&d_out_long, sizeof(unsigned int) * long_grid_size));
    HIP_CHECK(hipMalloc(&d_out_critical, sizeof(unsigned int) * critical_grid_size));

    std::cout << "\nRunning the long and critical kernels on separate streams..." << std::endl;

    const float long_ms = time_kernel_on_stream(stream_long,
                                                d_out_long,
                                                long_grid_size,
                                                block_size,
                                                long_iterations);

    const float critical_ms = time_kernel_on_stream(stream_critical,
                                                    d_out_critical,
                                                    critical_grid_size,
                                                    block_size,
                                                    critical_iterations);

    std::cout << "Long-running kernel GPU time: " << double_precision(long_ms, 2) << " ms"
              << std::endl;
    std::cout << "Critical kernel GPU time:     " << double_precision(critical_ms, 2) << " ms"
              << std::endl;

    // Clean up.
    HIP_CHECK(hipFree(d_out_long));
    HIP_CHECK(hipFree(d_out_critical));
    HIP_CHECK(hipStreamDestroy(stream_long));
    HIP_CHECK(hipStreamDestroy(stream_critical));

#ifdef __HIP_PLATFORM_AMD__
    HIP_CHECK(hipExecutionCtxDestroy(ctx_long));
    HIP_CHECK(hipExecutionCtxDestroy(ctx_critical));
#endif

    std::cout << "\nExecution context example completed successfully." << std::endl;

    return 0;
}
