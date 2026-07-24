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
/// iterations. It is used as a proxy for a compute-bound workload. The volatile
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

/// \brief Launches a long-running background kernel and a shorter critical
/// kernel on two separate streams so they execute concurrently, and returns the
/// GPU time the critical kernel takes to finish while the background kernel is
/// running. When the two streams share the device's compute units the critical
/// kernel contends with the background kernel; when the streams belong to
/// execution contexts backed by disjoint compute units the critical kernel runs
/// undisturbed.
float time_critical_with_background(hipStream_t        background_stream,
                                    hipStream_t        critical_stream,
                                    unsigned int*      d_out_background,
                                    unsigned int*      d_out_critical,
                                    const unsigned int background_grid_size,
                                    const unsigned int critical_grid_size,
                                    const unsigned int block_size,
                                    const unsigned int background_iterations,
                                    const unsigned int critical_iterations)
{
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Launch the long-running background kernel. The launch returns immediately
    // and the kernel keeps executing on its own stream, occupying the device.
    busy_kernel<<<dim3(background_grid_size), dim3(block_size), 0, background_stream>>>(
        d_out_background,
        background_iterations);
    HIP_CHECK(hipGetLastError());

    // Time the critical kernel on its stream while the background kernel runs.
    // The elapsed time includes any wait for compute units, which is exactly the
    // latency the reserved-CU path is meant to remove.
    HIP_CHECK(hipEventRecord(start, critical_stream));
    busy_kernel<<<dim3(critical_grid_size), dim3(block_size), 0, critical_stream>>>(
        d_out_critical,
        critical_iterations);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop, critical_stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

    // Let the background kernel finish before the next measurement.
    HIP_CHECK(hipStreamSynchronize(background_stream));

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
    // platforms this example only runs the shared-CU baseline, so read the
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
    // the long-running background work.
    const unsigned int critical_cus = (total_cus >= 8) ? total_cus / 4 : 2;

    // Workload sizing. The background kernel oversubscribes the whole device so
    // it saturates the CUs; the critical kernel is deliberately small (one block
    // per reserved CU) and does much less work, so on a dedicated partition it
    // finishes quickly.
    constexpr unsigned int block_size            = 256;
    const unsigned int     background_grid_size  = total_cus * 4;
    const unsigned int     critical_grid_size    = critical_cus;
    const unsigned int     background_iterations = 2000000;
    const unsigned int     critical_iterations   = 200000;

    unsigned int* d_out_background = nullptr;
    unsigned int* d_out_critical   = nullptr;
    HIP_CHECK(hipMalloc(&d_out_background, sizeof(unsigned int) * background_grid_size));
    HIP_CHECK(hipMalloc(&d_out_critical, sizeof(unsigned int) * critical_grid_size));

    // Baseline: both kernels share all of the device's CUs. Two ordinary
    // non-blocking streams let them run concurrently and contend for CUs.
    hipStream_t shared_background, shared_critical;
    HIP_CHECK(hipStreamCreateWithFlags(&shared_background, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&shared_critical, hipStreamNonBlocking));

    std::cout << "\nBaseline: critical kernel shares all " << total_cus
              << " CUs with the background kernel..." << std::endl;
    const float shared_ms = time_critical_with_background(shared_background,
                                                          shared_critical,
                                                          d_out_background,
                                                          d_out_critical,
                                                          background_grid_size,
                                                          critical_grid_size,
                                                          block_size,
                                                          background_iterations,
                                                          critical_iterations);

    HIP_CHECK(hipStreamDestroy(shared_background));
    HIP_CHECK(hipStreamDestroy(shared_critical));

#ifdef __HIP_PLATFORM_AMD__
    // Partitioned: split the CUs into two groups. The larger group runs the
    // background kernel; the smaller group is reserved for the critical kernel.
    // The split API can adjust the counts to satisfy architecture alignment
    // requirements.
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

    const unsigned int background_cus = groups[0].sm.smCount;
    const unsigned int reserved_cus   = groups[1].sm.smCount;
    std::cout << "\nBackground execution context provisioned with " << background_cus << " CUs"
              << std::endl;
    std::cout << "Critical execution context provisioned with " << reserved_cus << " CUs"
              << std::endl;

    // Generate a resource descriptor for each group.
    hipDevResourceDesc_t desc_background = {};
    hipDevResourceDesc_t desc_critical   = {};
    HIP_CHECK(hipDevResourceGenerateDesc(&desc_background, &groups[0], 1));
    HIP_CHECK(hipDevResourceGenerateDesc(&desc_critical, &groups[1], 1));

    // Create an execution context from each descriptor.
    hipExecutionCtx_t ctx_background = {};
    hipExecutionCtx_t ctx_critical   = {};
    HIP_CHECK(hipGreenCtxCreate(&ctx_background, desc_background, device_id, 0));
    HIP_CHECK(hipGreenCtxCreate(&ctx_critical, desc_critical, device_id, 0));

    // Create a stream belonging to each execution context. Kernels launched on
    // these streams stay within the context's reserved CUs.
    hipStream_t ctx_background_stream, ctx_critical_stream;
    HIP_CHECK(
        hipExecutionCtxStreamCreate(&ctx_background_stream, ctx_background, hipStreamDefault, 0));
    HIP_CHECK(hipExecutionCtxStreamCreate(&ctx_critical_stream, ctx_critical, hipStreamDefault, 0));

    std::cout << "\nPartitioned: critical kernel runs on its reserved " << reserved_cus
              << " CUs while the background kernel runs on the other " << background_cus << "..."
              << std::endl;
    const float partitioned_ms = time_critical_with_background(ctx_background_stream,
                                                               ctx_critical_stream,
                                                               d_out_background,
                                                               d_out_critical,
                                                               background_grid_size,
                                                               critical_grid_size,
                                                               block_size,
                                                               background_iterations,
                                                               critical_iterations);

    HIP_CHECK(hipStreamDestroy(ctx_background_stream));
    HIP_CHECK(hipStreamDestroy(ctx_critical_stream));
    HIP_CHECK(hipExecutionCtxDestroy(ctx_background));
    HIP_CHECK(hipExecutionCtxDestroy(ctx_critical));
#endif

    HIP_CHECK(hipFree(d_out_background));
    HIP_CHECK(hipFree(d_out_critical));

    // Report the comparison.
    std::cout << "\nCritical kernel latency while the background kernel runs:" << std::endl;
    std::cout << "  Shared CUs (baseline):   " << double_precision(shared_ms, 2) << " ms"
              << std::endl;
#ifdef __HIP_PLATFORM_AMD__
    std::cout << "  Reserved CUs (contexts): " << double_precision(partitioned_ms, 2) << " ms"
              << std::endl;
    if(partitioned_ms > 0.0f)
    {
        std::cout << "  Speedup for the critical kernel: "
                  << double_precision(shared_ms / partitioned_ms, 2) << "x" << std::endl;
    }
#else
    std::cout << "  Reserved CUs (contexts): not available on this platform" << std::endl;
#endif

    std::cout << "\nExecution context example completed successfully." << std::endl;

    return 0;
}
