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

#include <iomanip>
#include <iostream>
#include <string>

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

/// \brief Sizing of the two concurrent workloads. The background kernel is a
/// large, device-saturating job; the critical kernel is a small,
/// latency-sensitive job whose completion time is what the example measures.
struct workload
{
    unsigned int block_size;
    unsigned int background_grid_size;
    unsigned int critical_grid_size;
    unsigned int background_iterations;
    unsigned int critical_iterations;
};

/// \brief Launches the long-running background kernel and the shorter critical
/// kernel on two separate streams so they execute concurrently, and returns the
/// GPU time the critical kernel takes to finish while the background kernel is
/// running. When the two streams share the device's compute units the critical
/// kernel contends with the background kernel; when the streams belong to
/// execution contexts backed by disjoint compute units the critical kernel runs
/// undisturbed.
float time_critical_with_background(hipStream_t     background_stream,
                                    hipStream_t     critical_stream,
                                    unsigned int*   d_out_background,
                                    unsigned int*   d_out_critical,
                                    const workload& wl)
{
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Launch the long-running background kernel. The launch returns immediately
    // and the kernel keeps executing on its own stream, occupying the device.
    busy_kernel<<<dim3(wl.background_grid_size), dim3(wl.block_size), 0, background_stream>>>(
        d_out_background,
        wl.background_iterations);
    HIP_CHECK(hipGetLastError());

    // Time the critical kernel on its stream while the background kernel runs.
    // The elapsed time includes any wait for compute units, which is exactly the
    // latency the partitioned path is meant to remove.
    HIP_CHECK(hipEventRecord(start, critical_stream));
    busy_kernel<<<dim3(wl.critical_grid_size), dim3(wl.block_size), 0, critical_stream>>>(
        d_out_critical,
        wl.critical_iterations);
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

#ifdef __HIP_PLATFORM_AMD__
/// \brief Runs one partitioned measurement. Splits the device's CUs into a
/// background group and a critical group of (about) requested_critical_cus CUs,
/// creates an execution context and stream for each, times the critical kernel
/// while the background kernel runs, then tears the contexts down. The actual CU
/// counts (which the split API may adjust for alignment) are written back
/// through out_background_cus and out_critical_cus.
float time_partitioned_case(int             device_id,
                            hipDevResource& all_cu_resources,
                            unsigned int    total_cus,
                            unsigned int    requested_critical_cus,
                            const workload& wl,
                            unsigned int*   d_out_background,
                            unsigned int*   d_out_critical,
                            unsigned int&   out_background_cus,
                            unsigned int&   out_critical_cus)
{
    // Split the CUs into two groups: the rest for the background kernel and
    // requested_critical_cus for the critical kernel. The split API can adjust
    // the counts to satisfy architecture alignment requirements.
    hipDevResource              groups[2]       = {};
    hipDevSmResourceGroupParams group_params[2] = {
        {/*smCount=*/total_cus - requested_critical_cus, 0, 0, 0},
        {/*smCount=*/requested_critical_cus, 0, 0, 0}};

    HIP_CHECK(hipDevSmResourceSplit(&groups[0],
                                    2,
                                    &all_cu_resources,
                                    nullptr /* remainder */,
                                    0 /* flags */,
                                    &group_params[0]));

    out_background_cus = groups[0].sm.smCount;
    out_critical_cus   = groups[1].sm.smCount;

    // Wrap each group in a descriptor and create an execution context from it.
    hipDevResourceDesc_t desc_background = {};
    hipDevResourceDesc_t desc_critical   = {};
    HIP_CHECK(hipDevResourceGenerateDesc(&desc_background, &groups[0], 1));
    HIP_CHECK(hipDevResourceGenerateDesc(&desc_critical, &groups[1], 1));

    hipExecutionCtx_t ctx_background = {};
    hipExecutionCtx_t ctx_critical   = {};
    HIP_CHECK(hipGreenCtxCreate(&ctx_background, desc_background, device_id, 0));
    HIP_CHECK(hipGreenCtxCreate(&ctx_critical, desc_critical, device_id, 0));

    // A stream created on a context keeps its kernels inside that context's CUs.
    hipStream_t background_stream, critical_stream;
    HIP_CHECK(hipExecutionCtxStreamCreate(&background_stream, ctx_background, hipStreamDefault, 0));
    HIP_CHECK(hipExecutionCtxStreamCreate(&critical_stream, ctx_critical, hipStreamDefault, 0));

    const float elapsed_ms = time_critical_with_background(background_stream,
                                                           critical_stream,
                                                           d_out_background,
                                                           d_out_critical,
                                                           wl);

    HIP_CHECK(hipStreamDestroy(background_stream));
    HIP_CHECK(hipStreamDestroy(critical_stream));
    HIP_CHECK(hipExecutionCtxDestroy(ctx_background));
    HIP_CHECK(hipExecutionCtxDestroy(ctx_critical));

    return elapsed_ms;
}
#endif

/// \brief Prints one row of the results table.
void print_row(const std::string& configuration,
               const std::string& critical_cus,
               const std::string& background_cus,
               float              latency_ms,
               float              speedup)
{
    std::cout << "  " << std::left << std::setw(22) << configuration << std::setw(14)
              << critical_cus << std::setw(16) << background_cus << std::setw(15)
              << (double_precision(latency_ms, 2) + " ms") << double_precision(speedup, 2) << "x"
              << std::endl;
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

    // Workload sizing. The background kernel launches many more blocks than the
    // device can run at once (deep oversubscription) and each block runs for a
    // long time, so it keeps every CU busy for the whole measurement. The
    // critical kernel has a fixed amount of work; how fast it finishes depends on
    // how many CUs it can use and whether it has to share them with the
    // background kernel.
    workload wl;
    wl.block_size            = 256;
    wl.background_grid_size   = total_cus * 64;
    wl.critical_grid_size     = total_cus; // fixed critical work across all cases
    wl.background_iterations  = 2000000;
    wl.critical_iterations    = 500000; // fixed critical work across all cases

    std::cout
        << "\nThis example runs a small, latency-sensitive \"critical\" kernel at the same time\n"
           "as a large \"background\" kernel that saturates the GPU, and measures how long the\n"
           "critical kernel takes to complete. It first runs a baseline where both kernels share\n"
           "all CUs, then sweeps several execution-context partitions that give the critical\n"
           "kernel a growing slice of dedicated CUs. As the critical kernel gets more of its own\n"
           "CUs, its latency drops well below the contended baseline.\n"
        << std::endl;

    // The critical buffer only needs one entry per critical block; the critical
    // grid size is fixed, so this size is reused across every case.
    unsigned int* d_out_background = nullptr;
    unsigned int* d_out_critical   = nullptr;
    HIP_CHECK(hipMalloc(&d_out_background, sizeof(unsigned int) * wl.background_grid_size));
    HIP_CHECK(hipMalloc(&d_out_critical, sizeof(unsigned int) * wl.critical_grid_size));

    // Baseline: both kernels share all of the device's CUs. Two ordinary
    // non-blocking streams let them run concurrently and contend for CUs.
    hipStream_t shared_background, shared_critical;
    HIP_CHECK(hipStreamCreateWithFlags(&shared_background, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&shared_critical, hipStreamNonBlocking));

    const float baseline_ms = time_critical_with_background(shared_background,
                                                            shared_critical,
                                                            d_out_background,
                                                            d_out_critical,
                                                            wl);

    HIP_CHECK(hipStreamDestroy(shared_background));
    HIP_CHECK(hipStreamDestroy(shared_critical));

    // Header for the results table.
    std::cout << "Critical kernel latency while the background kernel runs:\n" << std::endl;
    std::cout << "  " << std::left << std::setw(22) << "Configuration" << std::setw(14)
              << "Critical CUs" << std::setw(16) << "Background CUs" << std::setw(15) << "Latency"
              << "Speedup" << std::endl;
    print_row("Baseline (shared)",
              "all (" + std::to_string(total_cus) + ")",
              "all (" + std::to_string(total_cus) + ")",
              baseline_ms,
              1.0f);

#ifdef __HIP_PLATFORM_AMD__
    // Sweep several partition sizes for the critical kernel: an eighth, a
    // quarter, and half of the device. Each request is clamped so both groups
    // keep at least two CUs, and duplicates (on small devices) are skipped.
    const unsigned int candidates[]  = {total_cus / 8, total_cus / 4, total_cus / 2};
    unsigned int       last_critical = 0;

    for(const unsigned int candidate : candidates)
    {
        unsigned int requested = candidate < 2 ? 2 : candidate;
        if(requested > total_cus - 2)
        {
            requested = total_cus - 2;
        }
        // Skip a request that resolves to the same size as the previous one.
        if(requested == last_critical)
        {
            continue;
        }
        last_critical = requested;

        unsigned int background_cus = 0;
        unsigned int critical_cus   = 0;
        const float  latency_ms     = time_partitioned_case(device_id,
                                                       all_cu_resources,
                                                       total_cus,
                                                       requested,
                                                       wl,
                                                       d_out_background,
                                                       d_out_critical,
                                                       background_cus,
                                                       critical_cus);

        const float speedup = latency_ms > 0.0f ? baseline_ms / latency_ms : 0.0f;
        print_row("Partitioned",
                  std::to_string(critical_cus),
                  std::to_string(background_cus),
                  latency_ms,
                  speedup);
    }

    std::cout << "\nThe baseline critical kernel waits behind the background kernel for the shared\n"
                 "CUs. With an execution context it runs on its own CUs and finishes sooner; the\n"
                 "more CUs the partition holds, the lower its latency."
              << std::endl;
#else
    std::cout << "\nExecution contexts are not available on this platform, so only the shared\n"
                 "baseline was measured."
              << std::endl;
#endif

    HIP_CHECK(hipFree(d_out_background));
    HIP_CHECK(hipFree(d_out_critical));

    std::cout << "\nExecution context example completed successfully." << std::endl;

    return 0;
}
