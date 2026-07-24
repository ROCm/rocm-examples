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

/// \brief GPU time in milliseconds for each of the two concurrent kernels.
struct timings
{
    float background_ms;
    float critical_ms;
};

/// \brief Launches the long-running background kernel and the shorter critical
/// kernel on two separate streams so they execute concurrently, and returns the
/// GPU time each kernel takes. The critical kernel's time is the latency of
/// interest: when the two streams share the device's compute units it contends
/// with the background kernel, but when the streams belong to execution contexts
/// backed by disjoint compute units it runs undisturbed.
timings time_critical_with_background(hipStream_t     background_stream,
                                      hipStream_t     critical_stream,
                                      unsigned int*   d_out_background,
                                      unsigned int*   d_out_critical,
                                      const workload& wl)
{
    hipEvent_t bg_start, bg_stop, crit_start, crit_stop;
    HIP_CHECK(hipEventCreate(&bg_start));
    HIP_CHECK(hipEventCreate(&bg_stop));
    HIP_CHECK(hipEventCreate(&crit_start));
    HIP_CHECK(hipEventCreate(&crit_stop));

    // Launch and time the long-running background kernel. The launch returns
    // immediately and the kernel keeps executing on its own stream, occupying
    // the device while the critical kernel below runs concurrently.
    HIP_CHECK(hipEventRecord(bg_start, background_stream));
    busy_kernel<<<dim3(wl.background_grid_size), dim3(wl.block_size), 0, background_stream>>>(
        d_out_background,
        wl.background_iterations);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(bg_stop, background_stream));

    // Time the critical kernel on its stream while the background kernel runs.
    // The elapsed time includes any wait for compute units, which is exactly the
    // latency the partitioned path is meant to remove.
    HIP_CHECK(hipEventRecord(crit_start, critical_stream));
    busy_kernel<<<dim3(wl.critical_grid_size), dim3(wl.block_size), 0, critical_stream>>>(
        d_out_critical,
        wl.critical_iterations);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(crit_stop, critical_stream));

    // Wait for both kernels to finish before reading the timers.
    HIP_CHECK(hipEventSynchronize(crit_stop));
    HIP_CHECK(hipEventSynchronize(bg_stop));

    timings result = {};
    HIP_CHECK(hipEventElapsedTime(&result.background_ms, bg_start, bg_stop));
    HIP_CHECK(hipEventElapsedTime(&result.critical_ms, crit_start, crit_stop));

    HIP_CHECK(hipEventDestroy(bg_start));
    HIP_CHECK(hipEventDestroy(bg_stop));
    HIP_CHECK(hipEventDestroy(crit_start));
    HIP_CHECK(hipEventDestroy(crit_stop));

    return result;
}

#ifdef __HIP_PLATFORM_AMD__
/// \brief Runs one partitioned measurement. Splits the device's CUs into a
/// background group and a critical group of (about) requested_critical_cus CUs,
/// creates an execution context and stream for each, times the critical kernel
/// while the background kernel runs, then tears the contexts down. The actual CU
/// counts (which the split API may adjust for alignment) are written back
/// through out_background_cus and out_critical_cus.
timings time_partitioned_case(int             device_id,
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

    const timings result = time_critical_with_background(background_stream,
                                                         critical_stream,
                                                         d_out_background,
                                                         d_out_critical,
                                                         wl);

    HIP_CHECK(hipStreamDestroy(background_stream));
    HIP_CHECK(hipStreamDestroy(critical_stream));
    HIP_CHECK(hipExecutionCtxDestroy(ctx_background));
    HIP_CHECK(hipExecutionCtxDestroy(ctx_critical));

    return result;
}
#endif

/// \brief Prints one row of the results table.
void print_row(const std::string& configuration,
               const std::string& critical_cus,
               const std::string& background_cus,
               float              critical_ms,
               float              background_ms,
               float              speedup)
{
    std::cout << "  " << std::left << std::setw(20) << configuration << std::setw(14)
              << critical_cus << std::setw(16) << background_cus << std::setw(16)
              << (double_precision(critical_ms, 2) + " ms") << std::setw(16)
              << (double_precision(background_ms, 2) + " ms") << double_precision(speedup, 2) << "x"
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

    // Warm up the GPU so one-time costs (module load, first-launch setup) are not
    // charged to the first measured case. The result is discarded.
    {
        hipStream_t warmup;
        HIP_CHECK(hipStreamCreateWithFlags(&warmup, hipStreamNonBlocking));
        busy_kernel<<<dim3(total_cus), dim3(wl.block_size), 0, warmup>>>(d_out_critical, 1000);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(warmup));
        HIP_CHECK(hipStreamDestroy(warmup));
    }

    // Baseline: both kernels share all of the device's CUs. Two ordinary
    // non-blocking streams let them run concurrently and contend for CUs.
    hipStream_t shared_background, shared_critical;
    HIP_CHECK(hipStreamCreateWithFlags(&shared_background, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&shared_critical, hipStreamNonBlocking));

    const timings baseline = time_critical_with_background(shared_background,
                                                           shared_critical,
                                                           d_out_background,
                                                           d_out_critical,
                                                           wl);

    HIP_CHECK(hipStreamDestroy(shared_background));
    HIP_CHECK(hipStreamDestroy(shared_critical));

    // Make sure the device is fully idle before the first partitioned case
    // creates and destroys execution contexts.
    HIP_CHECK(hipDeviceSynchronize());

    // Header for the results table.
    std::cout << "Kernel runtimes while both kernels run concurrently:\n" << std::endl;
    std::cout << "  " << std::left << std::setw(20) << "Configuration" << std::setw(14)
              << "Critical CUs" << std::setw(16) << "Background CUs" << std::setw(16)
              << "Critical" << std::setw(16) << "Background" << "Speedup" << std::endl;
    print_row("Baseline (shared)",
              "all (" + std::to_string(total_cus) + ")",
              "all (" + std::to_string(total_cus) + ")",
              baseline.critical_ms,
              baseline.background_ms,
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

        unsigned int  background_cus = 0;
        unsigned int  critical_cus   = 0;
        const timings result         = time_partitioned_case(device_id,
                                                     all_cu_resources,
                                                     total_cus,
                                                     requested,
                                                     wl,
                                                     d_out_background,
                                                     d_out_critical,
                                                     background_cus,
                                                     critical_cus);

        const float speedup
            = result.critical_ms > 0.0f ? baseline.critical_ms / result.critical_ms : 0.0f;
        print_row("Partitioned",
                  std::to_string(critical_cus),
                  std::to_string(background_cus),
                  result.critical_ms,
                  result.background_ms,
                  speedup);

        // Return the device to an idle state before the next case tears down and
        // rebuilds execution contexts.
        HIP_CHECK(hipDeviceSynchronize());
    }

    std::cout << "\nThe baseline critical kernel waits behind the background kernel for the shared\n"
                 "CUs. With an execution context it runs on its own CUs and finishes sooner; the\n"
                 "more CUs the partition holds, the lower its latency. The background kernel is\n"
                 "confined to fewer CUs as the critical partition grows, so its runtime rises."
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
