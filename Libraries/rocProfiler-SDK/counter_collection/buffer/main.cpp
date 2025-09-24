// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "CmdParser/cmdparser.hpp"
#include "client.hpp"
#include "rocprofiler_utils.hpp"

#include <libgen.h>
#include <thread>
#include <vector>

__global__ void kernel_a(int x, int y)
{
    x = x + y;
}

__global__ void kernel_b(int x, int y)
{
    x = x + y;
}

template<typename T>
__global__ void kernel_c(T* c_d, const T* a_d, size_t n)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i = offset; i < n; i += stride)
    {
        c_d[i] = a_d[i] * a_d[i];
    }
}

void launch_kernels(const long num_launch, const long sync_interval, const int dev_id)
{
    // Normal HIP Calls
    HIP_CHECK(hipSetDevice(dev_id));
    [[maybe_unused]] hipDeviceProp_t dev_prop;
    HIP_CHECK(hipGetDeviceProperties(&dev_prop, dev_id));

    int* gpu_mem = nullptr;
    HIP_CHECK(hipMalloc((void**)&gpu_mem, 1 * sizeof(int)));

    for(long i = 0; i < num_launch; i++)
    {
        // kernel_a and kernel_b to be profiled as part of the session
        hipLaunchKernelGGL(kernel_a, dim3(1), dim3(1), 0, 0, 1, 2);
        hipLaunchKernelGGL(kernel_b, dim3(1), dim3(1), 0, 0, 1, 2);
        if(i % sync_interval == (sync_interval - 1))
        {
            HIP_CHECK(hipDeviceSynchronize());
        }
    }

    const int n_elems = 512 * 512;
    const int n_bytes = n_elems * sizeof(int);
    int *     a_d, *c_d;
    int       a_h[n_elems], c_h[n_elems];

    for(int i = 0; i < n_elems; i++)
    {
        a_h[i] = i;
    }

    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMalloc(&a_d, n_bytes));
    HIP_CHECK(hipMalloc(&c_d, n_bytes));
    HIP_CHECK(hipMemcpy(a_d, a_h, n_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());
    const unsigned blocks            = 512;
    const unsigned threads_per_block = 256;
    for(long i = 0; i < num_launch; i++)
    {
        hipLaunchKernelGGL(kernel_c,
                           dim3(blocks),
                           dim3(threads_per_block),
                           0,
                           0,
                           c_d,
                           a_d,
                           n_elems);
        if(i % sync_interval == (sync_interval - 1))
        {
            HIP_CHECK(hipDeviceSynchronize());
        }
    }
    HIP_CHECK(hipMemcpy(c_h, c_d, n_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(gpu_mem));
    HIP_CHECK(hipFree(a_d));
    HIP_CHECK(hipFree(c_d));
    HIP_CHECK(hipDeviceReset());
}

int main(int argc, char** argv)
{
    auto* exe_name = basename(argv[0]);

    int n_tot_device = 0;
    HIP_CHECK(hipGetDeviceCount(&n_tot_device));

    long n_itr    = 5000;
    long n_sync   = 50;
    long n_device = 0;

    cli::Parser parser(argc, argv);
    parser.set_optional<long>("i", "iterations", 5000, "Number of iterations");
    parser.set_optional<long>("s", "sync", 50, "Sync every N iterations");
    parser.set_optional<long>("d", "devices", 0, "Number of devices to use");
    parser.run_and_exit_if_error();

    n_itr    = parser.get<long>("i");
    n_sync   = parser.get<long>("s");
    n_device = parser.get<long>("d");

    if(n_device > n_tot_device)
    {
        n_device = n_tot_device;
    }
    if(n_device < 1)
    {
        n_device = n_tot_device;
    }

    common::safe_printer::printf("[%s] Number of devices used: %li\n", exe_name, n_device);
    common::safe_printer::printf("[%s] Number of iterations: %li\n", exe_name, n_itr);
    common::safe_printer::printf("[%s] Syncing every %li iterations\n", exe_name, n_sync);
    std::cout << std::flush;

    start();
    for(long dev_id = 0; dev_id < n_device; ++dev_id)
    {
        launch_kernels(n_itr, n_sync, dev_id);
    }

    std::cerr << "Run complete\n" << std::flush;
}
