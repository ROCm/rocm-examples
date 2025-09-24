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

#include "client.hpp"
#include "rocprofiler_utils.hpp"

__global__ void kernel_a(int dev_id, volatile int* wait_on, int value, int* no_opt)
{
    printf("[device=%i][begin]  Wait on %i: %i (%i)\n", dev_id, value, *wait_on, *no_opt);
    while(*wait_on != value)
    {
        (*no_opt)++;
    };
    printf("[device=%i][break]  Wait on %i: %i (%i)\n", dev_id, value, *wait_on, *no_opt);
    (*wait_on)--;
    printf("[device=%i][return] Wait on %i: %i (%i)\n", dev_id, value, *wait_on, *no_opt);
}

int main(int, char**)
{
    int n_tot_device = 0;
    HIP_CHECK(hipGetDeviceCount(&n_tot_device));
    if(n_tot_device < 2)
    {
        return 0;
    }

    start();
    volatile int* check_value = nullptr;
    int*          no_opt_0    = nullptr;
    int*          no_opt_1    = nullptr;
    HIP_CHECK(hipMallocManaged(&check_value, sizeof(*check_value)));
    HIP_CHECK(hipMallocManaged(&no_opt_0, sizeof(*no_opt_0)));
    HIP_CHECK(hipMallocManaged(&no_opt_1, sizeof(*no_opt_1)));
    *no_opt_0    = 0;
    *no_opt_1    = 0;
    *check_value = 1;

    // Will hang if per-device serialization is not functional
    HIP_CHECK(hipSetDevice(0));
    hipLaunchKernelGGL(kernel_a, dim3(1), dim3(1), 0, 0, 0, check_value, 0, no_opt_0);

    HIP_CHECK(hipSetDevice(1));
    hipLaunchKernelGGL(kernel_a, dim3(1), dim3(1), 0, 0, 1, check_value, 1, no_opt_1);

    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipDeviceSynchronize());

    std::cerr << "Run complete\n";
}
