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

// [sphinx-start]
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>

__global__ void simpleKernel(double *data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = idx * 2.0;
}

int main()
{
    int numDevices;
    hipGetDeviceCount(&numDevices);

    if (numDevices < 2)
    {
        std::cerr << "This example requires at least two GPUs." << std::endl;
        return EXIT_FAILURE;
    }

    double *deviceData0, *deviceData1;
    std::size_t size = 1024 * sizeof(*deviceData0);

    // Create streams and events for each device
    hipStream_t stream0, stream1;
    hipEvent_t startEvent0, stopEvent0, startEvent1, stopEvent1;

    // Initialize device 0
    hipSetDevice(0);
    hipStreamCreate(&stream0);
    hipEventCreate(&startEvent0);
    hipEventCreate(&stopEvent0);
    hipMalloc(&deviceData0, size);

    // Initialize device 1
    hipSetDevice(1);
    hipStreamCreate(&stream1);
    hipEventCreate(&startEvent1);
    hipEventCreate(&stopEvent1);
    hipMalloc(&deviceData1, size);

    // Record the start event on device 0
    hipSetDevice(0);
    hipEventRecord(startEvent0, stream0);

    // Launch the kernel asynchronously on device 0
    simpleKernel<<<1000, 128, 0, stream0>>>(deviceData0);

    // Record the stop event on device 0
    hipEventRecord(stopEvent0, stream0);

    // Wait for the stop event on device 0 to complete
    hipEventSynchronize(stopEvent0);

    // Record the start event on device 1
    hipSetDevice(1);
    hipEventRecord(startEvent1, stream1);

    // Launch the kernel asynchronously on device 1
    simpleKernel<<<1000, 128, 0, stream1>>>(deviceData1);

    // Record the stop event on device 1
    hipEventRecord(stopEvent1, stream1);

    // Wait for the stop event on device 1 to complete
    hipEventSynchronize(stopEvent1);

    // Calculate elapsed time between the events for both devices
    float milliseconds0 = 0, milliseconds1 = 0;
    hipEventElapsedTime(&milliseconds0, startEvent0, stopEvent0);
    hipEventElapsedTime(&milliseconds1, startEvent1, stopEvent1);

    std::cout << "Elapsed time on GPU 0: " << milliseconds0 << " ms" << std::endl;
    std::cout << "Elapsed time on GPU 1: " << milliseconds1 << " ms" << std::endl;

    // Cleanup for device 0
    hipSetDevice(0);
    hipEventDestroy(startEvent0);
    hipEventDestroy(stopEvent0);
    hipStreamSynchronize(stream0);
    hipStreamDestroy(stream0);
    hipFree(deviceData0);

    // Cleanup for device 1
    hipSetDevice(1);
    hipEventDestroy(startEvent1);
    hipEventDestroy(stopEvent1);
    hipStreamSynchronize(stream1);
    hipStreamDestroy(stream1);
    hipFree(deviceData1);

    return EXIT_SUCCESS;
}
// [sphinx-end]
