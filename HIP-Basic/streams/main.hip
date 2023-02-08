// MIT License
//
// Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

/// \brief Kernel function to perform matrix transpose using shared memory of constant size
template<unsigned int Width>
__global__ void matrix_transpose_static_shared(float* out, const float* in)
{
    // Shared memory of constant size
    __shared__ float shared_mem[Width * Width];

    // Get the two-dimensional global thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    // Perform matrix transpose in shared memroy
    shared_mem[y * Width + x] = in[x * Width + y];

    // Synchronize all threads within a thread block
    __syncthreads();

    // Write the matrix transpose into the global memory
    out[y * Width + x] = shared_mem[y * Width + x];
}

/// \brief Kernel function to perform matrix transpose using dynamic shared memory
__global__ void matrix_transpose_dynamic_shared(float* out, const float* in, const int width)
{
    // Dynamic shared memory
    extern __shared__ float shared_mem[];

    // Get the two-dimensional global thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    // Perform matrix transpose in shared memroy
    shared_mem[y * width + x] = in[x * width + y];

    // Synchronize all threads within a thread block
    __syncthreads();

    // Write the matrix transpose into the global memroy
    out[y * width + x] = shared_mem[y * width + x];
}

template<unsigned int Width, unsigned int Size>
void deploy_multiple_stream(const float*         h_in,
                            std::vector<float*>& h_transpose_matrix,
                            const int            num_streams)
{
    // Set the block dimensions
    constexpr unsigned int threads_per_block_x = 4;
    constexpr unsigned int threads_per_block_y = 4;

    // Create streams
    std::vector<hipStream_t> streams(num_streams);
    for(int i = 0; i < num_streams; i++)
    {
        HIP_CHECK(hipStreamCreate(&streams[i]));
    }

    // Allocate device input and output memory and copy host input data to device memory
    std::vector<float*> d_in(num_streams);
    std::vector<float*> d_transpose_matrix(num_streams);

    // Size in bytes for memory management
    const size_t size_in_bytes = sizeof(float) * Size;

    // Allocate device input memory
    HIP_CHECK(hipMalloc(&d_in[0], size_in_bytes));
    HIP_CHECK(hipMalloc(&d_in[1], size_in_bytes));

    // Allocate device output memory
    HIP_CHECK(hipMalloc(&d_transpose_matrix[0], size_in_bytes));
    HIP_CHECK(hipMalloc(&d_transpose_matrix[1], size_in_bytes));

    for(int i = 0; i < num_streams; i++)
    {
        // hipMemcpyAsync is used without needing to sync before the kernel launch
        // Because both the hipMemcpyAsync and the kernel launch reside in the same stream.
        // The kernel will be executed only after hipMemcpyAsync finishes. There is implicit synchronization.
        // Note: If the host memory is not pinned at allocation time using hipHostMalloc then hipMemcpyAsync
        // will behave as synchronous.
        HIP_CHECK(hipMemcpyAsync(d_in[i], h_in, size_in_bytes, hipMemcpyHostToDevice, streams[i]));
    }

    // Make sure that Width is evenly divisible by threads_per_block_x and threads_per_block_y
    static_assert(Width % threads_per_block_x == 0);
    static_assert(Width % threads_per_block_y == 0);

    // Launch kernel with stream[0]
    matrix_transpose_static_shared<Width>
        <<<dim3(Width / threads_per_block_x, Width / threads_per_block_y),
           dim3(threads_per_block_x, threads_per_block_y),
           0,
           streams[0]>>>(d_transpose_matrix[0], d_in[0]);

    // Launch kernel with stream[1]
    matrix_transpose_dynamic_shared<<<dim3(Width / threads_per_block_x,
                                           Width / threads_per_block_y),
                                      dim3(threads_per_block_x, threads_per_block_y),
                                      sizeof(float) * Width * Width,
                                      streams[1]>>>(d_transpose_matrix[1], d_in[1], Width);

    // Asynchronously copy the results from device to host
    for(int i = 0; i < num_streams; i++)
    {
        HIP_CHECK(hipMemcpyAsync(h_transpose_matrix[i],
                                 d_transpose_matrix[i],
                                 size_in_bytes,
                                 hipMemcpyDeviceToHost,
                                 streams[i]));
    }
    // Wait for all tasks in both the streams to complete on the device
    HIP_CHECK(hipDeviceSynchronize());

    // Destroy the streams
    for(int i = 0; i < num_streams; i++)
    {
        HIP_CHECK(hipStreamDestroy(streams[i]))
    }

    // Free device memory
    for(int i = 0; i < num_streams; i++)
    {
        HIP_CHECK(hipFree(d_in[i]));
        HIP_CHECK(hipFree(d_transpose_matrix[i]));
    }
}

int main()
{
    // Dimension of the input square matrix is width x width
    constexpr unsigned int width = 32;
    constexpr unsigned int size  = width * width;

    // Number of streams to be used. It is hardcoded to 2 as this example demonstrates
    // only two kernel launches and their management.
    constexpr unsigned int num_streams = 2;

    // Size in bytes for memory management
    const size_t size_in_bytes = sizeof(float) * size;

    // Allocate host input and output memory as pinned memory using hipHostMalloc.
    // It will ensure that the memory copies will be performed
    // asynchronously when using hipMemcpyAsync

    // Host input memory
    float* h_in = nullptr;
    HIP_CHECK(hipHostMalloc(&h_in, size_in_bytes));

    // Here we use two streams therefore declare two separate output storage structures
    // one for each stream.

    // Host output memory
    std::vector<float*> h_transpose_matrix(num_streams);

    // Allocate host output memory
    HIP_CHECK(hipHostMalloc(&h_transpose_matrix[0], size_in_bytes));
    HIP_CHECK(hipHostMalloc(&h_transpose_matrix[1], size_in_bytes));

    // Initialize the host input matrix
    for(unsigned int i = 0; i < size; i++)
    {
        h_in[i] = static_cast<float>(i);
    }

    deploy_multiple_stream<width, size>(h_in, h_transpose_matrix, num_streams);

    // Free host memory
    HIP_CHECK(hipHostFree(h_in));
    for(unsigned int i = 0; i < num_streams; i++)
    {

        HIP_CHECK(hipHostFree(h_transpose_matrix[i]));
    }

    std::cout << "streams completed!" << std::endl;
}
