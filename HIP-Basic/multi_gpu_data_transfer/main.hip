// MIT License
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cmath>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

/// \brief Checks whether peer-to-peer is supported or not among the current available devices.
/// Returns, if exist, the IDs of the first two devices found with peer-to-peer memory access.
std::pair<int, int> check_peer_to_peer_support()
{
    // Get number of GPUs available.
    int gpu_count, can_access_peer;
    HIP_CHECK(hipGetDeviceCount(&gpu_count));

    // If there are not enough devices (at least 2) peer-to-peer is not possible.
    if(gpu_count < 2)
    {
        std::cout << "Peer-to-peer application requires at least 2 GPU devices." << std::endl;
        exit(0);
    }

    // Check accessibility for each device available.
    for(int current_gpu = 0; current_gpu < gpu_count; current_gpu++)
    {
        // Check if current_gpu device can access the memory of the devices with lower ID.
        for(int peer_gpu = 0; peer_gpu < current_gpu; peer_gpu++)
        {
            HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, current_gpu, peer_gpu));

            // The first pair found with peer-to-peer memory access is returned.
            if(can_access_peer)
            {
                return std::pair<int, int>(current_gpu, peer_gpu);
            }
        }
    }
    // No pair of devices supporting peer-to-peer between them has been found.
    std::cout << "Peer-to-peer application requires at least 2 GPU devices accessible between them."
              << std::endl;
    exit(0);
}

/// \brief Enables (if possible) direct memory access from <tt>current_gpu<\tt> to <tt>peer_gpu<\tt>.
void enable_peer_to_peer(const int current_gpu, const int peer_gpu)
{
    // Must be on a multi-gpu system.
    if(current_gpu == peer_gpu)
    {
        std::cerr << "Current and peer devices must be different." << std::endl;
        exit(error_exit_code);
    }

    // Set current GPU as default device for subsequent API calls.
    HIP_CHECK(hipSetDevice(current_gpu));

    // Enable direct memory access from current to peer device.
    HIP_CHECK(hipDeviceEnablePeerAccess(peer_gpu, 0 /*flags*/));
}

/// \brief Disables (if possible) direct memory access from <tt>current_gpu<\tt> to <tt>peer_gpu<\tt>.
void disable_peer_to_peer(const unsigned int current_gpu, const unsigned int peer_gpu)
{
    // Must be on a multi-gpu system.
    if(current_gpu == peer_gpu)
    {
        std::cerr << "Current and peer devices must be different." << std::endl;
        exit(error_exit_code);
    }

    // Set current GPU as default device for subsequent API calls.
    HIP_CHECK(hipSetDevice(current_gpu));

    // Disable direct memory access from current to peer device.
    HIP_CHECK(hipDeviceDisablePeerAccess(peer_gpu));
}

/// \brief Simple matrix transpose kernel using static shared memory.
template<const unsigned int Width = 32, const unsigned int Height = 32>
__global__ void static_shared_matrix_transpose_kernel(float* out, const float* in)
{
    // Allocate the necessary amount of shared memory to store the transpose of the matrix.
    // Note that the amount of shared memory needed is known at compile time.
    constexpr unsigned int size = Width * Height;
    __shared__ float       shared_matrix_memory[size];

    // Compute the row and column indexes of the matrix element that each thread is going
    // to process.
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    // If out of bounds, do nothing.
    if(!(x < Width && y < Height))
    {
        return;
    }

    // Store transposed element (x,y) in shared memory.
    shared_matrix_memory[y * Width + x] = in[x * Height + y];

    // Synchronize threads so all writes are done before accessing shared memory again.
    __syncthreads();

    // Copy transposed element from shared to global memory (output matrix).
    out[y * Width + x] = shared_matrix_memory[y * Width + x];
}

/// \brief Simple matrix transpose kernel using dynamic shared memory.
__global__ void dynamic_shared_matrix_transpose_kernel(float*             out,
                                                       const float*       in,
                                                       const unsigned int width,
                                                       const unsigned int height)
{
    // Declare that this kernel is using dynamic shared memory to store a number of floats.
    // The unsized array type indicates that the total amount of memory that is going
    // to be used here is not known ahead of time, and will be computed at runtime and
    // passed to the kernel launch function.
    extern __shared__ float shared_matrix_memory[];

    // Compute the row and column indexes of the matrix element that each thread is going
    // to process.
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    // If out of bounds, do nothing.
    if(!(x < width && y < height))
    {
        return;
    }

    // Store transposed element (x,y) in shared memory.
    shared_matrix_memory[y * width + x] = in[x * height + y];

    // Synchronize threads so all writes are done before accessing shared memory again.
    __syncthreads();

    // Copy transposed element from shared to global memory (output matrix).
    out[y * width + x] = shared_matrix_memory[y * height + x];
}

int main()
{
    // Check peer-to-peer access for all devices and get the IDs of the first pair (if exist)
    // that support peer-to-peer memory access.
    std::pair<int, int> gpus = check_peer_to_peer_support();

    std::cout << "Devices with IDs " << gpus.first << " and " << gpus.second << " selected."
              << std::endl;

    // Number of rows and columns, total number of elements and size in bytes of the matrix
    // to be transposed.
    constexpr unsigned int width      = 4;
    constexpr unsigned int height     = width;
    constexpr unsigned int size       = width * height;
    constexpr size_t       size_bytes = size * sizeof(float);

    // Number of threads in each dimension of the kernel block.
    constexpr unsigned int block_size = 4;

    // Number of blocks in each dimension of the grid. Calculated as
    // ceiling(matrix_dimension/block_size) with matrix_dimension being width or height.
    constexpr unsigned int grid_size_x = ceiling_div(width, block_size);
    constexpr unsigned int grid_size_y = ceiling_div(height, block_size);

    // Block and grid sizes in 2D.
    const dim3 block_dim(block_size, block_size);
    const dim3 grid_dim(grid_size_x, grid_size_y);

    // Allocate host input matrix and initialize with increasing sequence 1, 2, 3, ....
    std::vector<float> matrix(size);
    std::iota(matrix.begin(), matrix.end(), 1.f);

    // Allocate host matrix to store the results of the kernel execution on the second device.
    std::vector<float> transposed_matrix(size, 0.f);

    // Declare input and output matrices for the executions on both devices.
    float* d_matrix[2]{};
    float* d_transposed_matrix[2]{};

    // Set first gpu as default device for subsequent API calls.
    HIP_CHECK(hipSetDevice(gpus.first));

    // Allocate input and output matrices on current device.
    HIP_CHECK(hipMalloc(&d_transposed_matrix[0], size_bytes));
    HIP_CHECK(hipMalloc(&d_matrix[0], size_bytes));

    // Copy input matrix data from host to current device.
    HIP_CHECK(hipMemcpy(d_matrix[0], matrix.data(), size_bytes, hipMemcpyHostToDevice));

    std::cout << "Computing matrix transpose on device " << gpus.first << "." << std::endl;

    // Launch kernel in current device. Note that, as this kernel uses static shared memory, no
    // bytes of shared memory need to be allocated when launching the kernel.
    static_shared_matrix_transpose_kernel<width, height>
        <<<grid_dim, block_dim, 0 /*shared_memory_bytes*/, hipStreamDefault>>>(
            d_transposed_matrix[0],
            d_matrix[0]);

    // Wait on all active streams on the current device.
    HIP_CHECK(hipDeviceSynchronize());

    // Set second gpu as default device for subsequent API calls.
    HIP_CHECK(hipSetDevice(gpus.second));

    // Allocate input and output matrices on current device.
    HIP_CHECK(hipMalloc(&d_transposed_matrix[1], size_bytes));
    HIP_CHECK(hipMalloc(&d_matrix[1], size_bytes));

    std::cout << "Transferring results from device " << gpus.first << " to device " << gpus.second
              << "." << std::endl;

    // Enable (if possible) direct memory access from current (second) to peer (first) GPU.
    enable_peer_to_peer(gpus.second, gpus.first);

    // Copy output matrix from peer device to input matrix on current device. This copy is made
    // directly between devices (no host needed) because direct access memory was previously
    // enabled from second to first device.
    HIP_CHECK(hipMemcpy(d_matrix[1], d_transposed_matrix[0], size_bytes, hipMemcpyDeviceToDevice));

    std::cout << "Computing matrix transpose on device " << gpus.second << "." << std::endl;

    // Launch kernel in current device. Note that size_bytes bytes of shared memory are required to launch
    // this kernel because it uses dynamically allocated shared memory.
    dynamic_shared_matrix_transpose_kernel<<<grid_dim,
                                             block_dim,
                                             size_bytes /*shared_memory_bytes*/,
                                             hipStreamDefault>>>(d_transposed_matrix[1],
                                                                 d_matrix[1],
                                                                 width,
                                                                 height);

    // Wait on all active streams on the current device.
    HIP_CHECK(hipDeviceSynchronize());

    // Copy results from second device to host.
    HIP_CHECK(hipMemcpy(transposed_matrix.data(),
                        d_transposed_matrix[1],
                        size_bytes,
                        hipMemcpyDeviceToHost));

    // Disable direct memory access.
    disable_peer_to_peer(gpus.second, gpus.first);

    // Free device memory.
    for(unsigned int i = 0; i < 2; i++)
    {
        HIP_CHECK(hipFree(d_matrix[i]));
        HIP_CHECK(hipFree(d_transposed_matrix[i]));
    }

    // Validate results. The input matrix for the kernel execution on the first device must be
    // the same as the output matrix from the kernel execution on the second device.
    unsigned int    errors = 0;
    constexpr float eps    = 1.0E-6f;
    std::cout << "Validating peer-to-peer." << std::endl;
    for(unsigned int i = 0; i < size; i++)
    {
        errors += (std::fabs(matrix[i] - transposed_matrix[i]) > eps);
    }
    if(errors)
    {
        std::cout << "Validation failed with " << errors << " errors." << std::endl;
        return error_exit_code;
    }
    else
    {
        std::cout << "Validation passed." << std::endl;
    }
}
