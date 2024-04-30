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

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

/// \brief Transposes the matrix \p in and stores the result in \p out using static shared memory.
template<const unsigned int Width = 64>
__global__ void matrix_transpose_kernel(float* out, const float* in)
{
    // Allocate the necessary amount of shared memory to store the transpose of the matrix.
    constexpr unsigned int size = Width * Width;
    __shared__ float       shared_matrix_memory[size];

    // Compute the row and column indexes of the matrix element that each thread is going
    // to process.
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    // If not out of bounds, transpose element (x,y).
    if(x < Width && y < Width)
    {
        // Store transposed element in shared memory.
        shared_matrix_memory[y * Width + x] = in[x * Width + y];
    }

    // Syncronize threads so all writes are done before accessing shared memory again.
    __syncthreads();

    // If not out of bounds, transpose element (x,y).
    if(x < Width && y < Width)
    {
        // Copy transposed element from shared memory to global memory.
        out[y * Width + x] = shared_matrix_memory[y * Width + x];
    }
}

// CPU implementation of matrix transpose.
std::vector<float> expected_matrix_transpose(const std::vector<float>& input,
                                             const unsigned int        width)
{
    std::vector<float> output(width * width);
    for(unsigned int j = 0; j < width; j++)
    {
        for(unsigned int i = 0; i < width; i++)
        {
            output[i * width + j] = input[j * width + i];
        }
    }
    return output;
}

int main()
{
    // Number of rows and columns, total number of elements and size in bytes of the matrix
    // to be transposed.
    constexpr unsigned int width      = 64;
    constexpr unsigned int size       = width * width;
    constexpr unsigned int size_bytes = size * sizeof(float);

    // Number of threads in each dimension of the kernel block.
    constexpr unsigned int block_size = 4;

    // Number of blocks in each dimension of the grid. Calculated as ceil(width/block_size).
    constexpr unsigned int grid_size = ceiling_div(width, block_size);

    // Block and grid sizes in 2D.
    const dim3 block_dim(block_size, block_size);
    const dim3 grid_dim(grid_size, grid_size);

    // Allocate host input matrix and initialize with increasing sequence 10, 20, 30, ....
    std::vector<float> matrix(size);
    std::iota(matrix.begin(), matrix.end(), 1.f);
    std::for_each(matrix.begin(), matrix.end(), [](float& f) { f = 10.f * f; });

    // Allocate matrix to store the results of the kernel execution.
    std::vector<float> transposed_matrix(size);

    // Allocate input and output matrices on device.
    float* d_matrix{};
    float* d_transposed_matrix{};
    HIP_CHECK(hipMalloc(&d_matrix, size_bytes));
    HIP_CHECK(hipMalloc(&d_transposed_matrix, size_bytes));

    // Copy input matrix data from host to device.
    HIP_CHECK(hipMemcpy(d_matrix, matrix.data(), size_bytes, hipMemcpyHostToDevice));

    // Print trace message.
    std::cout << "Computing matrix transpose." << std::endl;

    // Launch kernel on the default stream. Passing kernel arguments at the end of the
    // <<<...>>> function call.
    matrix_transpose_kernel<width>
        <<<grid_dim, block_dim, 0, hipStreamDefault>>>(d_transposed_matrix, d_matrix);

    // Check if the kernel launch was successful.
    HIP_CHECK(hipGetLastError());

    // Copy results from device to host.
    HIP_CHECK(hipMemcpy(transposed_matrix.data(),
                        d_transposed_matrix,
                        size_bytes,
                        hipMemcpyDeviceToHost));

    // Free device memory.
    HIP_CHECK(hipFree(d_matrix));
    HIP_CHECK(hipFree(d_transposed_matrix));

    // Calculate expected transposed matrix with the CPU version of the kernel.
    std::vector<float> expected_transposed_matrix = expected_matrix_transpose(matrix, width);

    // Validate results comparing with expected transposed matrix.
    unsigned int    errors = 0;
    constexpr float eps    = 1.0E-6f;
    std::cout << "Validating transposed matrix." << std::endl;
    for(unsigned int i = 0; i < size; i++)
    {
        errors += (std::fabs(transposed_matrix[i] - expected_transposed_matrix[i]) > eps);
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
