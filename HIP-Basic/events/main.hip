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

#include <iostream>
#include <vector>

#include <cstdlib>

/// \brief Performs a simple matrix transpose on the GPU.
__global__ void matrix_transpose_kernel(float* out, const float* in, const unsigned int width)
{
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    out[y * width + x] = in[x * width + y];
}

/// \brief Performs a simple matrix transpose on the host.
std::vector<float> matrix_transpose_reference(const std::vector<float>& input,
                                              const unsigned int        width)
{
    std::vector<float> output(input.size());
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
    // Number of columns in the transposed square matrix.
    constexpr unsigned int width = 1024;

    // Number of threads in each kernel block along the X dimension
    constexpr unsigned int threads_per_block_x = 4;

    // Number of threads in each kernel block along the Y dimension
    constexpr unsigned int threads_per_block_y = 4;

    // Total element count of the transposed matrix.
    constexpr unsigned int size = width * width;

    // Total size (in bytes) of the transposed matrix.
    constexpr size_t size_bytes = sizeof(float) * size;

    // Allocate host vectors.
    std::vector<float> h_matrix(size);
    std::vector<float> h_transposed_matrix(size);

    // Set up input data.
    for(unsigned int i = 0; i < size; i++)
    {
        h_matrix[i] = i * 10.0f;
    }

    // Allocate device memory for the input and output matrices.
    float* d_matrix{};
    float* d_transposed_matrix{};
    HIP_CHECK(hipMalloc(&d_matrix, size_bytes));
    HIP_CHECK(hipMalloc(&d_transposed_matrix, size_bytes));

    // HIP events has to be initialized using hipEventCreate.
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Record the start event.
    HIP_CHECK(hipEventRecord(start, hipStreamDefault));

    // Memory transfer from host to device
    HIP_CHECK(hipMemcpy(d_matrix, h_matrix.data(), size_bytes, hipMemcpyHostToDevice));

    // Record and synchronize with the stop event.
    HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
    HIP_CHECK(hipEventSynchronize(stop));

    // Calculate and print the elapsed time between the start and the stop event.
    float elapsed_ms{};
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "hipMemcpyHostToDevice time = " << elapsed_ms << " ms" << std::endl;

    // Record the start event.
    HIP_CHECK(hipEventRecord(start, hipStreamDefault));

    // Lauching kernel from host.
    matrix_transpose_kernel<<<dim3(width / threads_per_block_x, width / threads_per_block_y),
                              dim3(threads_per_block_x, threads_per_block_y),
                              0,
                              hipStreamDefault>>>(d_transposed_matrix, d_matrix, width);

    // Record and synchronize with the stop event.
    HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
    HIP_CHECK(hipEventSynchronize(stop));

    // Calculate and print the elapsed time between the start and the stop event.
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "Kernel execution time      = " << elapsed_ms << " ms" << std::endl;

    // Record the start event.
    HIP_CHECK(hipEventRecord(start, hipStreamDefault));

    // Memory transfer from device to host.
    HIP_CHECK(hipMemcpy(h_transposed_matrix.data(),
                        d_transposed_matrix,
                        size_bytes,
                        hipMemcpyDeviceToHost));

    // Record and synchronize with the stop event.
    HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
    HIP_CHECK(hipEventSynchronize(stop));

    // Calculate and print the elapsed time between the start and the stop event.
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "hipMemcpyDeviceToHost time = " << elapsed_ms << " ms" << std::endl;

    // Free the resources on the device.
    HIP_CHECK(hipFree(d_matrix));
    HIP_CHECK(hipFree(d_transposed_matrix));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    // Perform the reference (CPU) calculation.
    const auto ref_transposed_matrix = matrix_transpose_reference(h_matrix, width);

    // Check the results' validity.
    constexpr float eps    = 1.0E-6F;
    unsigned int    errors = 0;
    for(unsigned int i = 0; i < size; i++)
    {
        // Most likely the values are bitwise equal, since they were plainly copied,
        // however it is a good practice to compare floating point values using an epsilon.
        if(std::abs(h_transposed_matrix[i] - ref_transposed_matrix[i]) > eps)
        {
            errors++;
        }
    }

    if(errors != 0)
    {
        std::cout << "Validation failed. Errors: " << errors << std::endl;
        return error_exit_code;
    }
    else
    {
        std::cout << "Validation passed." << std::endl;
    }
}
