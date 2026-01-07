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

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define HIP_CHECK(expression)                \
{                                            \
    const hipError_t status = expression;    \
    if(status != hipSuccess)                 \
    {                                        \
            std::cerr << "HIP error "        \
                << status << ": "            \
                << hipGetErrorString(status) \
                << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;    \
    }                                        \
}

const static int width    = 8192;
const static int height   = 8192;
const static int tile_dim = 16;

// [Sphinx HIP matrix transpose simple kernel start]
// Simple matrix transpose kernel
__global__ void transpose_naive_kernel(float* in, float* out, int width, int height)
{
    int x_index = blockIdx.x * tile_dim + threadIdx.x;
    int y_index = blockIdx.y * tile_dim + threadIdx.y;

    int in_index  = y_index * width + x_index;
    int out_index = x_index * height + y_index;

    out[out_index] = in[in_index];
}
// [Sphinx HIP matrix transpose simple kernel end]

// [Sphinx HIP matrix transpose LDS kernel start]
// LDS-based matrix transpose kernel for better memory coalescing
__global__ void transpose_lds_kernel(float* in, float* out, int width, int height)
{
    __shared__ float tile[tile_dim][tile_dim];

    int x_tile_index = blockIdx.x * tile_dim;
    int y_tile_index = blockIdx.y * tile_dim;

    int in_index  = (y_tile_index + threadIdx.y) * width + (x_tile_index + threadIdx.x);
    int out_index = (x_tile_index + threadIdx.y) * height + (y_tile_index + threadIdx.x);

    tile[threadIdx.y][threadIdx.x] = in[in_index];

    __syncthreads();

    out[out_index] = tile[threadIdx.x][threadIdx.y];
}
// [Sphinx HIP matrix transpose LDS kernel end]

// Helper function to compare floating-point values with tolerance
bool are_close(float a, float b, float rel_tol = 1e-5f, float abs_tol = 1e-8f)
{
    return std::abs(a - b) <= std::max(rel_tol * std::max(std::abs(a), std::abs(b)), abs_tol);
}

// Verify results between two output arrays
bool verify_results(const std::vector<float>& naive_out,
                    const std::vector<float>& lds_out,
                    int& mismatch_count,
                    float& max_diff)
{
    mismatch_count = 0;
    max_diff       = 0.0f;

    for(size_t i = 0; i < naive_out.size(); i++)
    {
        if(!are_close(naive_out[i], lds_out[i]))
        {
            mismatch_count++;
            float diff = std::abs(naive_out[i] - lds_out[i]);
            max_diff   = std::max(max_diff, diff);
        }
    }

    return mismatch_count == 0;
}

int main()
{
    std::cout << "Matrix Transpose Example\n";
    std::cout << "Matrix size: " << width << "x" << height << "\n";
    std::cout << "Tile dimension: " << tile_dim << "x" << tile_dim << "\n\n";

    // Host memory allocation
    std::vector<float> matrix_in(width * height);
    std::vector<float> matrix_out_naive(width * height);
    std::vector<float> matrix_out_lds(width * height);

    // Initialize input matrix with random values
    for(int i = 0; i < width * height; i++)
    {
        matrix_in[i] = (float)rand() / (float)RAND_MAX;
    }

    // Device memory allocation
    float* d_in;
    float* d_out_naive;
    float* d_out_lds;

    HIP_CHECK(hipMalloc((void**)&d_in, width * height * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_out_naive, width * height * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_out_lds, width * height * sizeof(float)));

    // Copy input data to device
    HIP_CHECK(
        hipMemcpy(d_in, matrix_in.data(), width * height * sizeof(float), hipMemcpyHostToDevice));

    // Kernel launch configuration
    int block_x = width / tile_dim;
    int block_y = height / tile_dim;
    dim3 grid_dim(block_x, block_y);
    dim3 block_dim(tile_dim, tile_dim);

    // Create events for timing
    hipEvent_t start_naive, stop_naive, start_lds, stop_lds;
    HIP_CHECK(hipEventCreate(&start_naive));
    HIP_CHECK(hipEventCreate(&stop_naive));
    HIP_CHECK(hipEventCreate(&start_lds));
    HIP_CHECK(hipEventCreate(&stop_lds));

    // Run naive kernel with timing
    HIP_CHECK(hipEventRecord(start_naive, 0));
    transpose_naive_kernel<<<grid_dim, block_dim>>>(d_in, d_out_naive, width, height);
    HIP_CHECK(hipEventRecord(stop_naive, 0));
    HIP_CHECK(hipEventSynchronize(stop_naive));

    // Run LDS kernel with timing
    HIP_CHECK(hipEventRecord(start_lds, 0));
    transpose_lds_kernel<<<grid_dim, block_dim>>>(d_in, d_out_lds, width, height);
    HIP_CHECK(hipEventRecord(stop_lds, 0));
    HIP_CHECK(hipEventSynchronize(stop_lds));

    // Calculate elapsed times
    float time_naive_ms, time_lds_ms;
    HIP_CHECK(hipEventElapsedTime(&time_naive_ms, start_naive, stop_naive));
    HIP_CHECK(hipEventElapsedTime(&time_lds_ms, start_lds, stop_lds));

    // Copy results back to host
    HIP_CHECK(hipMemcpy(matrix_out_naive.data(),
                        d_out_naive,
                        width * height * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(matrix_out_lds.data(),
                        d_out_lds,
                        width * height * sizeof(float),
                        hipMemcpyDeviceToHost));

    // Verify results
    int mismatch_count;
    float max_diff;
    bool results_match = verify_results(matrix_out_naive, matrix_out_lds, mismatch_count, max_diff);

    // Display verification results
    std::cout << "Verification: ";
    if(results_match)
    {
        std::cout << "PASSED - Results match within tolerance\n";
    }
    else
    {
        std::cout << "FAILED - " << mismatch_count << " mismatches found\n";
        std::cout << "Maximum difference: " << max_diff << "\n";
    }

    // Display performance results
    std::cout << "\nPerformance:\n";
    std::cout << "Naive kernel time: " << time_naive_ms << " ms\n";
    std::cout << "LDS kernel time:   " << time_lds_ms << " ms\n";
    std::cout << "Speedup:           " << (time_naive_ms / time_lds_ms) << "x\n";

    // Cleanup
    HIP_CHECK(hipEventDestroy(start_naive));
    HIP_CHECK(hipEventDestroy(stop_naive));
    HIP_CHECK(hipEventDestroy(start_lds));
    HIP_CHECK(hipEventDestroy(stop_lds));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out_naive));
    HIP_CHECK(hipFree(d_out_lds));

    return 0;
}
