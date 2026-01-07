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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#define HIP_CHECK(expression)                      \
    {                                              \
        const hipError_t status = expression;      \
        if (status != hipSuccess)                  \
        {                                          \
            std::cerr << "HIP error "              \
                      << status << ": "            \
                      << hipGetErrorString(status) \
                      << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;    \
        }                                          \
    }

// Matrix dimensions
constexpr int M = 8192;
constexpr int N = 8192;
constexpr int K = 8192;

// [Sphinx HIP matrix multiplication naive kernel start]
// Naive matrix multiplication kernel
__global__ void matrix_multiply_naive(float *a, float *b, float *out, int m, int n, int k)
{
    int gid_x = blockDim.x * blockIdx.x + threadIdx.x;
    int gid_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (gid_x < n && gid_y < m)
    {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i)
        {
            sum += a[gid_y * k + i] * b[i * n + gid_x];
        }
        out[gid_y * n + gid_x] = sum;
    }
}
// [Sphinx HIP matrix multiplication naive kernel end]

// [Sphinx HIP matrix multiplication LDS tiling kernel start]
// LDS tiling parameters
constexpr int base_tile_size = 16; // LDS tile dimension

// LDS tiling kernel
__global__ void matrix_multiply_lds_tiling(float *a, float *b, float *out, int m, int n, int k)
{
    __shared__ float tilea[base_tile_size][base_tile_size];
    __shared__ float tileb[base_tile_size][base_tile_size];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * base_tile_size + ty;
    int col = blockIdx.x * base_tile_size + tx;

    float sum = 0.0f;

    int numtiles = (k + base_tile_size - 1) / base_tile_size;
    for (int t = 0; t < numtiles; t++)
    {
        int acol = t * base_tile_size + tx;
        if (row < m && acol < k)
        {
            tilea[ty][tx] = a[row * k + acol];
        }
        else
        {
            tilea[ty][tx] = 0.0f;
        }

        int brow = t * base_tile_size + ty;
        if (brow < k && col < n)
        {
            tileb[ty][tx] = b[brow * n + col];
        }
        else
        {
            tileb[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < base_tile_size; i++)
        {
            sum += tilea[ty][i] * tileb[i][tx];
        }

        __syncthreads();
    }

    if (row < m && col < n)
    {
        out[row * n + col] = sum;
    }
}
// [Sphinx HIP matrix multiplication LDS tiling kernel end]

#if defined(__GFX8__) || defined(__GFX9__)
#define warp_size 64
#else
#define warp_size 32
#endif

// [Sphinx HIP matrix multiplication register tiling kernel start]
// Register tiling parameters
constexpr int thread_tile_m = 4; // Each thread computes 4x4 output
constexpr int thread_tile_n = 4;
constexpr int warp_threads_n = thread_tile_n; // 4
constexpr int block_warps_m = 2;
constexpr int block_warps_n = 2;
constexpr int k_tile_size = 16; // K-dimension tile size

// Register tiling kernel
__global__ void matrix_multiply_register_tiling(float *a, float *b, float *out, int m, int n, int k)
{
    constexpr int warp_threads_m = warp_size / warp_threads_n;
    constexpr int warp_tile_m = warp_threads_m * thread_tile_m;
    constexpr int warp_tile_n = warp_threads_n * thread_tile_n;
    constexpr int block_threads = warp_size * block_warps_m * block_warps_n;
    constexpr int block_tile_m = block_warps_m * warp_tile_m;
    constexpr int block_tile_n = block_warps_n * warp_tile_n;

    __shared__ float tilea[block_tile_m][k_tile_size + 4];
    __shared__ float tileb[k_tile_size][block_tile_n];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / warp_size;
    int lane_id = tid % warp_size;

    int warp_row = warp_id / block_warps_n;
    int warp_col = warp_id % block_warps_n;

    int lane_row = lane_id / warp_threads_n;
    int lane_col = lane_id % warp_threads_n;

    int thread_row_in_block = warp_row * warp_tile_m + lane_row * thread_tile_m;
    int thread_col_in_block = warp_col * warp_tile_n + lane_col * thread_tile_n;

    int block_row_start = blockIdx.y * block_tile_m;
    int block_col_start = blockIdx.x * block_tile_n;

    float regc[thread_tile_m][thread_tile_n] = {0.0f};

    int numtiles = (k + k_tile_size - 1) / k_tile_size;
    for (int t = 0; t < numtiles; t++)
    {
        int elements_to_load_a = (block_tile_m * k_tile_size) / block_threads;
        for (int i = 0; i < elements_to_load_a; i++)
        {
            int idx = tid + i * block_threads;
            int tile_m = idx / k_tile_size;
            int tile_n = idx % k_tile_size;

            int global_m = block_row_start + tile_m;
            int global_n = t * k_tile_size + tile_n;

            if (global_m < m && global_n < k && tile_m < block_tile_m)
            {
                tilea[tile_m][tile_n] = a[global_m * k + global_n];
            }
            else if (tile_m < block_tile_m)
            {
                tilea[tile_m][tile_n] = 0.0f;
            }
        }

        int elements_to_load_b = (k_tile_size * block_tile_n) / block_threads;
        for (int i = 0; i < elements_to_load_b; i++)
        {
            int idx = tid + i * block_threads;
            int tile_m = idx / block_tile_n;
            int tile_n = idx % block_tile_n;

            int global_m = t * k_tile_size + tile_m;
            int global_n = block_col_start + tile_n;

            if (global_m < k && global_n < n && tile_m < k_tile_size)
            {
                tileb[tile_m][tile_n] = b[global_m * n + global_n];
            }
            else if (tile_m < k_tile_size)
            {
                tileb[tile_m][tile_n] = 0.0f;
            }
        }

        __syncthreads();

        for (int kk = 0; kk < k_tile_size; kk++)
        {
            float rega[thread_tile_m];
            for (int i = 0; i < thread_tile_m; i++)
            {
                rega[i] = tilea[thread_row_in_block + i][kk];
            }

            float regb[thread_tile_n];
            for (int j = 0; j < thread_tile_n; j++)
            {
                regb[j] = tileb[kk][thread_col_in_block + j];
            }

            for (int i = 0; i < thread_tile_m; i++)
            {
                for (int j = 0; j < thread_tile_n; j++)
                {
                    regc[i][j] += rega[i] * regb[j];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < thread_tile_m; i++)
    {
        for (int j = 0; j < thread_tile_n; j++)
        {
            int global_row = block_row_start + thread_row_in_block + i;
            int global_col = block_col_start + thread_col_in_block + j;

            if (global_row < m && global_col < n)
            {
                out[global_row * n + global_col] = regc[i][j];
            }
        }
    }
}
// [Sphinx HIP matrix multiplication register tiling kernel end]

int main()
{
    int deviceid = 0;
    int warpsizehost;
    HIP_CHECK(hipDeviceGetAttribute(&warpsizehost, hipDeviceAttributeWarpSize, deviceid));
    std::cout << "Warp size: " << warpsizehost << std::endl;

    std::vector<float> a(M * K);
    std::vector<float> b(K * N);
    std::vector<float> out(M * N);

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < M * K; ++i)
    {
        a[i] = dis(gen);
    }

    std::fill(b.begin(), b.end(), 0.0f);
    for (int i = 0; i < std::min(K, N); ++i)
    {
        b[i * N + i] = 1.0f;
    }

    float *d_a, *d_b, *d_out;
    HIP_CHECK(hipMalloc(&d_a, sizeof(float) * M * K));
    HIP_CHECK(hipMalloc(&d_b, sizeof(float) * K * N));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float) * M * N));

    HIP_CHECK(hipMemcpy(d_a, a.data(), sizeof(float) * M * K, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, b.data(), sizeof(float) * K * N, hipMemcpyHostToDevice));

    std::cout << "Running matrix multiplication kernels..." << std::endl;

    // Naive kernel
    dim3 block_naive(16, 16);
    dim3 grid_naive((N + block_naive.x - 1) / block_naive.x,
                    (M + block_naive.y - 1) / block_naive.y);

    matrix_multiply_naive<<<grid_naive, block_naive>>>(d_a, d_b, d_out, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float) * M * N, hipMemcpyDeviceToHost));

    // Verify naive kernel result
    bool passed = true;
    const float tolerance = 1e-3f;
    for (int i = 0; i < M * N; i++)
    {
        if (std::fabs(out[i] - a[i]) > tolerance)
        {
            passed = false;
            break;
        }
    }
    std::cout << "  Naive kernel: " << (passed ? "PASSED" : "FAILED") << std::endl;

    // LDS tiling kernel
    dim3 block_lds(base_tile_size, base_tile_size);
    dim3 grid_lds((N + base_tile_size - 1) / base_tile_size,
                  (M + base_tile_size - 1) / base_tile_size);

    matrix_multiply_lds_tiling<<<grid_lds, block_lds>>>(d_a, d_b, d_out, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float) * M * N, hipMemcpyDeviceToHost));

    // Verify LDS tiling kernel result
    passed = true;
    for (int i = 0; i < M * N; i++)
    {
        if (std::fabs(out[i] - a[i]) > tolerance)
        {
            passed = false;
            break;
        }
    }
    std::cout << "  LDS tiling kernel: " << (passed ? "PASSED" : "FAILED") << std::endl;

    // Register tiling kernel
    int block_threads = warpsizehost * block_warps_m * block_warps_n;
    dim3 block_reg(16, block_threads / 16);
    const int block_tile_n = block_warps_n * warp_threads_n * thread_tile_n;
    const int block_tile_m = block_warps_m * (warpsizehost / warp_threads_n) * thread_tile_m;
    dim3 grid_reg((N + block_tile_n - 1) / block_tile_n, (M + block_tile_m - 1) / block_tile_m);

    matrix_multiply_register_tiling<<<grid_reg, block_reg>>>(d_a, d_b, d_out, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float) * M * N, hipMemcpyDeviceToHost));

    // Verify register tiling kernel result
    passed = true;
    for (int i = 0; i < M * N; i++)
    {
        if (std::fabs(out[i] - a[i]) > tolerance)
        {
            passed = false;
            break;
        }
    }
    std::cout << "  Register tiling kernel: " << (passed ? "PASSED" : "FAILED") << std::endl;

    std::cout << "Execution completed successfully." << std::endl;

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_out));

    return EXIT_SUCCESS;
}
