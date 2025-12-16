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

int main()
{
    std::vector<float> matrix_in;
    std::vector<float> matrix_out;

    matrix_in.resize(width * height);
    matrix_out.resize(width * height);

    for(int i = 0; i < width * height; i++)
    {
        matrix_in[i] = (float)rand() / (float)RAND_MAX;
    }

    // printf("\\nInput matrix:\\n");
    // for (int i = 0; i < height; i++) {
    //   for (int j = 0; j < width; j++) {
    //     printf("%.2f ", matrix_in[i * width + j]);
    //   }
    //   printf("\\n");
    // }

    float* d_in;
    float* d_out;

    hipMalloc((void**)&d_in, width * height * sizeof(float));
    hipMalloc((void**)&d_out, width * height * sizeof(float));

    hipMemcpy(d_in, matrix_in.data(), width * height * sizeof(float), hipMemcpyHostToDevice);

    int block_x = width / tile_dim;
    int block_y = height / tile_dim;

    transpose_naive_kernel<<<dim3(block_x, block_y), dim3(tile_dim, tile_dim)>>>(d_in,
                                                                                 d_out,
                                                                                 width,
                                                                                 height);
    transpose_lds_kernel<<<dim3(block_x, block_y), dim3(tile_dim, tile_dim)>>>(d_in,
                                                                               d_out,
                                                                               width,
                                                                               height);

    hipMemcpy(matrix_out.data(), d_out, width * height * sizeof(float), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    // printf("\\nOutput matrix:\\n");
    // for (int i = 0; i < height; i++) {
    //   for (int j = 0; j < width; j++) {
    //     printf("%.2f ", matrix_out[i*width + j]);
    //   }
    //   printf("\\n");
    // }

    hipFree(d_in);
    hipFree(d_out);

    return 0;
}
