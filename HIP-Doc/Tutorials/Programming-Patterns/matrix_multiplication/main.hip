// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cstdlib>
#include <iostream>

#define N 32
#define BLOCK_SIZE 16

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

// CPU matrix multiplication for verification
void cpu_matrix_multiplication(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// GPU matrix multiplication kernel
__global__ void gpu_matrix_multiplication(float *a, float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    if (row < n && col < n) {
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    // Allocate host memory
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    h_a = (float*)malloc(sizeof(float) * N * N);
    h_b = (float*)malloc(sizeof(float) * N * N);
    h_c_cpu = (float*)malloc(sizeof(float) * N * N);
    h_c_gpu = (float*)malloc(sizeof(float) * N * N);
    
    // Initialize matrix A
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_a[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    
    // Initialize matrix B
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_b[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    
    // Compute CPU result
    std::cout << "Computing matrix multiplication on CPU..." << std::endl;
    cpu_matrix_multiplication(h_a, h_b, h_c_cpu, N);
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc((void**)&d_a, sizeof(float) * N * N));
    HIP_CHECK(hipMalloc((void**)&d_b, sizeof(float) * N * N));
    HIP_CHECK(hipMalloc((void**)&d_c, sizeof(float) * N * N));
    
    // Copy data to device
    HIP_CHECK(hipMemcpy(d_a, h_a, sizeof(float) * N * N, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, sizeof(float) * N * N, hipMemcpyHostToDevice));
    
    // Launch kernel
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    int n_blocks = static_cast<int>(ceil(static_cast<float>(N) / BLOCK_SIZE));
    dim3 blocksPerGrid(n_blocks, n_blocks);
    
    std::cout << "Computing matrix multiplication on GPU..." << std::endl;
    hipLaunchKernelGGL(gpu_matrix_multiplication,
                       blocksPerGrid,
                       threadsPerBlock,
                       0, 0,
                       d_a, d_b, d_c, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_c_gpu, d_c, sizeof(float) * N * N, hipMemcpyDeviceToHost));
    
    // Compare CPU and GPU results
    bool passed = true;
    float tolerance = 1e-3f;
    for (int i = 0; i < N * N; ++i) {
        if (std::abs(h_c_cpu[i] - h_c_gpu[i]) > tolerance) {
            passed = false;
            std::cerr << "Validation failed! CPU: " << h_c_cpu[i]
                      << " GPU: " << h_c_gpu[i] << " at index: " << i << std::endl;
            break;
        }
    }
    
    if (passed) {
        std::cout << "Validation passed! CPU and GPU results match." << std::endl;
    } else {
        std::cerr << "Validation failed! CPU and GPU results do not match." << std::endl;
    }
    
    // Free device memory
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    
    return 0;
}
